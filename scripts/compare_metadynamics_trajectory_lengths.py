#!/usr/bin/env python3
"""Compare metadynamics and GP reconstructions versus trajectory length."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from freegp.config import configure_torch
from freegp.gp import build_joint_gp
from freegp.hmc import HyperPriorConfig, NUTSConfig, run_hmc_nuts
from freegp.hyperopt import optimize_stationary_hyperparameters
from freegp.metadynamics import (
    build_metadynamics_joint_observations,
    load_colvar,
    load_fes,
    process_lagrangian_metadynamics,
)
from freegp.posterior import (
    summarize_fixed_posterior_predictive,
    summarize_hyperposterior_predictive,
)


FIXED_ELL = math.pi / 2.0
FIXED_W = 4.184 * math.sqrt(10.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="/home/bshanks/freeGP-datasets/membranes/zuzka_metadynamics/metad",
    )
    parser.add_argument("--colvar", default="COLVAR")
    parser.add_argument("--hills", default="HILLS")
    parser.add_argument("--fes", default="fes.dat")
    parser.add_argument(
        "--results-dir", default="results/metadynamics-trajectory-length-comparison"
    )
    parser.add_argument("--force-constant", type=float, default=10000.0)
    parser.add_argument("--interval", type=float, nargs=2, default=(-1.0, 5.0))
    parser.add_argument(
        "--trajectory-fractions",
        default="0.05,0.10,0.25,0.50,1.00",
        help="Comma-separated fractions of the trajectory to retain.",
    )
    parser.add_argument("--n-histogram-windows", type=int, default=60)
    parser.add_argument("--n-derivative-bins", type=int, default=120)
    parser.add_argument("--histogram-radius-bins", type=float, default=5.0)
    parser.add_argument("--min-window-samples", type=int, default=10)
    parser.add_argument("--min-derivative-samples", type=int, default=10)
    parser.add_argument("--num-test-points", type=int, default=250)
    parser.add_argument("--pmf-alignment", choices=("max", "min"), default="max")
    parser.add_argument("--objective", choices=("lml", "loo"), default="loo")
    parser.add_argument("--opt-steps", type=int, default=300)
    parser.add_argument("--opt-restarts", type=int, default=3)
    parser.add_argument("--opt-learning-rate", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--predictive-samples", type=int, default=100)
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument(
        "--skip-hmc",
        action="store_true",
        help="Skip HMC-NUTS to make a quick MAP/fixed/standard comparison.",
    )
    parser.add_argument("--hills-chunk-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _parse_fractions(value: str) -> list[float]:
    fractions = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not fractions:
        raise ValueError("At least one trajectory fraction is required.")
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("Trajectory fractions must be in (0, 1].")
    return sorted(set(fractions))


def _shift(values: np.ndarray, alignment: str) -> np.ndarray:
    anchor = np.max(values) if alignment == "max" else np.min(values)
    return values - anchor


def _aligned_rmse(reference: np.ndarray, prediction: np.ndarray, alignment: str) -> float:
    return float(np.sqrt(np.mean((_shift(reference, alignment) - _shift(prediction, alignment)) ** 2)))


def _avg_variance(summary) -> float:
    values = summary.total_variance.detach().cpu().numpy()
    return float(np.clip(values, 0.0, None).mean())


def _avg_std(summary) -> float:
    values = summary.total_variance.detach().cpu().numpy()
    return float(np.sqrt(np.clip(values, 0.0, None)).mean())


def _load_hills(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments=("#", "@", ";"))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 5:
        raise ValueError(f"HILLS file {path} must have at least five columns.")
    return data


def _standard_well_tempered_fes(
    hills: np.ndarray,
    x_grid: np.ndarray,
    *,
    fraction: float,
    chunk_size: int,
) -> np.ndarray:
    n_hills = max(1, int(np.floor(hills.shape[0] * fraction)))
    selected = hills[:n_hills]
    centers = selected[:, 1]
    sigmas = selected[:, 2]
    heights = selected[:, 3]
    bias_factor = float(np.median(selected[:, 4]))
    if bias_factor <= 1.0:
        scale = 1.0
    else:
        scale = bias_factor / (bias_factor - 1.0)

    bias = np.zeros_like(x_grid, dtype=float)
    for start in range(0, n_hills, chunk_size):
        stop = min(start + chunk_size, n_hills)
        dx = (x_grid[None, :] - centers[start:stop, None]) / sigmas[start:stop, None]
        bias += np.sum(heights[start:stop, None] * np.exp(-0.5 * dx**2), axis=0)
    return -scale * bias


def _reference_on_grid(reference, x_grid: np.ndarray, interval: tuple[float, float]) -> np.ndarray:
    mask = (reference.x >= interval[0]) & (reference.x <= interval[1])
    if not np.any(mask):
        raise ValueError("FES reference does not overlap requested interval.")
    return np.interp(x_grid, reference.x[mask], reference.pmf[mask])


def _fit_gp_methods(args, trajectory, x_test: torch.Tensor, fraction: float):
    processed = process_lagrangian_metadynamics(
        trajectory,
        force_constant=args.force_constant,
        interval=(float(args.interval[0]), float(args.interval[1])),
        n_histogram_windows=args.n_histogram_windows,
        histogram_radius_bins=args.histogram_radius_bins,
        n_derivative_bins=args.n_derivative_bins,
        time_fraction=fraction,
        min_window_samples=args.min_window_samples,
        min_derivative_samples=args.min_derivative_samples,
    )
    observations = build_metadynamics_joint_observations(
        processed,
        force_constant=args.force_constant,
    )

    fixed_posterior = build_joint_gp(
        x_func=observations.x_obs,
        y_func=observations.y_obs,
        x_der=observations.x_der,
        dy_der=observations.dy_der,
        ell=torch.tensor(FIXED_ELL, dtype=torch.float64),
        w=torch.tensor(FIXED_W, dtype=torch.float64),
        noise_func_cov=observations.noise_func_cov,
        noise_deriv_diag=observations.noise_deriv_diag,
        H_func=observations.H_obs,
        jitter=1e-6,
    )
    fixed_summary = summarize_fixed_posterior_predictive(fixed_posterior, x_test)

    priors = HyperPriorConfig()
    optimized = optimize_stationary_hyperparameters(
        observations,
        priors=priors,
        objective=args.objective,
        steps=args.opt_steps,
        learning_rate=args.opt_learning_rate,
        restarts=args.opt_restarts,
        seed=args.seed,
    )
    map_summary = summarize_fixed_posterior_predictive(optimized.posterior, x_test)

    hmc_summary = None
    hmc_diagnostics = {}
    if not args.skip_hmc:
        nuts_config = NUTSConfig(
            num_samples=args.num_samples,
            warmup_steps=args.warmup_steps,
            target_accept_prob=args.target_accept_prob,
            objective=args.objective,
            kernel="stationary",
        )
        mcmc, samples = run_hmc_nuts(
            observations,
            priors=priors,
            config=nuts_config,
        )
        hmc_summary = summarize_hyperposterior_predictive(
            observations,
            samples,
            x_test,
            priors=priors,
            config=nuts_config,
            max_samples=args.predictive_samples,
        )
        diagnostics = getattr(mcmc.kernel, "_diagnostics", None)
        hmc_diagnostics = {
            "n_hmc_samples": int(samples["theta_ell"].shape[0]),
            "diagnostics_available": diagnostics is not None,
        }

    return {
        "observations": observations,
        "processed": processed,
        "fixed": fixed_summary,
        "map": map_summary,
        "hmc": hmc_summary,
        "map_hyperparameters": {
            name: float(value.detach().cpu().item())
            for name, value in optimized.params.items()
        },
        "map_objective_value": float(optimized.objective_value),
        "hmc_diagnostics": hmc_diagnostics,
    }


def _write_metrics(output_dir: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "fraction",
        "method",
        "rmse_kj_mol",
        "avg_variance",
        "avg_std_kj_mol",
        "n_function_observations",
        "n_derivative_observations",
        "ell",
        "w",
        "sigma_f",
        "sigma_d",
        "objective_value",
    ]
    with (output_dir / "trajectory_length_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _plot_fit_grid(
    output_dir: Path,
    *,
    x_grid: np.ndarray,
    reference: np.ndarray,
    fractions: list[float],
    predictions: dict[float, dict[str, object]],
    alignment: str,
) -> None:
    methods = [
        ("standard_metadynamics", "Standard metadynamics", "tab:gray"),
        ("fixed_gp", "Fixed-hyper GP", "tab:blue"),
        ("map_gp", "MAP-hyper GP", "tab:orange"),
        ("hmc_gp", "HMC-NUTS GP", "tab:green"),
    ]
    active_methods = [
        item
        for item in methods
        if any(predictions[fraction].get(item[0]) is not None for fraction in fractions)
    ]
    fig, axes = plt.subplots(
        len(fractions),
        len(active_methods),
        figsize=(4.2 * len(active_methods), 3.0 * len(fractions)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    ref_shifted = _shift(reference, alignment)
    for row_i, fraction in enumerate(fractions):
        for col_i, (key, title, color) in enumerate(active_methods):
            ax = axes[row_i, col_i]
            ax.plot(x_grid, ref_shifted, color="black", linewidth=1.6, label="Final FES")
            item = predictions[fraction][key]
            if key == "standard_metadynamics":
                values = _shift(item, alignment)
                ax.plot(x_grid, values, color=color, linewidth=2.0, label=title)
            else:
                summary = item
                mean = summary.mean.detach().cpu().numpy()
                std = np.sqrt(
                    np.clip(summary.total_variance.detach().cpu().numpy(), 0.0, None)
                )
                shifted = _shift(mean, alignment)
                ax.plot(x_grid, shifted, color=color, linewidth=2.0, label=title)
                ax.fill_between(
                    x_grid,
                    shifted - 2.0 * std,
                    shifted + 2.0 * std,
                    color=color,
                    alpha=0.16,
                )
            if row_i == 0:
                ax.set_title(title)
            if col_i == 0:
                ax.set_ylabel(f"{fraction:g} trajectory\nFree energy [kJ/mol]")
            if row_i == len(fractions) - 1:
                ax.set_xlabel("Position [nm]")
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "trajectory_length_fit_grid.png", dpi=200)
    plt.close(fig)


def _plot_metrics(output_dir: Path, rows: list[dict[str, object]]) -> None:
    labels = {
        "standard_metadynamics": "Standard metadynamics",
        "fixed_gp": "Fixed-hyper GP",
        "map_gp": "MAP-hyper GP",
        "hmc_gp": "HMC-NUTS GP",
    }
    colors = {
        "standard_metadynamics": "tab:gray",
        "fixed_gp": "tab:blue",
        "map_gp": "tab:orange",
        "hmc_gp": "tab:green",
    }
    methods = [method for method in labels if any(row["method"] == method for row in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        x = np.array([float(row["fraction"]) for row in method_rows])
        rmse = np.array([float(row["rmse_kj_mol"]) for row in method_rows])
        axes[0].plot(
            x,
            rmse,
            marker="o",
            linewidth=2.0,
            color=colors[method],
            label=labels[method],
        )
        variances = np.array(
            [
                float(row.get("avg_variance"))
                if row.get("avg_variance") not in ("", None)
                else np.nan
                for row in method_rows
            ]
        )
        if np.any(np.isfinite(variances)):
            axes[1].plot(
                x,
                variances,
                marker="o",
                linewidth=2.0,
                color=colors[method],
                label=labels[method],
            )
    axes[0].set_xlabel("Trajectory fraction")
    axes[0].set_ylabel("RMSE vs final FES [kJ/mol]")
    axes[1].set_xlabel("Trajectory fraction")
    axes[1].set_ylabel("Average predictive variance [(kJ/mol)$^2$]")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_length_metrics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parser().parse_args()
    configure_torch(seed=args.seed)
    root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.results_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fractions = _parse_fractions(args.trajectory_fractions)
    interval = (float(args.interval[0]), float(args.interval[1]))
    x_np = np.linspace(interval[0], interval[1], args.num_test_points)
    x_test = torch.tensor(x_np, dtype=torch.float64)

    trajectory = load_colvar(_resolve(root, args.colvar))
    hills = _load_hills(_resolve(root, args.hills))
    reference = load_fes(_resolve(root, args.fes))
    reference_values = _reference_on_grid(reference, x_np, interval)

    predictions: dict[float, dict[str, object]] = {}
    metric_rows: list[dict[str, object]] = []
    run_metadata = {
        "data_root": str(root),
        "fractions": fractions,
        "interval": interval,
        "force_constant": args.force_constant,
        "objective": args.objective,
        "skip_hmc": args.skip_hmc,
    }

    for fraction in fractions:
        print(f"fraction {fraction:g}", flush=True)
        standard = _standard_well_tempered_fes(
            hills,
            x_np,
            fraction=fraction,
            chunk_size=args.hills_chunk_size,
        )
        gp_results = _fit_gp_methods(args, trajectory, x_test, fraction)
        observations = gp_results["observations"]
        n_function = int(observations.x_obs.numel())
        n_derivative = int(observations.x_der.numel())

        predictions[fraction] = {
            "standard_metadynamics": standard,
            "fixed_gp": gp_results["fixed"],
            "map_gp": gp_results["map"],
            "hmc_gp": gp_results["hmc"],
        }
        metric_rows.append(
            {
                "fraction": fraction,
                "method": "standard_metadynamics",
                "rmse_kj_mol": _aligned_rmse(
                    reference_values, standard, args.pmf_alignment
                ),
                "n_function_observations": n_function,
                "n_derivative_observations": n_derivative,
            }
        )
        for method, summary in (
            ("fixed_gp", gp_results["fixed"]),
            ("map_gp", gp_results["map"]),
            ("hmc_gp", gp_results["hmc"]),
        ):
            if summary is None:
                continue
            row = {
                "fraction": fraction,
                "method": method,
                "rmse_kj_mol": _aligned_rmse(
                    reference_values,
                    summary.mean.detach().cpu().numpy(),
                    args.pmf_alignment,
                ),
                "avg_variance": _avg_variance(summary),
                "avg_std_kj_mol": _avg_std(summary),
                "n_function_observations": n_function,
                "n_derivative_observations": n_derivative,
            }
            if method == "map_gp":
                row.update(gp_results["map_hyperparameters"])
                row["objective_value"] = gp_results["map_objective_value"]
            metric_rows.append(row)

    _write_metrics(output_dir, metric_rows)
    _plot_fit_grid(
        output_dir,
        x_grid=x_np,
        reference=reference_values,
        fractions=fractions,
        predictions=predictions,
        alignment=args.pmf_alignment,
    )
    _plot_metrics(output_dir, metric_rows)

    with (output_dir / "run_summary.json").open("w") as handle:
        json.dump(run_metadata, handle, indent=2)
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
