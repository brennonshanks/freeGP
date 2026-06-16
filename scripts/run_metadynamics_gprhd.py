#!/usr/bin/env python3
"""Run GPR(H+D) on Lagrangian metadynamics COLVAR/FES files."""

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
from freegp.hyperopt import optimize_stationary_hyperparameters
from freegp.metadynamics import (
    build_metadynamics_joint_observations,
    load_colvar,
    load_fes,
    MetadynamicsReference,
    metad_reference_curves,
    process_lagrangian_metadynamics,
)
from freegp.metrics import compare_to_reference_curves
from freegp.posterior import (
    summarize_fixed_posterior_derivative,
    summarize_fixed_posterior_predictive,
)


FIXED_ELL = math.pi / 2.0
FIXED_W = 4.184 * math.sqrt(10.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--colvar", type=str, default="COLVAR")
    parser.add_argument("--fes", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results/metadynamics-gprhd")
    parser.add_argument("--force-constant", type=float, required=True)
    parser.add_argument("--interval", type=float, nargs=2, default=None)
    parser.add_argument("--time-fraction", type=float, default=1.0)
    parser.add_argument("--histogram-bin-width", type=float, default=None)
    parser.add_argument("--n-histogram-windows", type=int, default=40)
    parser.add_argument("--histogram-radius-bins", type=float, default=5.0)
    parser.add_argument("--derivative-bin-width", type=float, default=None)
    parser.add_argument("--n-derivative-bins", type=int, default=80)
    parser.add_argument("--min-window-samples", type=int, default=5)
    parser.add_argument("--min-derivative-samples", type=int, default=5)
    parser.add_argument("--num-test-points", type=int, default=400)
    parser.add_argument("--pmf-alignment", choices=("max", "min"), default="max")
    parser.add_argument("--mode", choices=("fixed", "map"), default="map")
    parser.add_argument("--ell", type=float, default=FIXED_ELL)
    parser.add_argument("--w", type=float, default=FIXED_W)
    parser.add_argument("--objective", choices=("lml", "loo"), default="loo")
    parser.add_argument("--opt-steps", type=int, default=500)
    parser.add_argument("--opt-restarts", type=int, default=5)
    parser.add_argument("--opt-learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _resolve_input_path(root: Path | None, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and root is not None:
        path = root / path
    return path.resolve()


def _shift(values: np.ndarray, alignment: str) -> np.ndarray:
    anchor = np.max(values) if alignment == "max" else np.min(values)
    return values - anchor


def _crop_reference(reference, interval: tuple[float, float]):
    if reference is None:
        return None
    mask = (reference.x >= interval[0]) & (reference.x <= interval[1])
    if not np.any(mask):
        return reference

    derivative = reference.derivative[mask] if reference.derivative is not None else None
    return MetadynamicsReference(
        x=reference.x[mask],
        pmf=reference.pmf[mask],
        derivative=derivative,
    )


def _plot_prediction(
    *,
    output_path: Path,
    title: str,
    summary,
    reference,
    alignment: str,
) -> None:
    x = summary.x_test.detach().cpu().numpy()
    mean = summary.mean.detach().cpu().numpy()
    std = np.sqrt(
        np.clip(summary.total_variance.detach().cpu().numpy(), 0.0, None)
    )
    shifted_mean = _shift(mean, alignment)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, shifted_mean, color="tab:red", linewidth=2.0, label="GP mean")
    ax.fill_between(
        x,
        shifted_mean - 2.0 * std,
        shifted_mean + 2.0 * std,
        color="tab:red",
        alpha=0.18,
        label="GP +/- 2 SD",
    )
    if reference is not None:
        ax.plot(
            reference.x,
            _shift(reference.pmf, alignment),
            color="black",
            linewidth=1.5,
            label="Metadynamics FES reference",
        )
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel("Aligned free energy [kJ/mol]")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_derivative(*, output_path: Path, summary, observations, reference) -> None:
    x = summary.x_test.detach().cpu().numpy()
    mean = summary.mean.detach().cpu().numpy()
    std = np.sqrt(
        np.clip(summary.total_variance.detach().cpu().numpy(), 0.0, None)
    )
    obs_x = observations.x_der.detach().cpu().numpy()
    obs_y = observations.dy_der.detach().cpu().numpy()
    obs_std = np.sqrt(
        np.clip(observations.noise_deriv_diag.detach().cpu().numpy(), 0.0, None)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        obs_x,
        obs_y,
        yerr=obs_std,
        color="black",
        linestyle="none",
        marker="o",
        markersize=3,
        capsize=2,
        alpha=0.65,
        label="Binned restraint-force derivative",
    )
    ax.plot(x, mean, color="tab:purple", linewidth=2.0, label="GP derivative mean")
    ax.fill_between(
        x,
        mean - 2.0 * std,
        mean + 2.0 * std,
        color="tab:purple",
        alpha=0.16,
        label="GP +/- 2 SD",
    )
    if reference is not None and reference.derivative is not None:
        ax.plot(
            reference.x,
            reference.derivative,
            color="tab:green",
            linewidth=1.5,
            label="Metadynamics FES derivative",
        )
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel("Free-energy derivative [kJ/mol/nm]")
    ax.set_title("Metadynamics derivative reconstruction")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parser().parse_args()
    configure_torch(seed=args.seed)
    root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    colvar_path = _resolve_input_path(root, args.colvar)
    fes_path = _resolve_input_path(root, args.fes)
    if colvar_path is None:
        raise ValueError("--colvar is required.")

    output_dir = Path(args.results_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = load_colvar(colvar_path)
    reference = load_fes(fes_path) if fes_path is not None and fes_path.is_file() else None
    if args.interval is None:
        lower = float(np.nanmin(trajectory.cv))
        upper = float(np.nanmax(trajectory.cv))
        interval = (lower, upper)
    else:
        interval = (float(args.interval[0]), float(args.interval[1]))
    reference_for_comparison = _crop_reference(reference, interval)

    processed = process_lagrangian_metadynamics(
        trajectory,
        force_constant=args.force_constant,
        interval=interval,
        histogram_bin_width=args.histogram_bin_width,
        n_histogram_windows=args.n_histogram_windows,
        histogram_radius_bins=args.histogram_radius_bins,
        derivative_bin_width=args.derivative_bin_width,
        n_derivative_bins=args.n_derivative_bins,
        time_fraction=args.time_fraction,
        min_window_samples=args.min_window_samples,
        min_derivative_samples=args.min_derivative_samples,
    )
    observations = build_metadynamics_joint_observations(
        processed,
        force_constant=args.force_constant,
    )
    x_test = torch.linspace(
        interval[0], interval[1], args.num_test_points, dtype=torch.float64
    )

    if args.mode == "map":
        optimized = optimize_stationary_hyperparameters(
            observations,
            objective=args.objective,
            steps=args.opt_steps,
            learning_rate=args.opt_learning_rate,
            restarts=args.opt_restarts,
            seed=args.seed,
        )
        posterior = optimized.posterior
        model_label = "MAP stationary GP"
        hyperparameters = {
            name: float(value.detach().cpu().item())
            for name, value in optimized.params.items()
        }
        objective_value = float(optimized.objective_value)
    else:
        posterior = build_joint_gp(
            x_func=observations.x_obs,
            y_func=observations.y_obs,
            x_der=observations.x_der,
            dy_der=observations.dy_der,
            ell=torch.tensor(args.ell, dtype=torch.float64),
            w=torch.tensor(args.w, dtype=torch.float64),
            noise_func_cov=observations.noise_func_cov,
            noise_deriv_diag=observations.noise_deriv_diag,
            H_func=observations.H_obs,
            jitter=1e-6,
        )
        model_label = "Fixed stationary GP"
        hyperparameters = {"ell": args.ell, "w": args.w}
        objective_value = None

    function_summary = summarize_fixed_posterior_predictive(posterior, x_test)
    derivative_summary = summarize_fixed_posterior_derivative(posterior, x_test)
    references = metad_reference_curves(reference_for_comparison)
    metrics = compare_to_reference_curves(
        function_summary,
        references,
        alignment=args.pmf_alignment,
    )

    _plot_prediction(
        output_path=output_dir / "metadynamics_gprhd_pmf.png",
        title=f"{model_label} metadynamics reconstruction",
        summary=function_summary,
        reference=reference_for_comparison,
        alignment=args.pmf_alignment,
    )
    _plot_derivative(
        output_path=output_dir / "metadynamics_gprhd_derivative.png",
        summary=derivative_summary,
        observations=observations,
        reference=reference_for_comparison,
    )

    with (output_dir / "metadynamics_observation_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n_pseudo_windows",
                "n_function_observations",
                "n_derivative_observations",
                "min_window_samples",
                "median_window_samples",
                "max_window_samples",
                "min_derivative_samples",
                "median_derivative_samples",
                "max_derivative_samples",
            ]
        )
        writer.writerow(
            [
                int(processed.window_centers.numel()),
                int(observations.x_obs.numel()),
                int(observations.x_der.numel()),
                float(processed.n_samples_per_window.min().item()),
                float(processed.n_samples_per_window.median().item()),
                float(processed.n_samples_per_window.max().item()),
                float(processed.derivative_sample_counts.min().item()),
                float(processed.derivative_sample_counts.median().item()),
                float(processed.derivative_sample_counts.max().item()),
            ]
        )

    summary = {
        "colvar_path": str(colvar_path),
        "fes_path": str(fes_path) if fes_path is not None else None,
        "mode": args.mode,
        "force_constant": args.force_constant,
        "interval": interval,
        "time_fraction": args.time_fraction,
        "n_pseudo_windows": int(processed.window_centers.numel()),
        "n_function_observations": int(observations.x_obs.numel()),
        "n_derivative_observations": int(observations.x_der.numel()),
        "hyperparameters": hyperparameters,
        "objective_value": objective_value,
        "pmf_alignment": args.pmf_alignment,
        "rmse_metad_fes": metrics.rmse_wham,
        "avg_total_std": metrics.avg_total_std,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
