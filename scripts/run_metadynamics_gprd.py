#!/usr/bin/env python3
"""Run derivative-only MTD/ICF/GPR on Lagrangian metadynamics data."""

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
from freegp.gp import (
    build_derivative_gp,
    predict_derivative_gp_derivative,
    predict_derivative_gp_function,
)
from freegp.hmc import HyperPriorConfig
from freegp.hyperopt import optimize_derivative_hyperparameters
from freegp.metadynamics import (
    load_colvar,
    load_fes,
    process_metadynamics_icf_derivatives,
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
    parser.add_argument("--fes", default="fes.dat")
    parser.add_argument("--results-dir", default="results/metadynamics-gprd")
    parser.add_argument("--force-constant", type=float, default=10000.0)
    parser.add_argument("--interval", type=float, nargs=2, default=(-1.0, 5.0))
    parser.add_argument("--time-fraction", type=float, default=1.0)
    parser.add_argument(
        "--derivative-binning",
        choices=("uniform", "quantile"),
        default="uniform",
        help="Use uniform-width bins or quantile bins with approximately equal sample counts.",
    )
    parser.add_argument("--derivative-bin-width", type=float, default=None)
    parser.add_argument("--n-derivative-bins", type=int, default=120)
    parser.add_argument("--min-derivative-samples", type=int, default=10)
    parser.add_argument("--num-test-points", type=int, default=300)
    parser.add_argument("--pmf-alignment", choices=("max", "min"), default="max")
    parser.add_argument("--mode", choices=("fixed", "map"), default="map")
    parser.add_argument("--ell", type=float, default=FIXED_ELL)
    parser.add_argument("--w", type=float, default=FIXED_W)
    parser.add_argument(
        "--derivative-sigma",
        type=float,
        default=1.0,
        help="Fixed derivative-noise standard deviation for --mode fixed.",
    )
    parser.add_argument("--objective", choices=("lml", "loo"), default="lml")
    parser.add_argument("--opt-steps", type=int, default=300)
    parser.add_argument("--opt-restarts", type=int, default=3)
    parser.add_argument("--opt-learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _shift(values: np.ndarray, alignment: str) -> np.ndarray:
    anchor = np.max(values) if alignment == "max" else np.min(values)
    return values - anchor


def _crop_reference(reference, interval: tuple[float, float]):
    mask = (reference.x >= interval[0]) & (reference.x <= interval[1])
    if not np.any(mask):
        return reference.x, reference.pmf, reference.derivative
    derivative = reference.derivative[mask] if reference.derivative is not None else None
    return reference.x[mask], reference.pmf[mask], derivative


def _rmse_against_reference(
    x_test: np.ndarray,
    mean: np.ndarray,
    ref_x: np.ndarray,
    ref_pmf: np.ndarray,
    *,
    alignment: str,
) -> float:
    ref_interp = np.interp(x_test, ref_x, ref_pmf)
    return float(
        np.sqrt(np.mean((_shift(mean, alignment) - _shift(ref_interp, alignment)) ** 2))
    )


def _plot_pmf(
    *,
    output_path: Path,
    x_test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ref_x: np.ndarray,
    ref_pmf: np.ndarray,
    alignment: str,
    title: str,
) -> None:
    shifted_mean = _shift(mean, alignment)
    shifted_ref = _shift(ref_pmf, alignment)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_test, shifted_mean, color="tab:red", linewidth=2.0, label="GPRD mean")
    ax.fill_between(
        x_test,
        shifted_mean - 2.0 * std,
        shifted_mean + 2.0 * std,
        color="tab:red",
        alpha=0.18,
        label="GPRD +/- 2 SD",
    )
    ax.plot(ref_x, shifted_ref, color="black", linewidth=1.5, label="Metadynamics FES")
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel("Aligned free energy [kJ/mol]")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_derivative(
    *,
    output_path: Path,
    x_test: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    obs_x: np.ndarray,
    obs_y: np.ndarray,
    ref_x: np.ndarray,
    ref_derivative: np.ndarray | None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(obs_x, obs_y, color="black", s=12, alpha=0.65, label="Binned ICF")
    ax.plot(x_test, mean, color="tab:purple", linewidth=2.0, label="GPRD derivative")
    ax.fill_between(
        x_test,
        mean - 2.0 * std,
        mean + 2.0 * std,
        color="tab:purple",
        alpha=0.16,
        label="GPRD +/- 2 SD",
    )
    if ref_derivative is not None:
        ax.plot(
            ref_x,
            ref_derivative,
            color="tab:green",
            linewidth=1.5,
            label="FES derivative",
        )
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel("Free-energy derivative [kJ/mol/nm]")
    ax.set_title("Derivative-only metadynamics GPR")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parser().parse_args()
    configure_torch(seed=args.seed)
    root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.results_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    interval = (float(args.interval[0]), float(args.interval[1]))

    trajectory = load_colvar(_resolve(root, args.colvar))
    reference = load_fes(_resolve(root, args.fes))
    ref_x, ref_pmf, ref_derivative = _crop_reference(reference, interval)
    derivative_obs = process_metadynamics_icf_derivatives(
        trajectory,
        force_constant=args.force_constant,
        interval=interval,
        derivative_bin_width=args.derivative_bin_width,
        n_derivative_bins=args.n_derivative_bins,
        derivative_binning=args.derivative_binning,
        time_fraction=args.time_fraction,
        min_derivative_samples=args.min_derivative_samples,
    )
    x_test = torch.linspace(
        interval[0], interval[1], args.num_test_points, dtype=torch.float64
    )

    if args.mode == "map":
        optimized = optimize_derivative_hyperparameters(
            x_der=derivative_obs.x_der,
            dy_der=derivative_obs.dy_der,
            priors=HyperPriorConfig(),
            objective=args.objective,
            steps=args.opt_steps,
            learning_rate=args.opt_learning_rate,
            restarts=args.opt_restarts,
            seed=args.seed,
        )
        posterior = optimized.posterior
        hyperparameters = {
            name: float(value.detach().cpu().item())
            for name, value in optimized.params.items()
        }
        objective_value = float(optimized.objective_value)
        model_label = "MAP derivative-only GP"
    else:
        posterior = build_derivative_gp(
            x_der=derivative_obs.x_der,
            dy_der=derivative_obs.dy_der,
            ell=torch.tensor(args.ell, dtype=torch.float64),
            w=torch.tensor(args.w, dtype=torch.float64),
            noise_deriv_diag=(args.derivative_sigma**2)
            * torch.ones(derivative_obs.x_der.numel(), dtype=torch.float64),
            jitter=1e-6,
        )
        hyperparameters = {
            "ell": args.ell,
            "w": args.w,
            "sigma_d": args.derivative_sigma,
        }
        objective_value = None
        model_label = "Fixed derivative-only GP"

    function_mean, function_cov = predict_derivative_gp_function(posterior, x_test)
    derivative_mean, derivative_cov = predict_derivative_gp_derivative(posterior, x_test)
    x_np = x_test.detach().cpu().numpy()
    function_mean_np = function_mean.detach().cpu().numpy()
    function_std_np = np.sqrt(
        np.clip(torch.diagonal(function_cov).detach().cpu().numpy(), 0.0, None)
    )
    derivative_mean_np = derivative_mean.detach().cpu().numpy()
    derivative_std_np = np.sqrt(
        np.clip(torch.diagonal(derivative_cov).detach().cpu().numpy(), 0.0, None)
    )
    rmse = _rmse_against_reference(
        x_np,
        function_mean_np,
        ref_x,
        ref_pmf,
        alignment=args.pmf_alignment,
    )

    _plot_pmf(
        output_path=output_dir / "metadynamics_gprd_pmf.png",
        x_test=x_np,
        mean=function_mean_np,
        std=function_std_np,
        ref_x=ref_x,
        ref_pmf=ref_pmf,
        alignment=args.pmf_alignment,
        title=f"{model_label} metadynamics reconstruction",
    )
    _plot_derivative(
        output_path=output_dir / "metadynamics_gprd_derivative.png",
        x_test=x_np,
        mean=derivative_mean_np,
        std=derivative_std_np,
        obs_x=derivative_obs.x_der.detach().cpu().numpy(),
        obs_y=derivative_obs.dy_der.detach().cpu().numpy(),
        ref_x=ref_x,
        ref_derivative=ref_derivative,
    )

    with (output_dir / "metadynamics_derivative_observations.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x",
                "mean_derivative",
                "sample_count",
                "ar1_autocorr_time",
                "sample_variance",
            ]
        )
        for row in range(derivative_obs.x_der.numel()):
            writer.writerow(
                [
                    float(derivative_obs.x_der[row].item()),
                    float(derivative_obs.dy_der[row].item()),
                    float(derivative_obs.sample_counts[row].item()),
                    float(derivative_obs.autocorr_times[row].item()),
                    float(derivative_obs.sample_variances[row].item()),
                ]
            )

    summary = {
        "colvar_path": str(_resolve(root, args.colvar)),
        "fes_path": str(_resolve(root, args.fes)),
        "mode": args.mode,
        "objective": args.objective if args.mode == "map" else None,
        "force_constant": args.force_constant,
        "interval": interval,
        "time_fraction": args.time_fraction,
        "derivative_binning": args.derivative_binning,
        "n_derivative_observations": int(derivative_obs.x_der.numel()),
        "hyperparameters": hyperparameters,
        "objective_value": objective_value,
        "pmf_alignment": args.pmf_alignment,
        "rmse_metad_fes": rmse,
        "avg_total_std": float(function_std_np.mean()),
    }
    with (output_dir / "run_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
