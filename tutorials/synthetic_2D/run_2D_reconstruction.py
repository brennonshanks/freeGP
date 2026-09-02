#!/usr/bin/env python3
"""Run fixed, MAP, and optional short-HMC GP reconstructions on 2D synthetic umbrella-sampling data.

Two-dimensional analogue of ``tutorials/synthetic_umbrella/run_reconstruction.py``:
instead of a single collective variable, each umbrella window here is biased by an
isotropic harmonic restraint in (x, y), and the joint GP is fit on histogram-derived
free-energy estimates plus full 2D gradient ("restoring force") observations per
window. See ``freegp.gp.build_joint_gp_nd`` for the multidimensional kernel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freegp.gp import build_joint_gp_nd, predict_function
from freegp.hmc import HyperPriorConfig, NUTSConfig, run_hmc_nuts
from freegp.hyperopt import optimize_stationary_hyperparameters_nd
from freegp.posterior import summarize_hyperposterior_predictive
from freegp.workflow import prepare_gprhd_inputs_nd

N_DIM = 2


def load_2d_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a (x, y, U) reference CSV produced on a regular grid (see make_dataset.py)."""
    data = np.loadtxt(path, delimiter=",", comments="#")
    x, y, u = data[:, 0], data[:, 1], data[:, 2]
    x_axis = np.unique(x)
    y_axis = np.unique(y)
    grid = u.reshape(len(y_axis), len(x_axis))
    return x_axis, y_axis, grid


def bilinear_interpolate(x_axis: np.ndarray, y_axis: np.ndarray, grid: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate a regular-grid reference surface at query points."""
    xi = np.clip(np.searchsorted(x_axis, xq) - 1, 0, len(x_axis) - 2)
    yi = np.clip(np.searchsorted(y_axis, yq) - 1, 0, len(y_axis) - 2)
    x0, x1 = x_axis[xi], x_axis[xi + 1]
    y0, y1 = y_axis[yi], y_axis[yi + 1]
    tx = (xq - x0) / (x1 - x0)
    ty = (yq - y0) / (y1 - y0)
    g00 = grid[yi, xi]
    g10 = grid[yi, xi + 1]
    g01 = grid[yi + 1, xi]
    g11 = grid[yi + 1, xi + 1]
    return g00 * (1 - tx) * (1 - ty) + g10 * tx * (1 - ty) + g01 * (1 - tx) * ty + g11 * tx * ty


def shift_to_reference_minimum(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return y - y.min() + ref.min()


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="example_data")
    parser.add_argument("--reference-path", default=None)
    parser.add_argument("--results-dir", default="tutorial_results")
    parser.add_argument("--num-bins", type=int, default=5, help="Histogram bins per dimension, per window.")
    parser.add_argument("--num-test-points", type=int, default=25, help="Test-grid points per dimension.")
    parser.add_argument("--fixed-ell", type=float, default=1.5)
    parser.add_argument("--fixed-w", type=float, default=50.0)
    parser.add_argument("--opt-steps", type=int, default=40)
    parser.add_argument("--opt-restarts", type=int, default=2)
    parser.add_argument("--objective", choices=("lml", "loo"), default="loo")
    parser.add_argument("--vmax", type=float, default=350.0, help="Max color scale for plots.")
    parser.add_argument("--skip-hmc", action="store_true", help="Skip the short HMC-NUTS tutorial run.")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--predictive-samples", type=int, default=10)
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=5,
        help="NUTS max tree depth (caps leapfrog steps/iteration at 2**depth). Each step is a full "
        "Cholesky factorization of the joint covariance, so deep trees are the dominant cost; the "
        "default here trades some sampling efficiency for a bounded per-iteration runtime.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    reference_path = (
        Path(args.reference_path).expanduser().resolve()
        if args.reference_path
        else dataset_root / "ground_truth.csv"
    )
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = prepare_gprhd_inputs_nd(
        n_dim=N_DIM,
        dataset_root=str(dataset_root),
        num_bins=args.num_bins,
        num_test_points_per_dim=args.num_test_points,
        test_grid_source="histogram_support",
    )
    obs = bundle.observations
    x_test = bundle.x_test

    fixed = build_joint_gp_nd(
        x_func=obs.x_obs,
        y_func=obs.y_obs,
        x_der=obs.x_der,
        dy_der=obs.dy_der,
        ell=torch.tensor(args.fixed_ell, dtype=torch.float64),
        w=torch.tensor(args.fixed_w, dtype=torch.float64),
        noise_func_cov=obs.noise_func_cov,
        noise_deriv_cov=obs.noise_deriv_cov,
        H_func=obs.H_obs,
        jitter=1e-6,
    )
    fixed_mean, fixed_cov = predict_function(fixed, x_test)

    opt = optimize_stationary_hyperparameters_nd(
        obs,
        objective=args.objective,
        steps=args.opt_steps,
        restarts=args.opt_restarts,
        use_hyperpriors=True,
        seed=3,
        jitter=1e-6,
    )
    map_mean, map_cov = predict_function(opt.posterior, x_test)

    hmc_mean = hmc_std = None
    if not args.skip_hmc:
        config = NUTSConfig(
            objective=args.objective,
            warmup_steps=args.warmup_steps,
            num_samples=args.num_samples,
            num_chains=args.num_chains,
            max_tree_depth=args.max_tree_depth,
            seed=11,
            jitter=1e-6,
        )
        # Warm-start every chain at the MAP estimate. The hierarchical GP hyperparameter
        # posterior has a spurious short-lengthscale/high-noise mode (the kernel decorrelates
        # down to per-bin noise); with only a short warmup budget an arbitrarily-initialized
        # chain can get stuck there instead of the physically sensible mode MAP already found.
        init_params = {
            "theta_ell": torch.log(opt.params["ell"]).detach(),
            "theta_w": torch.log(opt.params["w"]).detach(),
            "theta_sf": torch.log(opt.params["sigma_f"]).detach(),
            "theta_sd": torch.log(opt.params["sigma_d"]).detach(),
        }
        _, samples = run_hmc_nuts(obs, priors=HyperPriorConfig(), config=config, init_params=init_params)
        hmc = summarize_hyperposterior_predictive(
            obs,
            samples,
            x_test,
            priors=HyperPriorConfig(),
            config=config,
            max_samples=args.predictive_samples,
        )
        hmc_mean = hmc.mean.detach().cpu().numpy()
        hmc_std = torch.sqrt(hmc.total_variance).detach().cpu().numpy()

    xy = x_test.detach().cpu().numpy()
    x, y = xy[:, 0], xy[:, 1]
    fixed_mean = fixed_mean.detach().cpu().numpy()
    fixed_std = torch.sqrt(torch.diagonal(fixed_cov)).detach().cpu().numpy()
    map_mean = map_mean.detach().cpu().numpy()
    map_std = torch.sqrt(torch.diagonal(map_cov)).detach().cpu().numpy()

    has_reference = reference_path.exists()
    if has_reference:
        x_axis, y_axis, ref_grid = load_2d_reference(reference_path)
        ref = bilinear_interpolate(x_axis, y_axis, ref_grid, x, y)
        ref = ref - ref.min()
    else:
        ref = np.zeros_like(x)

    fixed_mean = shift_to_reference_minimum(fixed_mean, ref)
    map_mean = shift_to_reference_minimum(map_mean, ref)
    if hmc_mean is not None:
        hmc_mean = shift_to_reference_minimum(hmc_mean, ref)

    rmse_by_method = {"Fixed GP": rmse(fixed_mean, ref), "MAP GP": rmse(map_mean, ref)}
    if hmc_mean is not None:
        rmse_by_method["HMC GP"] = rmse(hmc_mean, ref)

    rows = [x, y, ref, fixed_mean, fixed_std, map_mean, map_std]
    header = "x,y,reference,fixed_mean,fixed_std,map_mean,map_std"
    if hmc_mean is not None and hmc_std is not None:
        rows += [hmc_mean, hmc_std]
        header += ",hmc_mean,hmc_std"
    np.savetxt(out / "synthetic_2D_reconstruction.csv", np.column_stack(rows), delimiter=",", header=header, comments="")

    n = args.num_test_points
    extent = (x.min(), x.max(), y.min(), y.max())
    mean_panels = [("Reference", ref, None), ("Fixed GP", fixed_mean, fixed_std), ("MAP GP", map_mean, map_std)]
    if hmc_mean is not None:
        mean_panels.append(("HMC GP", hmc_mean, hmc_std))
    n_cols = len(mean_panels)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.75 * n_cols, 7.2), constrained_layout=True)
    window_centers = obs.x_der.detach().cpu().numpy()
    for col, (title, mean_values, std_values) in enumerate(mean_panels):
        ax_mean = axes[0, col]
        im = ax_mean.imshow(
            mean_values.reshape(n, n).T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="viridis",
            vmax=args.vmax,
        )
        ax_mean.set_title(title, fontsize=10)
        ax_mean.scatter(window_centers[:, 0], window_centers[:, 1], s=3, c="white", alpha=0.4, linewidths=0)
        fig.colorbar(im, ax=ax_mean, shrink=0.85)

        ax_std = axes[1, col]
        if std_values is None:
            ax_std.axis("off")
        else:
            im_std = ax_std.imshow(
                std_values.reshape(n, n).T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="magma",
            )
            # Each std panel gets its own color scale (not shared across methods): the
            # absolute noise/amplitude scale differs a lot between fixed/MAP/HMC, and a
            # shared scale would wash out the (often much smaller) spatial variation
            # within each panel -- e.g. the std rising near the domain corners, farthest
            # from any window's coverage.
            ax_std.set_title(f"{title} std [{std_values.min():.3g}, {std_values.max():.3g}]", fontsize=10)
            ax_std.scatter(window_centers[:, 0], window_centers[:, 1], s=3, c="white", alpha=0.4, linewidths=0)
            fig.colorbar(im_std, ax=ax_std, shrink=0.85)
        ax_std.set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[1, 0].set_ylabel("y")

    fig.savefig(out / "synthetic_2D_reconstruction.png", dpi=250)
    print(f"Wrote tutorial outputs to {out.resolve()}")
    print("MAP parameters:", {k: float(v.detach().cpu()) for k, v in opt.params.items()})
    if has_reference:
        print("RMSE vs reference:", {k: round(v, 4) for k, v in rmse_by_method.items()})
    else:
        print(f"No reference surface found at {reference_path}; skipping RMSE.")


if __name__ == "__main__":
    main()
