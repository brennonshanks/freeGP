#!/usr/bin/env python3
"""Run fixed, MAP, and optional short-HMC GP reconstructions on tutorial data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from freegp.gp import build_joint_gp, predict_function
from freegp.hmc import HyperPriorConfig, NUTSConfig, run_hmc_nuts
from freegp.hyperopt import optimize_stationary_hyperparameters
from freegp.posterior import summarize_hyperposterior_predictive
from freegp.workflow import prepare_gprhd_hmc_inputs
from freegp.data import load_umbrella_windows


def shift_to_reference_minimum(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return y - y.min() + ref.min()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="example_data")
    parser.add_argument("--reference-path", default=None)
    parser.add_argument("--results-dir", default="tutorial_results")
    parser.add_argument("--num-bins", type=int, default=12)
    parser.add_argument("--num-test-points", type=int, default=300)
    parser.add_argument("--opt-steps", type=int, default=200)
    parser.add_argument("--opt-restarts", type=int, default=3)
    parser.add_argument("--objective", choices=("lml", "loo"), default="loo")
    parser.add_argument("--skip-hmc", action="store_true", help="Skip the short HMC-NUTS tutorial run.")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--predictive-samples", type=int, default=50)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    reference_path = (
        Path(args.reference_path).expanduser().resolve()
        if args.reference_path
        else dataset_root / "ground_truth.csv"
    )
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    bundle = prepare_gprhd_hmc_inputs(
        dataset_root=str(dataset_root),
        reference_wham_path=str(reference_path) if reference_path.exists() else None,
        n_equilibration=0,
        num_bins=args.num_bins,
        num_test_points=args.num_test_points,
        test_grid_source="histogram_support",
    )
    obs = bundle.observations
    x_test = bundle.x_test

    fixed = build_joint_gp(
        x_func=obs.x_obs,
        y_func=obs.y_obs,
        x_der=obs.x_der,
        dy_der=obs.dy_der,
        ell=torch.tensor(np.pi / 2.0, dtype=torch.float64),
        w=torch.tensor(4.184 * np.sqrt(10.0), dtype=torch.float64),
        noise_func_cov=obs.noise_func_cov,
        noise_deriv_diag=obs.noise_deriv_diag,
        H_func=obs.H_obs,
        jitter=1e-6,
    )
    fixed_mean, fixed_cov = predict_function(fixed, x_test)

    opt = optimize_stationary_hyperparameters(
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
            seed=11,
            jitter=1e-6,
        )
        _, samples = run_hmc_nuts(obs, priors=HyperPriorConfig(), config=config)
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

    x = x_test.detach().cpu().numpy()
    fixed_mean = fixed_mean.detach().cpu().numpy()
    fixed_std = torch.sqrt(torch.diagonal(fixed_cov)).detach().cpu().numpy()
    map_mean = map_mean.detach().cpu().numpy()
    map_std = torch.sqrt(torch.diagonal(map_cov)).detach().cpu().numpy()

    if bundle.references.has_wham:
        ref = np.interp(x, bundle.references.wham_x, bundle.references.wham_f)
        ref = ref - ref.min()
    else:
        ref = np.zeros_like(x)

    fixed_mean = shift_to_reference_minimum(fixed_mean, ref)
    map_mean = shift_to_reference_minimum(map_mean, ref)
    if hmc_mean is not None:
        hmc_mean = shift_to_reference_minimum(hmc_mean, ref)

    rows = [
        x,
        ref,
        fixed_mean,
        fixed_std,
        map_mean,
        map_std,
    ]
    header = "x_nm,reference,fixed_mean,fixed_std,map_mean,map_std"
    if hmc_mean is not None and hmc_std is not None:
        rows += [hmc_mean, hmc_std]
        header += ",hmc_mean,hmc_std"
    np.savetxt(out / "synthetic_reconstruction.csv", np.column_stack(rows), delimiter=",", header=header, comments="")

    windows = load_umbrella_windows(dataset_root)
    sample_positions = [w.position.detach().cpu().numpy() for w in windows]

    fig, (ax_hist, ax) = plt.subplots(
        2,
        1,
        figsize=(5.0, 4.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 3.0], "hspace": 0.05},
        constrained_layout=True,
    )
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sample_positions)))
    for samples, color in zip(sample_positions, colors):
        ax_hist.hist(
            samples,
            bins=35,
            density=True,
            histtype="step",
            lw=0.8,
            color=color,
            alpha=0.9,
        )
    ax_hist.set_ylabel("Density")
    ax_hist.tick_params(direction="in", labelbottom=False)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    ax.plot(x, ref, color="black", lw=1.8, ls="--", label="Reference")
    ax.plot(x, fixed_mean, color="#882255", lw=1.6, label="Fixed GP")
    ax.fill_between(x, fixed_mean - fixed_std, fixed_mean + fixed_std, color="#882255", alpha=0.18, lw=0)
    ax.plot(x, map_mean, color="#4477AA", lw=1.8, label="MAP GP")
    ax.fill_between(x, map_mean - map_std, map_mean + map_std, color="#4477AA", alpha=0.18, lw=0)
    if hmc_mean is not None and hmc_std is not None:
        ax.plot(x, hmc_mean, color="#228833", lw=1.8, label="HMC GP")
        ax.fill_between(x, hmc_mean - hmc_std, hmc_mean + hmc_std, color="#228833", alpha=0.18, lw=0)
    ax.set_xlabel("Coordinate [nm]")
    ax.set_ylabel("Free Energy [kJ/mol]")
    ax.tick_params(direction="in")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "synthetic_reconstruction.png", dpi=250)
    print(f"Wrote tutorial outputs to {out.resolve()}")
    print("MAP parameters:", {k: float(v.detach().cpu()) for k, v in opt.params.items()})


if __name__ == "__main__":
    main()
