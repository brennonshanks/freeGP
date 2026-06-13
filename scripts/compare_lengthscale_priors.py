#!/usr/bin/env python3
"""Compare low-data GP reconstructions under four length-scale hyperpriors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
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

from compare_uq_methods import (
    FIXED_ELL,
    FIXED_W,
    _plot_summary,
    _plot_ui_reference,
    _prepare_training_bundle,
    _ui_display_offset,
)
from freegp.config import configure_torch
from freegp.gp import build_joint_gp
from freegp.hmc import (
    HyperPriorConfig,
    NUTSConfig,
    run_hmc_nuts,
    summarize_chain_diagnostics,
)
from freegp.hyperopt import optimize_stationary_hyperparameters
from freegp.posterior import (
    summarize_fixed_posterior_derivative,
    summarize_fixed_posterior_predictive,
    summarize_hyperposterior_derivative,
    summarize_hyperposterior_predictive,
)


@dataclass(frozen=True)
class PriorCase:
    slug: str
    label: str
    m_ell: float
    s_ell: float | None
    color: str


PRIOR_CASES = (
    PriorCase("flat", "Flat log(ell)", math.log(0.5), None, "tab:gray"),
    PriorCase("current", "Current: log(ell) ~ N(log(4), 1)", math.log(4.0), 1.0, "tab:blue"),
    PriorCase("ell_0p5_narrow", "ell=0.5, sigma_log=0.5", math.log(0.5), 0.5, "tab:orange"),
    PriorCase("ell_0p5_very_narrow", "ell=0.5, sigma_log=0.1", math.log(0.5), 0.01, "tab:green"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="~/freeGP-datasets/membranes/katka")
    parser.add_argument("--results-dir", default="results/lengthscale-prior-comparison")
    parser.add_argument("--objective", choices=("lml", "loo"), default="lml")
    parser.add_argument("--window-count", type=int, default=6)
    parser.add_argument("--trajectory-fraction", type=float, default=0.25)
    parser.add_argument("--n-equilibration", type=int, default=40_000)
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--num-test-points", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--predictive-samples", type=int, default=200)
    parser.add_argument("--target-accept-prob", type=float, default=0.9)
    parser.add_argument("--opt-steps", type=int, default=500)
    parser.add_argument("--opt-restarts", type=int, default=5)
    parser.add_argument("--opt-learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _fixed_posterior(observations):
    return build_joint_gp(
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


def _finish_grid(axes, *, ylabel: str) -> None:
    for ax in axes.flat:
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)


def main() -> None:
    args = _parser().parse_args()
    configure_torch(seed=args.seed)
    output_dir = Path(args.results_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, selected_centers = _prepare_training_bundle(args)
    observations = bundle.observations
    fixed_posterior = _fixed_posterior(observations)
    fixed_function = summarize_fixed_posterior_predictive(
        fixed_posterior, bundle.x_test
    )
    fixed_derivative = summarize_fixed_posterior_derivative(
        fixed_posterior, bundle.x_test
    )
    config = NUTSConfig(
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
        target_accept_prob=args.target_accept_prob,
        objective=args.objective,
        kernel="stationary",
    )

    results = []
    for case_index, case in enumerate(PRIOR_CASES):
        priors = HyperPriorConfig(m_ell=case.m_ell, s_ell=case.s_ell)
        optimized = optimize_stationary_hyperparameters(
            observations,
            objective=args.objective,
            priors=priors,
            steps=args.opt_steps,
            learning_rate=args.opt_learning_rate,
            restarts=args.opt_restarts,
            seed=args.seed + case_index,
        )
        map_function = summarize_fixed_posterior_predictive(
            optimized.posterior, bundle.x_test
        )
        map_derivative = summarize_fixed_posterior_derivative(
            optimized.posterior, bundle.x_test
        )
        mcmc, samples = run_hmc_nuts(observations, priors=priors, config=config)
        diagnostics = summarize_chain_diagnostics(mcmc, samples, config=config)
        hyper_function = summarize_hyperposterior_predictive(
            observations,
            samples,
            bundle.x_test,
            priors=priors,
            config=config,
            max_samples=args.predictive_samples,
        )
        hyper_derivative = summarize_hyperposterior_derivative(
            observations,
            samples,
            bundle.x_test,
            priors=priors,
            config=config,
            max_samples=args.predictive_samples,
        )
        ell_samples = torch.exp(samples["theta_ell"]).detach().cpu().numpy()
        results.append(
            {
                "case": case,
                "priors": priors,
                "optimized": optimized,
                "map_function": map_function,
                "map_derivative": map_derivative,
                "hyper_function": hyper_function,
                "hyper_derivative": hyper_derivative,
                "ell_samples": ell_samples,
                "diagnostics": diagnostics,
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    for ax, result in zip(axes.flat, results):
        case = result["case"]
        ui_offset = _ui_display_offset(bundle.references, result["hyper_function"])
        _plot_ui_reference(ax, bundle.references, offset=ui_offset)
        _plot_summary(
            ax, fixed_function, label="Fixed-hyperparameter GP", color="tab:purple"
        )
        _plot_summary(
            ax, result["map_function"], label="MAP plug-in GP", color="tab:orange"
        )
        _plot_summary(
            ax,
            result["hyper_function"],
            label="Hyperposterior GP",
            color="tab:green",
        )
        ax.set_title(case.label)
        result["ui_offset"] = ui_offset
    _finish_grid(axes, ylabel="Free energy [kJ/mol]")
    fig.suptitle("Function UQ sensitivity to the length-scale hyperprior")
    fig.tight_layout()
    fig.savefig(output_dir / "function_uq_by_lengthscale_prior.png", dpi=200)
    plt.close(fig)

    derivative_error = np.sqrt(
        np.clip(observations.noise_deriv_diag.detach().cpu().numpy(), 0.0, None)
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    for ax, result in zip(axes.flat, results):
        case = result["case"]
        ax.errorbar(
            observations.x_der.detach().cpu().numpy(),
            observations.dy_der.detach().cpu().numpy(),
            yerr=derivative_error,
            color="black",
            linestyle="none",
            marker="o",
            markersize=3,
            capsize=2,
            alpha=0.65,
            label="Derivative observations",
        )
        _plot_summary(
            ax, fixed_derivative, label="Fixed-hyperparameter GP", color="tab:purple"
        )
        _plot_summary(
            ax, result["map_derivative"], label="MAP plug-in GP", color="tab:orange"
        )
        _plot_summary(
            ax,
            result["hyper_derivative"],
            label="Hyperposterior GP",
            color="tab:green",
        )
        ax.set_title(case.label)
    _finish_grid(axes, ylabel="Free-energy derivative [kJ/mol/nm]")
    fig.suptitle("Derivative UQ sensitivity to the length-scale hyperprior")
    fig.tight_layout()
    fig.savefig(output_dir / "derivative_uq_by_lengthscale_prior.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    positive_ell = np.concatenate([result["ell_samples"] for result in results])
    bins = np.geomspace(max(positive_ell.min(), 1e-4), positive_ell.max(), 45)
    for result in results:
        case = result["case"]
        ax.hist(
            result["ell_samples"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=case.color,
            label=case.label,
        )
        ax.axvline(
            float(result["optimized"].params["ell"].detach().cpu().item()),
            color=case.color,
            linestyle="--",
            alpha=0.8,
        )
    ax.axvline(FIXED_ELL, color="tab:purple", linestyle=":", linewidth=2.0, label="Fixed ell")
    ax.set_xscale("log")
    ax.set_xlabel("Length scale ell [nm]")
    ax.set_ylabel("Posterior density")
    ax.set_title("Length-scale posterior by hyperprior; dashed lines are MAP estimates")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "ell_posterior_by_prior.png", dpi=200)
    plt.close(fig)

    with (output_dir / "prior_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "prior",
                "m_ell_log",
                "s_ell_log",
                "map_ell_nm",
                "posterior_ell_mean_nm",
                "posterior_ell_median_nm",
                "posterior_ell_q05_nm",
                "posterior_ell_q95_nm",
                "function_avg_total_std",
                "function_avg_between_std",
                "derivative_avg_total_std",
                "derivative_avg_between_std",
                "ui_display_offset",
                "mean_accept_prob",
                "divergence_count",
                "chain_looks_stuck",
            ]
        )
        for result in results:
            case = result["case"]
            function_summary = result["hyper_function"]
            derivative_summary = result["hyper_derivative"]
            ell_samples = result["ell_samples"]
            writer.writerow(
                [
                    case.slug,
                    case.m_ell,
                    "" if case.s_ell is None else case.s_ell,
                    float(result["optimized"].params["ell"].detach().cpu().item()),
                    float(np.mean(ell_samples)),
                    float(np.median(ell_samples)),
                    float(np.quantile(ell_samples, 0.05)),
                    float(np.quantile(ell_samples, 0.95)),
                    float(
                        torch.sqrt(
                            torch.clamp(function_summary.total_variance, min=0.0)
                        ).mean()
                    ),
                    float(
                        torch.sqrt(
                            torch.clamp(function_summary.between_variance, min=0.0)
                        ).mean()
                    ),
                    float(
                        torch.sqrt(
                            torch.clamp(derivative_summary.total_variance, min=0.0)
                        ).mean()
                    ),
                    float(
                        torch.sqrt(
                            torch.clamp(derivative_summary.between_variance, min=0.0)
                        ).mean()
                    ),
                    result["ui_offset"],
                    result["diagnostics"].mean_accept_prob,
                    result["diagnostics"].divergence_count,
                    result["diagnostics"].looks_stuck,
                ]
            )

    summary = {
        "objective": args.objective,
        "window_count": len(selected_centers),
        "selected_window_centers_nm": selected_centers,
        "trajectory_fraction": args.trajectory_fraction,
        "warmup_steps": args.warmup_steps,
        "num_samples": args.num_samples,
        "predictive_samples": args.predictive_samples,
        "unchanged_other_hyperpriors": {
            key: value
            for key, value in asdict(HyperPriorConfig()).items()
            if key not in {"m_ell", "s_ell"}
        },
        "lengthscale_prior_cases": [
            {
                "slug": result["case"].slug,
                "label": result["case"].label,
                "m_ell_log": result["case"].m_ell,
                "s_ell_log": result["case"].s_ell,
                "map_parameters": {
                    name: float(value.detach().cpu().item())
                    for name, value in result["optimized"].params.items()
                },
                "hmc_diagnostics": asdict(result["diagnostics"]),
            }
            for result in results
        ],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
