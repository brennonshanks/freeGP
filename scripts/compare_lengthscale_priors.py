#!/usr/bin/env python3
"""Compare GP reconstructions under four length-scale hyperpriors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from compare_uq_methods import (
    _plot_summary,
    _plot_ui_reference,
    _prepare_training_bundle,
    _ui_display_offset,
)
from freegp.config import configure_torch
from freegp.hmc import (
    HyperPriorConfig,
    NUTSConfig,
    run_hmc_nuts,
    stationary_log_ell_bounds,
)
from freegp.metrics import compare_to_reference_curves
from freegp.posterior import (
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
    PriorCase("flat", r"Flat in $\log \ell$", math.log(0.5), None, "tab:gray"),
    PriorCase(
        "current",
        r"Default: $\log \ell \sim \mathcal{N}(\log 4, 1)$",
        math.log(4.0),
        1.0,
        "tab:blue",
    ),
    PriorCase(
        "ell_0p5_narrow",
        r"$\ell_0=0.5$ nm, $\sigma_{\log \ell}=0.5$",
        math.log(0.5),
        0.5,
        "tab:orange",
    ),
    PriorCase(
        "ell_0p5_very_narrow",
        r"$\ell_0=0.5$ nm, $\sigma_{\log \ell}=0.1$",
        math.log(0.5),
        0.1,
        "tab:green",
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="~/freeGP-datasets/membranes/katka")
    parser.add_argument(
        "--results-dir", default="results/lengthscale-prior-sensitivity-hard"
    )
    parser.add_argument("--objective", choices=("lml", "loo"), default="loo")
    parser.add_argument("--pmf-alignment", choices=("max", "min"), default="max")
    parser.add_argument("--window-count", type=int, default=7)
    parser.add_argument("--trajectory-fraction", type=float, default=0.25)
    parser.add_argument("--n-equilibration", type=int, default=40_000)
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--num-test-points", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument(
        "--chain-execution",
        choices=("spawn", "serial", "parallel"),
        default="spawn",
        help="Run independent single-chain workers with spawn by default.",
    )
    parser.add_argument("--predictive-samples", type=int, default=200)
    parser.add_argument("--target-accept-prob", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _finish_grid(axes, *, ylabel: str) -> None:
    for ax in axes.flat:
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)


def _to_python(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _flatten_finite(value) -> list[float]:
    if isinstance(value, dict):
        return [
            number
            for item in value.values()
            for number in _flatten_finite(item)
        ]
    if isinstance(value, list):
        return [number for item in value for number in _flatten_finite(item)]
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    return []


def _divergence_count(value) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(_divergence_count(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return int(value)


def _summarize_raw_diagnostics(raw) -> dict[str, object]:
    raw = _to_python(raw)
    parameter_diagnostics = {
        key: value
        for key, value in raw.items()
        if isinstance(value, dict) and "r_hat" in value and "n_eff" in value
    }
    r_hats = [
        number
        for value in parameter_diagnostics.values()
        for number in _flatten_finite(value["r_hat"])
    ]
    n_effs = [
        number
        for value in parameter_diagnostics.values()
        for number in _flatten_finite(value["n_eff"])
    ]
    acceptance_rates = _flatten_finite(raw.get("acceptance rate", {}))
    return {
        "max_r_hat": max(r_hats) if r_hats else None,
        "min_n_eff": min(n_effs) if n_effs else None,
        "divergence_total": _divergence_count(raw.get("divergences", {})),
        "min_acceptance_rate": min(acceptance_rates) if acceptance_rates else None,
        "max_acceptance_rate": max(acceptance_rates) if acceptance_rates else None,
        "raw": raw,
    }


def _run_single_chain_worker(payload):
    observations, priors, config, seed = payload
    configure_torch(seed=seed)
    mcmc, samples = run_hmc_nuts(observations, priors=priors, config=config)
    return (
        {name: values.detach().cpu() for name, values in samples.items()},
        _to_python(mcmc.diagnostics()),
    )


def _run_chains(
    observations,
    *,
    priors: HyperPriorConfig,
    config: NUTSConfig,
    seed: int,
    execution: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if config.num_chains == 1 or execution == "parallel":
        mcmc, samples = run_hmc_nuts(observations, priors=priors, config=config)
        return samples, _summarize_raw_diagnostics(mcmc.diagnostics())

    from pyro.infer.mcmc.util import diagnostics as pyro_diagnostics

    single_chain_config = replace(config, num_chains=1)
    payloads = [
        (observations, priors, single_chain_config, seed + chain_index)
        for chain_index in range(config.num_chains)
    ]
    if execution == "spawn":
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(config.num_chains) as pool:
            chain_outputs = pool.map(_run_single_chain_worker, payloads)
    else:
        chain_outputs = [_run_single_chain_worker(payload) for payload in payloads]

    chain_samples = []
    acceptance_rates = {}
    divergences = {}
    for chain_index, (samples, chain_diagnostics) in enumerate(chain_outputs):
        chain_samples.append(samples)
        acceptance = _flatten_finite(
            chain_diagnostics.get("acceptance rate", {})
        )
        acceptance_rates[f"chain {chain_index}"] = (
            acceptance[0] if acceptance else None
        )
        chain_divergences = chain_diagnostics.get("divergences", {})
        if isinstance(chain_divergences, dict):
            values = list(chain_divergences.values())
            divergences[f"chain {chain_index}"] = values[0] if values else []
        else:
            divergences[f"chain {chain_index}"] = chain_divergences

    grouped_samples = {
        name: torch.stack([samples[name] for samples in chain_samples], dim=0)
        for name in chain_samples[0]
    }
    flat_samples = {
        name: values.reshape((-1,) + values.shape[2:])
        for name, values in grouped_samples.items()
    }
    raw = _to_python(pyro_diagnostics(grouped_samples))
    raw["acceptance rate"] = acceptance_rates
    raw["divergences"] = divergences
    return flat_samples, _summarize_raw_diagnostics(raw)


def _parameter_samples(samples: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        "ell": torch.exp(samples["theta_ell"]).detach().cpu().numpy(),
        "w": torch.exp(samples["theta_w"]).detach().cpu().numpy(),
        "sigma_f": torch.exp(samples["theta_sf"]).detach().cpu().numpy(),
        "sigma_d": torch.exp(samples["theta_sd"]).detach().cpu().numpy(),
    }


def _aligned_curve(summary, alignment: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = summary.x_test.detach().cpu().numpy()
    mean = summary.mean.detach().cpu().numpy()
    anchor = np.max(mean) if alignment == "max" else np.min(mean)
    std = np.sqrt(
        np.clip(summary.total_variance.detach().cpu().numpy(), 0.0, None)
    )
    return x, mean - anchor, std


def _ell_prior_density(
    ell: np.ndarray,
    case: PriorCase,
    *,
    log_bounds: tuple[float, float],
) -> np.ndarray:
    if case.s_ell is None:
        lower, upper = log_bounds
        density = 1.0 / (ell * (upper - lower))
        return np.where((np.log(ell) >= lower) & (np.log(ell) <= upper), density, 0.0)
    z = (np.log(ell) - case.m_ell) / case.s_ell
    return np.exp(-0.5 * z**2) / (
        ell * case.s_ell * math.sqrt(2.0 * math.pi)
    )


def _positive_kde(
    samples: np.ndarray,
    x_grid: np.ndarray,
    *,
    bandwidth_scale: float = 1.0,
) -> np.ndarray:
    """Evaluate a Gaussian KDE in log space and transform it to x space."""
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 2:
        return np.zeros_like(x_grid)

    log_values = np.log(values)
    sample_std = float(np.std(log_values, ddof=1))
    bandwidth = bandwidth_scale * sample_std * values.size ** (-1.0 / 5.0)
    bandwidth = max(bandwidth, 1e-3)
    log_grid = np.log(x_grid)
    z = (log_grid[:, None] - log_values[None, :]) / bandwidth
    log_density = np.exp(-0.5 * z**2).mean(axis=1) / (
        bandwidth * math.sqrt(2.0 * math.pi)
    )
    return log_density / x_grid


def _positive_kde_grid(sample_groups: list[np.ndarray]) -> np.ndarray:
    pooled = np.concatenate(sample_groups)
    pooled = pooled[np.isfinite(pooled) & (pooled > 0.0)]
    lower = float(np.quantile(pooled, 0.001))
    upper = float(np.quantile(pooled, 0.999))
    padding = 0.08 * (math.log(upper) - math.log(lower))
    return np.geomspace(
        math.exp(math.log(lower) - padding),
        math.exp(math.log(upper) + padding),
        600,
    )


def _plot_prior_posterior_comparison(
    ax,
    results,
    *,
    ell_grid: np.ndarray,
    ell_kde_grid: np.ndarray,
    log_ell_bounds: tuple[float, float],
) -> None:
    for result in results:
        case = result["case"]
        prior_density = _ell_prior_density(
            ell_grid, case, log_bounds=log_ell_bounds
        )
        positive = prior_density > 0.0
        prior_density = prior_density / np.max(prior_density)
        posterior_density = _positive_kde(
            result["parameter_samples"]["ell"], ell_kde_grid
        )
        posterior_density = posterior_density / np.max(posterior_density)
        ax.plot(
            ell_grid[positive],
            prior_density[positive],
            color=case.color,
            linestyle="--",
            linewidth=1.8,
        )
        ax.plot(
            ell_kde_grid,
            posterior_density,
            color=case.color,
            linewidth=2.2,
            label=case.label,
        )

    style_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Prior"),
        Line2D([0], [0], color="black", linestyle="-", linewidth=2.2, label="Posterior"),
    ]
    case_handles = [
        Line2D([0], [0], color=case.color, linewidth=2.2, label=case.label)
        for case in PRIOR_CASES
    ]
    ax.legend(handles=case_handles + style_handles, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"Length scale $\ell$ [nm]")
    ax.set_ylabel("Peak-normalized density")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.2)


def main() -> None:
    args = _parser().parse_args()
    configure_torch(seed=args.seed)
    output_dir = Path(args.results_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, selected_centers = _prepare_training_bundle(args)
    observations = bundle.observations
    config = NUTSConfig(
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
        target_accept_prob=args.target_accept_prob,
        objective=args.objective,
        kernel="stationary",
        num_chains=args.num_chains,
    )

    results = []
    for case_index, case in enumerate(PRIOR_CASES):
        priors = HyperPriorConfig(m_ell=case.m_ell, s_ell=case.s_ell)
        samples, diagnostics = _run_chains(
            observations,
            priors=priors,
            config=config,
            seed=args.seed + 1000 * case_index,
            execution=args.chain_execution,
        )
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
        parameter_samples = _parameter_samples(samples)
        reference_metrics = compare_to_reference_curves(
            hyper_function,
            bundle.references,
            alignment=args.pmf_alignment,
        )
        results.append(
            {
                "case": case,
                "priors": priors,
                "hyper_function": hyper_function,
                "hyper_derivative": hyper_derivative,
                "parameter_samples": parameter_samples,
                "reference_metrics": reference_metrics,
                "diagnostics": diagnostics,
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    for ax, result in zip(axes.flat, results):
        case = result["case"]
        ui_offset = _ui_display_offset(bundle.references, result["hyper_function"])
        _plot_ui_reference(ax, bundle.references, offset=ui_offset)
        _plot_summary(
            ax,
            result["hyper_function"],
            label="Hyperposterior-propagated GP",
            color=case.color,
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
            ax,
            result["hyper_derivative"],
            label="Hyperposterior-propagated GP",
            color=case.color,
        )
        ax.set_title(case.label)
    _finish_grid(axes, ylabel="Free-energy derivative [kJ/mol/nm]")
    fig.suptitle("Derivative UQ sensitivity to the length-scale hyperprior")
    fig.tight_layout()
    fig.savefig(output_dir / "derivative_uq_by_lengthscale_prior.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ell_kde_grid = _positive_kde_grid(
        [result["parameter_samples"]["ell"] for result in results]
    )
    for result in results:
        case = result["case"]
        ax.plot(
            ell_kde_grid,
            _positive_kde(result["parameter_samples"]["ell"], ell_kde_grid),
            linewidth=2.0,
            color=case.color,
            label=case.label,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Length scale $\ell$ [nm]")
    ax.set_ylabel("Posterior density")
    ax.set_title("Length-scale hyperposterior by hyperprior")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "ell_posterior_by_prior.png", dpi=200)
    plt.close(fig)

    log_ell_bounds = stationary_log_ell_bounds(observations)
    ell_grid = np.geomspace(
        math.exp(log_ell_bounds[0]), math.exp(log_ell_bounds[1]), 500
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_prior_posterior_comparison(
        ax,
        results,
        ell_grid=ell_grid,
        ell_kde_grid=ell_kde_grid,
        log_ell_bounds=log_ell_bounds,
    )
    ax.set_title("Length-scale priors and hyperposteriors")
    fig.tight_layout()
    fig.savefig(output_dir / "ell_prior_and_posterior.png", dpi=200)
    plt.close(fig)

    parameter_labels = {
        "ell": r"Length scale $\ell$ [nm]",
        "w": r"Kernel amplitude $w$",
        "sigma_f": r"Function noise $\sigma_f$",
        "sigma_d": r"Derivative noise $\sigma_d$",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (parameter, label) in zip(axes.flat, parameter_labels.items()):
        kde_grid = _positive_kde_grid(
            [result["parameter_samples"][parameter] for result in results]
        )
        for result in results:
            case = result["case"]
            ax.plot(
                kde_grid,
                _positive_kde(result["parameter_samples"][parameter], kde_grid),
                linewidth=2.0,
                color=case.color,
                label=case.label,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Posterior density")
        ax.set_xscale("log")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Joint hyperparameter sensitivity to the length-scale prior")
    fig.tight_layout()
    fig.savefig(output_dir / "all_hyperparameter_posteriors_by_prior.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    for result in results:
        case = result["case"]
        x, mean, std = _aligned_curve(
            result["hyper_function"], args.pmf_alignment
        )
        axes[0].plot(x, mean, color=case.color, linewidth=2.0, label=case.label)
        axes[0].fill_between(
            x, mean - 2.0 * std, mean + 2.0 * std, color=case.color, alpha=0.1
        )
    refs = bundle.references
    if refs.has_ui:
        ui_anchor = (
            np.max(refs.umbrella_f)
            if args.pmf_alignment == "max"
            else np.min(refs.umbrella_f)
        )
        axes[0].errorbar(
            refs.umbrella_x,
            refs.umbrella_f - ui_anchor,
            yerr=refs.umbrella_e,
            color="black",
            linewidth=1.2,
            capsize=2,
            label="Block-averaged UI",
        )
    axes[0].set_xlabel("Position [nm]")
    axes[0].set_ylabel("Aligned free energy [kJ/mol]")
    axes[0].set_title("Posterior PMF sensitivity")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    _plot_prior_posterior_comparison(
        axes[1],
        results,
        ell_grid=ell_grid,
        ell_kde_grid=ell_kde_grid,
        log_ell_bounds=log_ell_bounds,
    )
    axes[1].set_title("Length-scale priors and hyperposteriors")
    fig.suptitle("Length-scale prior sensitivity analysis")
    fig.tight_layout()
    fig.savefig(output_dir / "si_prior_sensitivity.png", dpi=200)
    plt.close(fig)

    with (output_dir / "prior_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "prior",
                "m_ell_log",
                "s_ell_log",
                "posterior_ell_mean_nm",
                "posterior_ell_median_nm",
                "posterior_ell_q05_nm",
                "posterior_ell_q95_nm",
                "function_avg_total_std",
                "function_avg_between_std",
                "derivative_avg_total_std",
                "derivative_avg_between_std",
                "rmse_wham",
                "rmse_ui",
                "ui_display_offset",
                "max_r_hat",
                "min_n_eff",
                "divergence_total",
                "min_acceptance_rate",
                "max_acceptance_rate",
            ]
        )
        for result in results:
            case = result["case"]
            function_summary = result["hyper_function"]
            derivative_summary = result["hyper_derivative"]
            ell_samples = result["parameter_samples"]["ell"]
            writer.writerow(
                [
                    case.slug,
                    case.m_ell,
                    "" if case.s_ell is None else case.s_ell,
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
                    result["reference_metrics"].rmse_wham,
                    result["reference_metrics"].rmse_ui,
                    result["ui_offset"],
                    result["diagnostics"]["max_r_hat"],
                    result["diagnostics"]["min_n_eff"],
                    result["diagnostics"]["divergence_total"],
                    result["diagnostics"]["min_acceptance_rate"],
                    result["diagnostics"]["max_acceptance_rate"],
                ]
            )

    with (output_dir / "hyperparameter_quantiles.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["prior", "parameter", "q05", "median", "q95"])
        for result in results:
            for parameter, values in result["parameter_samples"].items():
                writer.writerow(
                    [
                        result["case"].slug,
                        parameter,
                        float(np.quantile(values, 0.05)),
                        float(np.median(values)),
                        float(np.quantile(values, 0.95)),
                    ]
                )

    summary = {
        "objective": args.objective,
        "window_count": len(selected_centers),
        "selected_window_centers_nm": selected_centers,
        "trajectory_fraction": args.trajectory_fraction,
        "warmup_steps": args.warmup_steps,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
        "chain_execution": args.chain_execution,
        "predictive_samples": args.predictive_samples,
        "pmf_alignment": args.pmf_alignment,
        "flat_log_ell_bounds": {
            "lower": log_ell_bounds[0],
            "upper": log_ell_bounds[1],
            "ell_lower_nm": math.exp(log_ell_bounds[0]),
            "ell_upper_nm": math.exp(log_ell_bounds[1]),
        },
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
                "posterior_parameter_quantiles": {
                    name: {
                        "q05": float(np.quantile(values, 0.05)),
                        "median": float(np.median(values)),
                        "q95": float(np.quantile(values, 0.95)),
                    }
                    for name, values in result["parameter_samples"].items()
                },
                "reference_metrics": asdict(result["reference_metrics"]),
                "hmc_diagnostics": result["diagnostics"],
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
