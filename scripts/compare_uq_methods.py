#!/usr/bin/env python3
"""Quick comparison of UI, fixed, plug-in optimized, and hyperposterior GP UQ."""

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
from freegp.data import UmbrellaWindow, load_umbrella_windows
from freegp.gp import build_joint_gp, predict_function_mean
from freegp.hmc import NUTSConfig, run_hmc_nuts
from freegp.hyperopt import optimize_stationary_hyperparameters
from freegp.metrics import compare_to_reference_curves
from freegp.posterior import (
    hyperposterior_conditional_means,
    summarize_fixed_posterior_derivative,
    summarize_fixed_posterior_predictive,
    summarize_hyperposterior_derivative,
    summarize_hyperposterior_predictive,
)
from freegp.preprocess import build_joint_observations, process_umbrella_windows
from freegp.workflow import WorkflowBundle, prepare_gprhd_hmc_inputs


FIXED_ELL = math.pi / 2.0
FIXED_W = 4.184 * math.sqrt(10.0)


def _shift(values: np.ndarray, alignment: str) -> np.ndarray:
    anchor = np.max(values) if alignment == "max" else np.min(values)
    return values - anchor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", default="~/freeGP-datasets/membranes/katka"
    )
    parser.add_argument("--results-dir", default="results/uq-method-comparison")
    parser.add_argument("--objective", choices=("lml", "loo"), default="lml")
    parser.add_argument("--pmf-alignment", choices=("max", "min"), default="max")
    parser.add_argument("--n-equilibration", type=int, default=40_000)
    parser.add_argument(
        "--window-count",
        type=int,
        default=None,
        help="Retain this many evenly spaced umbrella windows. Default: all windows.",
    )
    parser.add_argument(
        "--trajectory-fraction",
        type=float,
        default=1.0,
        help="Retain this contiguous fraction of each post-equilibration trajectory.",
    )
    parser.add_argument("--num-bins", type=int, default=20)
    parser.add_argument("--num-test-points", type=int, default=100)
    parser.add_argument("--opt-steps", type=int, default=200)
    parser.add_argument("--opt-restarts", type=int, default=3)
    parser.add_argument("--opt-learning-rate", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--predictive-samples", type=int, default=50)
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument(
        "--gauge-check-decay-lengths",
        type=float,
        default=8.0,
        help="Evaluate asymptotic means this many maximum sampled length scales beyond the data.",
    )
    parser.add_argument("--gauge-check-points", type=int, default=100)
    parser.add_argument(
        "--gauge-check-tolerance",
        type=float,
        default=0.1,
        help="Absolute kJ/mol tolerance for declaring a common asymptotic zero.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _prepare_training_bundle(args) -> tuple[WorkflowBundle, list[float]]:
    full_bundle = prepare_gprhd_hmc_inputs(
        dataset_root=args.dataset_root,
        project_root=str(REPO_ROOT),
        n_equilibration=args.n_equilibration,
        num_bins=args.num_bins,
        num_test_points=args.num_test_points,
    )
    if args.window_count is None and args.trajectory_fraction == 1.0:
        return full_bundle, full_bundle.processed.folder_numbers.tolist()
    if args.window_count is not None and args.window_count <= 0:
        raise ValueError("--window-count must be positive.")
    if not 0.0 < args.trajectory_fraction <= 1.0:
        raise ValueError("--trajectory-fraction must be in (0, 1].")

    windows = load_umbrella_windows(args.dataset_root)
    keep_count = len(windows) if args.window_count is None else min(
        args.window_count, len(windows)
    )
    selected_indices = np.unique(
        np.linspace(0, len(windows) - 1, keep_count, dtype=int)
    )
    selected_windows = []
    for index in selected_indices.tolist():
        window = windows[index]
        time_eq = (
            window.time[args.n_equilibration:]
            if len(window.time) > args.n_equilibration
            else window.time
        )
        position_eq = (
            window.position[args.n_equilibration:]
            if len(window.position) > args.n_equilibration
            else window.position
        )
        n_keep = min(
            position_eq.numel(),
            max(2, int(np.floor(position_eq.numel() * args.trajectory_fraction))),
        )
        selected_windows.append(
            UmbrellaWindow(
                folder=window.folder,
                folder_number=window.folder_number,
                time=time_eq[:n_keep].clone(),
                position=position_eq[:n_keep].clone(),
                mdp_last_key=window.mdp_last_key,
                mdp_last_value=window.mdp_last_value,
            )
        )

    processed = process_umbrella_windows(
        selected_windows, n_equilibration=0, num_bins=args.num_bins
    )
    return (
        WorkflowBundle(
            processed=processed,
            observations=build_joint_observations(processed),
            x_test=full_bundle.x_test,
            references=full_bundle.references,
            dataset_root=full_bundle.dataset_root,
        ),
        [window.folder_number for window in selected_windows],
    )


def _asymptotic_gauge_check(
    *,
    observations,
    samples,
    config,
    fixed_posterior,
    optimized_posterior,
    predictive_samples: int,
    decay_lengths: float,
    num_points: int,
    tolerance: float,
    output_dir: Path,
) -> dict[str, object]:
    if decay_lengths <= 0.0 or num_points < 2 or tolerance <= 0.0:
        raise ValueError("Gauge-check decay lengths, points, and tolerance must be positive.")

    n_available = samples["theta_ell"].shape[0]
    n_selected = min(predictive_samples, n_available)
    selected_indices = torch.from_numpy(
        np.linspace(0, n_available - 1, n_selected, dtype=int)
    ).to(device=samples["theta_ell"].device, dtype=torch.long)
    sampled_ell = torch.exp(samples["theta_ell"][selected_indices])
    max_ell = max(
        float(sampled_ell.max().detach().cpu().item()),
        float(fixed_posterior.kernel_params["ell"].detach().cpu().item()),
        float(optimized_posterior.kernel_params["ell"].detach().cpu().item()),
    )
    x_all = torch.cat([observations.x_obs, observations.x_der])
    x_min = float(x_all.min().detach().cpu().item())
    x_max = float(x_all.max().detach().cpu().item())
    far_distance = decay_lengths * max_ell
    dtype = observations.x_obs.dtype
    device = observations.x_obs.device
    x_left = torch.linspace(
        x_min - far_distance, x_min, num_points, dtype=dtype, device=device
    )
    x_right = torch.linspace(
        x_max, x_max + far_distance, num_points, dtype=dtype, device=device
    )
    x_tails = torch.cat([x_left, x_right])

    conditional_means, actual_indices = hyperposterior_conditional_means(
        observations,
        samples,
        x_tails,
        config=config,
        max_samples=predictive_samples,
    )
    sampled_ell = torch.exp(samples["theta_ell"][actual_indices])
    fixed_mean = predict_function_mean(fixed_posterior, x_tails)
    optimized_mean = predict_function_mean(optimized_posterior, x_tails)

    left_far = conditional_means[:, 0]
    right_far = conditional_means[:, -1]
    far_values = torch.cat([left_far, right_far])
    max_abs_far = float(far_values.abs().max().detach().cpu().item())
    endpoint_spread = float(far_values.std(unbiased=False).detach().cpu().item())
    diagnostic = {
        "decay_lengths": decay_lengths,
        "max_length_scale_nm": max_ell,
        "far_distance_nm": far_distance,
        "left_boundary_nm": float(x_left[0].detach().cpu().item()),
        "right_boundary_nm": float(x_right[-1].detach().cpu().item()),
        "tolerance_kj_per_mol": tolerance,
        "hyperposterior_max_abs_far_mean": max_abs_far,
        "hyperposterior_far_endpoint_std": endpoint_spread,
        "hyperposterior_left_far_mean": float(left_far.mean().detach().cpu().item()),
        "hyperposterior_right_far_mean": float(right_far.mean().detach().cpu().item()),
        "fixed_left_far_mean": float(fixed_mean[0].detach().cpu().item()),
        "fixed_right_far_mean": float(fixed_mean[-1].detach().cpu().item()),
        "map_left_far_mean": float(optimized_mean[0].detach().cpu().item()),
        "map_right_far_mean": float(optimized_mean[-1].detach().cpu().item()),
        "common_asymptotic_zero_within_tolerance": max_abs_far <= tolerance,
    }

    with (output_dir / "asymptotic_gauge_check.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_index", "ell_nm", "left_far_mean", "right_far_mean"]
        )
        for row, sample_index in enumerate(actual_indices.tolist()):
            writer.writerow(
                [
                    sample_index,
                    float(sampled_ell[row].detach().cpu().item()),
                    float(left_far[row].detach().cpu().item()),
                    float(right_far[row].detach().cpu().item()),
                ]
            )

    x_left_np = x_left.detach().cpu().numpy()
    x_right_np = x_right.detach().cpu().numpy()
    means_np = conditional_means.detach().cpu().numpy()
    fixed_np = fixed_mean.detach().cpu().numpy()
    optimized_np = optimized_mean.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for row in means_np:
        axes[0].plot(x_left_np, row[:num_points], color="tab:green", alpha=0.12)
        axes[1].plot(x_right_np, row[num_points:], color="tab:green", alpha=0.12)
    axes[0].plot(
        x_left_np,
        means_np[:, :num_points].mean(axis=0),
        color="tab:green",
        linewidth=2.5,
        label="Hyperposterior mean",
    )
    axes[1].plot(
        x_right_np,
        means_np[:, num_points:].mean(axis=0),
        color="tab:green",
        linewidth=2.5,
        label="Hyperposterior mean",
    )
    for ax, x_values, fixed_values, optimized_values, edge in (
        (axes[0], x_left_np, fixed_np[:num_points], optimized_np[:num_points], x_min),
        (axes[1], x_right_np, fixed_np[num_points:], optimized_np[num_points:], x_max),
    ):
        ax.plot(x_values, fixed_values, color="tab:blue", linestyle="--", label="Fixed GP")
        ax.plot(x_values, optimized_values, color="tab:orange", linestyle="--", label="MAP GP")
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.axvline(edge, color="gray", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Position [nm]")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Conditional posterior mean [kJ/mol]")
    axes[0].set_title("Left asymptotic tail")
    axes[1].set_title("Right asymptotic tail")
    axes[1].legend()
    fig.suptitle(
        f"Asymptotic gauge check ({decay_lengths:g} maximum length scales)"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "asymptotic_gauge_check.png", dpi=200)
    plt.close(fig)
    return diagnostic


def _plot_summary(ax, summary, *, label: str, color: str) -> None:
    x = summary.x_test.detach().cpu().numpy()
    mean = summary.mean.detach().cpu().numpy()
    std = np.sqrt(
        np.clip(summary.total_variance.detach().cpu().numpy(), 0.0, None)
    )
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(x, mean - 2.0 * std, mean + 2.0 * std, color=color, alpha=0.14)


def _ui_display_offset(refs, target_summary) -> float | None:
    if not refs.has_ui:
        return None
    target_x = target_summary.x_test.detach().cpu().numpy()
    target_mean = target_summary.mean.detach().cpu().numpy()
    target_on_ui = np.interp(refs.umbrella_x, target_x, target_mean)
    return float(np.mean(target_on_ui - refs.umbrella_f))


def _plot_ui_reference(ax, refs, *, offset: float | None) -> None:
    if not refs.has_ui or offset is None:
        return
    ax.errorbar(
        refs.umbrella_x,
        refs.umbrella_f + offset,
        yerr=refs.umbrella_e,
        color="black",
        linewidth=1.5,
        capsize=2,
        alpha=0.75,
        label="Block-averaged UI (offset for display)",
    )


def _finish_axis(ax, *, title: str, ylabel: str) -> None:
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()


def main() -> None:
    args = _parser().parse_args()
    configure_torch(seed=args.seed)
    output_dir = Path(args.results_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle, selected_window_centers = _prepare_training_bundle(args)
    obs = bundle.observations

    fixed_posterior = build_joint_gp(
        x_func=obs.x_obs,
        y_func=obs.y_obs,
        x_der=obs.x_der,
        dy_der=obs.dy_der,
        ell=torch.tensor(FIXED_ELL, dtype=torch.float64),
        w=torch.tensor(FIXED_W, dtype=torch.float64),
        noise_func_cov=obs.noise_func_cov,
        noise_deriv_diag=obs.noise_deriv_diag,
        H_func=obs.H_obs,
        jitter=1e-6,
    )
    fixed_summary = summarize_fixed_posterior_predictive(
        fixed_posterior, bundle.x_test
    )

    optimized = optimize_stationary_hyperparameters(
        obs,
        objective=args.objective,
        steps=args.opt_steps,
        learning_rate=args.opt_learning_rate,
        restarts=args.opt_restarts,
        seed=args.seed,
    )
    optimized_summary = summarize_fixed_posterior_predictive(
        optimized.posterior, bundle.x_test
    )

    nuts_config = NUTSConfig(
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
        target_accept_prob=args.target_accept_prob,
        objective=args.objective,
        kernel="stationary",
    )
    _, samples = run_hmc_nuts(obs, config=nuts_config)
    hyperposterior_summary = summarize_hyperposterior_predictive(
        obs,
        samples,
        bundle.x_test,
        config=nuts_config,
        max_samples=args.predictive_samples,
    )
    gauge_diagnostic = _asymptotic_gauge_check(
        observations=obs,
        samples=samples,
        config=nuts_config,
        fixed_posterior=fixed_posterior,
        optimized_posterior=optimized.posterior,
        predictive_samples=args.predictive_samples,
        decay_lengths=args.gauge_check_decay_lengths,
        num_points=args.gauge_check_points,
        tolerance=args.gauge_check_tolerance,
        output_dir=output_dir,
    )
    derivative_summaries = {
        "fixed_gp": summarize_fixed_posterior_derivative(
            fixed_posterior, bundle.x_test
        ),
        "optimized_plugin_gp": summarize_fixed_posterior_derivative(
            optimized.posterior, bundle.x_test
        ),
        "hyperposterior_gp": summarize_hyperposterior_derivative(
            obs,
            samples,
            bundle.x_test,
            config=nuts_config,
            max_samples=args.predictive_samples,
        ),
    }

    summaries = {
        "fixed_gp": fixed_summary,
        "optimized_plugin_gp": optimized_summary,
        "hyperposterior_gp": hyperposterior_summary,
    }
    metrics = {
        name: compare_to_reference_curves(
            summary, bundle.references, alignment=args.pmf_alignment
        )
        for name, summary in summaries.items()
    }

    refs = bundle.references
    method_styles = {
        "fixed_gp": ("Fixed-hyperparameter GP", "tab:blue"),
        "optimized_plugin_gp": ("MAP plug-in GP", "tab:orange"),
        "hyperposterior_gp": ("Hyperposterior GP", "tab:green"),
    }
    ui_offset = _ui_display_offset(refs, hyperposterior_summary)

    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_ui_reference(ax, refs, offset=ui_offset)
    for name, summary in summaries.items():
        label, color = method_styles[name]
        _plot_summary(ax, summary, label=label, color=color)
    _finish_axis(
        ax,
        title=f"Function UQ in native asymptotic-zero gauge ({args.objective.upper()})",
        ylabel="Free energy [kJ/mol]",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "uq_method_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    derivative_error = np.sqrt(
        np.clip(obs.noise_deriv_diag.detach().cpu().numpy(), 0.0, None)
    )
    ax.errorbar(
        obs.x_der.detach().cpu().numpy(),
        obs.dy_der.detach().cpu().numpy(),
        yerr=derivative_error,
        color="black",
        linestyle="none",
        marker="o",
        markersize=3,
        capsize=2,
        alpha=0.65,
        label="Derivative observations",
    )
    for name, summary in derivative_summaries.items():
        label, color = method_styles[name]
        _plot_summary(ax, summary, label=label, color=color)
    _finish_axis(
        ax,
        title="Gauge-invariant derivative UQ",
        ylabel="Free-energy derivative [kJ/mol/nm]",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "uq_derivatives.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 11), sharex=True)
    _plot_ui_reference(axes[0], refs, offset=ui_offset)
    for name, summary in summaries.items():
        label, color = method_styles[name]
        _plot_summary(axes[0], summary, label=label, color=color)
    _finish_axis(
        axes[0],
        title="Function UQ in native asymptotic-zero gauge",
        ylabel="Free energy [kJ/mol]",
    )
    axes[1].errorbar(
        obs.x_der.detach().cpu().numpy(),
        obs.dy_der.detach().cpu().numpy(),
        yerr=derivative_error,
        color="black",
        linestyle="none",
        marker="o",
        markersize=3,
        capsize=2,
        alpha=0.65,
        label="Derivative observations",
    )
    for name, summary in derivative_summaries.items():
        label, color = method_styles[name]
        _plot_summary(axes[1], summary, label=label, color=color)
    _finish_axis(
        axes[1],
        title="Gauge-invariant derivative UQ",
        ylabel="Free-energy derivative [kJ/mol/nm]",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "uq_representations.png", dpi=200)
    plt.close(fig)

    with (output_dir / "uq_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "representation",
                "method",
                "avg_total_std",
                "avg_within_std",
                "avg_between_std",
            ]
        )
        for representation, representation_summaries in (
            ("asymptotic_gauge_function", summaries),
            ("derivative", derivative_summaries),
        ):
            for name, summary in representation_summaries.items():
                total_std = torch.sqrt(torch.clamp(summary.total_variance, min=0.0))
                within_std = torch.sqrt(torch.clamp(summary.within_variance, min=0.0))
                between_std = torch.sqrt(torch.clamp(summary.between_variance, min=0.0))
                writer.writerow(
                    [
                        representation,
                        name,
                        float(total_std.mean().detach().cpu().item()),
                        float(within_std.mean().detach().cpu().item()),
                        float(between_std.mean().detach().cpu().item()),
                    ]
                )

    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "rmse_wham",
                "rmse_ui",
                "avg_total_std",
                "avg_within_std",
                "avg_between_std",
            ]
        )
        if refs.has_ui:
            ui_rmse_wham = float("nan")
            if refs.has_wham:
                wham_on_ui = np.interp(
                    refs.umbrella_x,
                    refs.wham_x,
                    _shift(refs.wham_f, args.pmf_alignment),
                )
                ui_rmse_wham = float(
                    np.sqrt(
                        np.mean(
                            (
                                _shift(refs.umbrella_f, args.pmf_alignment)
                                - wham_on_ui
                            )
                            ** 2
                        )
                    )
                )
            writer.writerow(
                [
                    "block_averaged_ui",
                    ui_rmse_wham,
                    0.0,
                    (
                        float(np.mean(refs.umbrella_e))
                        if refs.umbrella_e is not None
                        else float("nan")
                    ),
                    float("nan"),
                    float("nan"),
                ]
            )
        for name, result in metrics.items():
            writer.writerow(
                [
                    name,
                    result.rmse_wham,
                    result.rmse_ui,
                    result.avg_total_std,
                    result.avg_within_std,
                    result.avg_between_std,
                ]
            )

    payload = {
        "objective": args.objective,
        "window_count": len(selected_window_centers),
        "selected_window_centers_nm": selected_window_centers,
        "trajectory_fraction": args.trajectory_fraction,
        "selection_mode": "evenly_spaced_windows_contiguous_trajectory_prefix",
        "ui_reference_display_offset_kj_per_mol": ui_offset,
        "ui_reference_display_alignment": (
            "least-squares constant offset to hyperposterior mean"
            if ui_offset is not None
            else None
        ),
        "optimized_plugin_uses_hyperpriors": True,
        "optimized_objective_value": optimized.objective_value,
        "optimized_restart": optimized.restart,
        "optimized_parameters": {
            name: float(value.detach().cpu().item())
            for name, value in optimized.params.items()
        },
        "nuts_samples": args.num_samples,
        "nuts_warmup_steps": args.warmup_steps,
        "predictive_samples": int(
            hyperposterior_summary.selected_indices.numel()
        ),
        "asymptotic_gauge_check": gauge_diagnostic,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="ascii"
    )
    print(f"results: {output_dir}")
    print(f"optimized parameters: {payload['optimized_parameters']}")
    print(
        "common asymptotic zero: "
        f"{gauge_diagnostic['common_asymptotic_zero_within_tolerance']} "
        f"(max abs far mean={gauge_diagnostic['hyperposterior_max_abs_far_mean']:.3g} kJ/mol)"
    )


if __name__ == "__main__":
    main()
