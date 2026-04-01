"""Ablation-grid helpers for window and trajectory knockout studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import ReferenceCurves, UmbrellaWindow, load_reference_curves, load_umbrella_windows
from ..gp import GibbsKernelConfig, build_joint_gp, build_joint_gp_gibbs
from ..hmc import (
    HMCChainDiagnostics,
    NUTSConfig,
    display_samples_for_diagnostics,
    run_hmc_nuts,
    summarize_chain_diagnostics,
)
from ..metrics import ReferenceComparison, compare_to_reference_curves
from ..posterior import (
    HyperposteriorPredictiveSummary,
    summarize_fixed_posterior_predictive,
    summarize_hyperposterior_predictive,
)
from ..preprocess import build_joint_observations, build_test_grid, process_umbrella_windows
from ..workflow import WorkflowBundle


@dataclass(frozen=True)
class AblationCell:
    window_count: int
    trajectory_fraction: float


@dataclass(frozen=True)
class StudyModelConfig:
    method: str = "fixed_gp"
    kernel: str = "stationary"
    length_model: str = "exp_linear_bump"
    width_model: str = "tanh_decay"
    ell: float = 4.0
    w: float = 3.3
    a0: float = float(np.log(4.0))
    a1: float = 0.0
    b: float = 0.0
    c: float | None = None
    length_w: float = 0.5
    s: float = 1.65
    u: float | None = None
    w2: float = 0.5
    jitter: float = 1e-6
    objective: str = "lml"
    warmup_steps: int = 20
    num_samples: int = 20
    num_chains: int = 4
    target_accept_prob: float = 0.8
    predictive_samples: int = 20


@dataclass(frozen=True)
class AblationCellResult:
    cell: AblationCell
    metrics: ReferenceComparison
    dataset_root: Path
    x_test: torch.Tensor
    predictive_summary: HyperposteriorPredictiveSummary
    chain_diagnostics: HMCChainDiagnostics | None = None
    nuts_samples: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class AblationStudyResult:
    model: StudyModelConfig
    window_counts: list[int]
    trajectory_fractions: list[float]
    cells: list[AblationCellResult]
    references: ReferenceCurves
    test_grid_mode: str


def _select_evenly_spaced_windows(windows: list[UmbrellaWindow], keep_count: int) -> list[UmbrellaWindow]:
    if keep_count <= 0:
        raise ValueError("keep_count must be positive.")
    if keep_count >= len(windows):
        return list(windows)
    indices = np.linspace(0, len(windows) - 1, keep_count, dtype=int)
    indices = np.unique(indices)
    return [windows[idx] for idx in indices.tolist()]


def _truncate_window(
    window: UmbrellaWindow,
    *,
    retain_fraction: float,
    n_equilibration: int,
) -> UmbrellaWindow:
    if not (0.0 < retain_fraction <= 1.0):
        raise ValueError("retain_fraction must be in (0, 1].")
    time_eq = window.time[n_equilibration:] if len(window.time) > n_equilibration else window.time
    position_eq = window.position[n_equilibration:] if len(window.position) > n_equilibration else window.position
    if position_eq.numel() == 0:
        raise ValueError(f"Window {window.folder} has no usable data after equilibration.")
    n_keep = max(2, int(np.floor(position_eq.numel() * retain_fraction)))
    n_keep = min(int(position_eq.numel()), n_keep)
    return UmbrellaWindow(
        folder=window.folder,
        folder_number=window.folder_number,
        time=time_eq[:n_keep].clone(),
        position=position_eq[:n_keep].clone(),
        mdp_last_key=window.mdp_last_key,
        mdp_last_value=window.mdp_last_value,
    )


def _prepare_ablation_bundle(
    windows: list[UmbrellaWindow],
    references: ReferenceCurves,
    dataset_root: Path,
    *,
    cell: AblationCell,
    num_bins: int,
    num_test_points: int,
    test_grid_source: str,
    x_min: float | None,
    x_max: float | None,
    n_equilibration: int,
    common_x_test: torch.Tensor | None = None,
) -> WorkflowBundle:
    selected = _select_evenly_spaced_windows(windows, cell.window_count)
    truncated = [
        _truncate_window(window, retain_fraction=cell.trajectory_fraction, n_equilibration=n_equilibration)
        for window in selected
    ]
    processed = process_umbrella_windows(truncated, n_equilibration=0, num_bins=num_bins)
    observations = build_joint_observations(processed)
    if common_x_test is None:
        x_test = build_test_grid(
            processed,
            num_points=num_test_points,
            x_min=x_min,
            x_max=x_max,
            source=test_grid_source,
        )
    else:
        x_test = common_x_test.clone()
    return WorkflowBundle(
        processed=processed,
        observations=observations,
        x_test=x_test,
        references=references,
        dataset_root=dataset_root,
    )


def _fixed_summary(bundle: WorkflowBundle, model: StudyModelConfig):
    obs = bundle.observations
    if model.kernel == "stationary":
        posterior = build_joint_gp(
            x_func=obs.x_obs,
            y_func=obs.y_obs,
            x_der=obs.x_der,
            dy_der=obs.dy_der,
            ell=torch.tensor(model.ell, dtype=torch.float64),
            w=torch.tensor(model.w, dtype=torch.float64),
            noise_func_cov=obs.noise_func_cov,
            noise_deriv_diag=obs.noise_deriv_diag,
            H_func=obs.H_obs,
            jitter=model.jitter,
        )
    else:
        midpoint = float(0.5 * (obs.x_obs.min().item() + obs.x_obs.max().item()))
        posterior = build_joint_gp_gibbs(
            x_func=obs.x_obs,
            y_func=obs.y_obs,
            x_der=obs.x_der,
            dy_der=obs.dy_der,
            a0=torch.tensor(model.a0, dtype=torch.float64),
            a1=torch.tensor(model.a1, dtype=torch.float64),
            b=torch.tensor(model.b, dtype=torch.float64),
            c=torch.tensor(model.c if model.c is not None else midpoint, dtype=torch.float64),
            length_w=torch.tensor(model.length_w, dtype=torch.float64),
            s=torch.tensor(model.s, dtype=torch.float64),
            u=torch.tensor(model.u if model.u is not None else midpoint, dtype=torch.float64),
            width_w=torch.tensor(model.w2, dtype=torch.float64),
            noise_func_cov=obs.noise_func_cov,
            noise_deriv_diag=obs.noise_deriv_diag,
            H_func=obs.H_obs,
            config=GibbsKernelConfig(
                length_model=model.length_model,
                width_model=model.width_model,
            ),
            jitter=model.jitter,
        )
    return summarize_fixed_posterior_predictive(posterior, bundle.x_test)


def _nuts_summary(bundle: WorkflowBundle, model: StudyModelConfig):
    config = NUTSConfig(
        num_samples=model.num_samples,
        warmup_steps=model.warmup_steps,
        num_chains=model.num_chains,
        target_accept_prob=model.target_accept_prob,
        jitter=model.jitter,
        objective=model.objective,
        kernel=model.kernel,
        length_model=model.length_model,
        width_model=model.width_model,
    )
    mcmc, samples = run_hmc_nuts(bundle.observations, config=config)
    summary = summarize_hyperposterior_predictive(
        bundle.observations,
        samples,
        bundle.x_test,
        config=config,
        max_samples=model.predictive_samples,
    )
    diagnostics = summarize_chain_diagnostics(mcmc, samples, config=config)
    return summary, diagnostics, samples


def run_ablation_study(
    *,
    dataset_root: str,
    project_root: str | None = None,
    window_counts: list[int],
    trajectory_fractions: list[float],
    model: StudyModelConfig | None = None,
    n_equilibration: int = 40_000,
    num_bins: int = 20,
    num_test_points: int = 200,
    test_grid_source: str = "umbrella_centers",
    x_min: float | None = None,
    x_max: float | None = None,
    test_grid_mode: str = "full_dataset",
) -> AblationStudyResult:
    model = model or StudyModelConfig()
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    windows = load_umbrella_windows(dataset_root_path)
    references = load_reference_curves(project_root)

    common_x_test = None
    if test_grid_mode == "full_dataset":
        full_processed = process_umbrella_windows(
            windows,
            n_equilibration=n_equilibration,
            num_bins=num_bins,
        )
        common_x_test = build_test_grid(
            full_processed,
            num_points=num_test_points,
            x_min=x_min,
            x_max=x_max,
            source=test_grid_source,
        )
    elif test_grid_mode != "per_cell":
        raise ValueError(f"Unsupported test_grid_mode: {test_grid_mode}")

    cells: list[AblationCellResult] = []
    for window_count in window_counts:
        for trajectory_fraction in trajectory_fractions:
            cell = AblationCell(window_count=window_count, trajectory_fraction=trajectory_fraction)
            bundle = _prepare_ablation_bundle(
                windows,
                references,
                dataset_root_path,
                cell=cell,
                num_bins=num_bins,
                num_test_points=num_test_points,
                test_grid_source=test_grid_source,
                x_min=x_min,
                x_max=x_max,
                n_equilibration=n_equilibration,
                common_x_test=common_x_test,
            )
            if model.method == "fixed_gp":
                summary = _fixed_summary(bundle, model)
                diagnostics = None
                nuts_samples = None
            elif model.method == "nuts":
                summary, diagnostics, nuts_samples = _nuts_summary(bundle, model)
            else:
                raise ValueError(f"Unsupported ablation method: {model.method}")

            metrics = compare_to_reference_curves(summary, references)
            cells.append(
                AblationCellResult(
                    cell=cell,
                    metrics=metrics,
                    dataset_root=dataset_root_path,
                    x_test=bundle.x_test,
                    predictive_summary=summary,
                    chain_diagnostics=diagnostics,
                    nuts_samples=nuts_samples,
                )
            )

    return AblationStudyResult(
        model=model,
        window_counts=list(window_counts),
        trajectory_fractions=list(trajectory_fractions),
        cells=cells,
        references=references,
        test_grid_mode=test_grid_mode,
    )


def _metric_grid(result: AblationStudyResult, metric_name: str) -> np.ndarray:
    grid = np.full((len(result.window_counts), len(result.trajectory_fractions)), np.nan, dtype=float)
    lookup = {
        (cell.cell.window_count, cell.cell.trajectory_fraction): getattr(cell.metrics, metric_name)
        for cell in result.cells
    }
    for i, window_count in enumerate(result.window_counts):
        for j, trajectory_fraction in enumerate(result.trajectory_fractions):
            grid[i, j] = lookup[(window_count, trajectory_fraction)]
    return grid


def _shift_curve(y: np.ndarray) -> np.ndarray:
    return y - np.max(y)


def _cell_slug(cell: AblationCell) -> str:
    frac = f"{cell.trajectory_fraction:.2f}".replace(".", "p")
    return f"w{cell.window_count:02d}_f{frac}"


def _plot_predictive_cell(
    ax,
    cell_result: AblationCellResult,
    references: ReferenceCurves,
) -> None:
    summary = cell_result.predictive_summary
    x_test = summary.x_test.detach().cpu().numpy().reshape(-1)
    pred_mean = summary.mean.detach().cpu().numpy().reshape(-1)
    pred_std = np.sqrt(np.clip(summary.total_variance.detach().cpu().numpy().reshape(-1), a_min=0.0, a_max=None))
    shifted_mean = _shift_curve(pred_mean)

    wham_shift = _shift_curve(references.wham_f)
    ui_shift = _shift_curve(references.umbrella_f)

    ax.plot(x_test, shifted_mean, lw=2, color="royalblue")
    ax.fill_between(
        x_test,
        shifted_mean - 2.0 * pred_std,
        shifted_mean + 2.0 * pred_std,
        color="royalblue",
        alpha=0.2,
    )
    ax.plot(references.wham_x, wham_shift, color="crimson", alpha=0.7, lw=1.2)
    ax.plot(references.umbrella_x, ui_shift, color="steelblue", alpha=0.7, lw=1.2)
    ax.set_title(
        f"{cell_result.cell.window_count} windows, {cell_result.cell.trajectory_fraction:.2f} traj\n"
        f"RMSE(WHAM)={cell_result.metrics.rmse_wham:.2f}, avg std={cell_result.metrics.avg_total_std:.2f}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.15)


def _save_predictive_figures(
    result: AblationStudyResult,
    figure_dir: Path,
) -> None:
    predictive_dir = figure_dir / "predictive_cells"
    predictive_dir.mkdir(parents=True, exist_ok=True)

    for cell_result in result.cells:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_predictive_cell(ax, cell_result, result.references)
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Shifted free energy [kJ/mol]")
        fig.tight_layout()
        fig.savefig(predictive_dir / f"{_cell_slug(cell_result.cell)}.png", dpi=200)
        plt.close(fig)

    n_rows = len(result.window_counts)
    n_cols = len(result.trajectory_fractions)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.8 * n_rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    lookup = {
        (cell.cell.window_count, cell.cell.trajectory_fraction): cell
        for cell in result.cells
    }
    for i, window_count in enumerate(result.window_counts):
        for j, trajectory_fraction in enumerate(result.trajectory_fractions):
            ax = axes[i, j]
            cell_result = lookup[(window_count, trajectory_fraction)]
            _plot_predictive_cell(ax, cell_result, result.references)
            if i == n_rows - 1:
                ax.set_xlabel("Position [nm]")
            if j == 0:
                ax.set_ylabel("Shifted free energy [kJ/mol]")
    fig.suptitle(
        f"Ablation Predictive Curves ({result.model.method}, {result.model.kernel})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "ablation_predictive_grid.png", dpi=200)
    plt.close(fig)


def _nuts_config_from_model(model: StudyModelConfig) -> NUTSConfig:
    return NUTSConfig(
        num_samples=model.num_samples,
        warmup_steps=model.warmup_steps,
        num_chains=model.num_chains,
        target_accept_prob=model.target_accept_prob,
        jitter=model.jitter,
        objective=model.objective,
        kernel=model.kernel,
        length_model=model.length_model,
        width_model=model.width_model,
    )


def _plot_trace_panel(samples: dict[str, torch.Tensor], output_path: Path) -> None:
    names = list(samples.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(10, 3 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        values = samples[name].detach().cpu().numpy().ravel()
        axes[row, 0].plot(values, lw=1.0)
        axes[row, 0].set_title(f"{name} trace")
        axes[row, 1].hist(values, bins=min(20, max(5, len(values))), color="gray", alpha=0.8)
        axes[row, 1].set_title(f"{name} histogram")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_corner_panel(
    samples: dict[str, torch.Tensor],
    output_path: Path,
    *,
    config: NUTSConfig,
) -> bool:
    try:
        import corner
    except ImportError:
        return False

    chain, labels = display_samples_for_diagnostics(samples, config=config)
    chain_np = chain.detach().cpu().numpy()
    if chain_np.ndim != 2 or chain_np.shape[0] <= chain_np.shape[1]:
        return False

    figure = corner.corner(
        chain_np,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 11},
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return True


def _save_nuts_diagnostics(
    result: AblationStudyResult,
    figure_dir: Path,
) -> None:
    if result.model.method != "nuts":
        return

    config = _nuts_config_from_model(result.model)
    diagnostics_dir = figure_dir / "nuts_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for cell_result in result.cells:
        if cell_result.nuts_samples is None:
            continue
        cell_dir = diagnostics_dir / _cell_slug(cell_result.cell)
        cell_dir.mkdir(parents=True, exist_ok=True)
        _plot_trace_panel(cell_result.nuts_samples, cell_dir / "nuts_traces.png")
        _plot_corner_panel(
            cell_result.nuts_samples,
            cell_dir / "corner.png",
            config=config,
        )


def save_ablation_summary(
    result: AblationStudyResult,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    _save_predictive_figures(result, figure_dir)
    _save_nuts_diagnostics(result, figure_dir)

    metric_specs = [
        ("rmse_wham", "RMSE vs WHAM"),
        ("rmse_ui", "RMSE vs UI"),
        ("avg_total_std", "Average total std (common grid)"),
        ("avg_between_std", "Average between std"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (metric_name, title) in zip(axes.flat, metric_specs):
        grid = _metric_grid(result, metric_name)
        image = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(result.trajectory_fractions)))
        ax.set_xticklabels([f"{value:.2f}" for value in result.trajectory_fractions])
        ax.set_yticks(range(len(result.window_counts)))
        ax.set_yticklabels([str(value) for value in result.window_counts])
        ax.set_xlabel("Trajectory fraction retained")
        ax.set_ylabel("Windows retained")
        ax.set_title(title)
        ax.invert_xaxis()
        fig.colorbar(image, ax=ax, shrink=0.85)
    fig.suptitle(
        f"Ablation Summary ({result.model.method}, {result.model.kernel})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "ablation_summary.png", dpi=200)
    plt.close(fig)

    with (figure_dir / "ablation_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "window_count",
                "trajectory_fraction",
                "rmse_wham",
                "rmse_ui",
                "avg_total_std",
                "avg_within_std",
                "avg_between_std",
                "avg_total_variance",
                "avg_within_variance",
                "avg_between_variance",
                "hmc_step_size",
                "hmc_mean_accept_prob",
                "hmc_accept_count",
                "hmc_divergence_count",
                "hmc_mean_sample_std",
                "hmc_max_sample_std",
                "hmc_min_sample_std",
                "hmc_poor_acceptance",
                "hmc_looks_stuck",
            ]
        )
        for cell in result.cells:
            diagnostics = cell.chain_diagnostics
            writer.writerow(
                [
                    cell.cell.window_count,
                    cell.cell.trajectory_fraction,
                    cell.metrics.rmse_wham,
                    cell.metrics.rmse_ui,
                    cell.metrics.avg_total_std,
                    cell.metrics.avg_within_std,
                    cell.metrics.avg_between_std,
                    cell.metrics.avg_total_variance,
                    cell.metrics.avg_within_variance,
                    cell.metrics.avg_between_variance,
                    "" if diagnostics is None else diagnostics.step_size,
                    "" if diagnostics is None else diagnostics.mean_accept_prob,
                    "" if diagnostics is None else diagnostics.accept_count,
                    "" if diagnostics is None else diagnostics.divergence_count,
                    "" if diagnostics is None else diagnostics.mean_sample_std,
                    "" if diagnostics is None else diagnostics.max_sample_std,
                    "" if diagnostics is None else diagnostics.min_sample_std,
                    "" if diagnostics is None else diagnostics.poor_acceptance,
                    "" if diagnostics is None else diagnostics.looks_stuck,
                ]
            )

    lines = [
        f"method: {result.model.method}",
        f"kernel: {result.model.kernel}",
    ]
    if result.model.kernel == "stationary":
        lines.extend(
            [
                f"ell: {result.model.ell}",
                f"w: {result.model.w}",
            ]
        )
    else:
        lines.extend(
            [
                f"length_model: {result.model.length_model}",
                f"width_model: {result.model.width_model}",
                f"a0: {result.model.a0}",
                f"a1: {result.model.a1}",
                f"b: {result.model.b}",
                f"c: {result.model.c}",
                f"length_w: {result.model.length_w}",
                f"s: {result.model.s}",
                f"u: {result.model.u}",
                f"w2: {result.model.w2}",
            ]
        )
    lines.extend(
        [
            f"window_counts: {result.window_counts}",
            f"trajectory_fractions: {result.trajectory_fractions}",
            f"test_grid_mode: {result.test_grid_mode}",
            "uncertainty_summary: average total standard deviation on the prediction grid",
            f"predictive_cell_dir: {figure_dir / 'predictive_cells'}",
        ]
    )
    if result.model.method == "nuts":
        stuck_count = sum(
            1 for cell in result.cells
            if cell.chain_diagnostics is not None and cell.chain_diagnostics.looks_stuck
        )
        lines.append(f"stuck_cells: {stuck_count}")
        lines.append(f"nuts_diagnostics_dir: {figure_dir / 'nuts_diagnostics'}")
    (figure_dir / "run_summary.txt").write_text("\n".join(lines) + "\n")
