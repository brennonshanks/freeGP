"""Ablation-grid helpers for window and trajectory knockout studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import math

from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data import ReferenceCurves, UmbrellaWindow, load_reference_curves, load_umbrella_windows
from ..gp import GibbsKernelConfig, build_joint_gp, build_joint_gp_gibbs
from ..hyperopt import optimize_stationary_hyperparameters
from ..hmc import (
    HMCChainDiagnostics,
    NUTSConfig,
    display_samples_for_diagnostics,
    run_hmc_nuts,
    summarize_chain_diagnostics,
)
from ..config import resolve_device
from ..metrics import ReferenceComparison, compare_to_reference_curves
from ..posterior import (
    HyperposteriorPredictiveSummary,
    summarize_fixed_posterior_predictive,
    summarize_hyperposterior_predictive,
)
from ..preprocess import build_joint_observations, build_test_grid, process_umbrella_windows
from ..workflow import WorkflowBundle, move_workflow_bundle

CSANYI_FIXED_ELL = float(np.pi / 2.0)
CSANYI_FIXED_W = float(4.184 * np.sqrt(10.0))


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
    ell: float = CSANYI_FIXED_ELL
    w: float = CSANYI_FIXED_W
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
    num_chains: int = 1
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    predictive_samples: int = 20
    barrier_bins: int = 30
    selection_replicates: int | None = None
    fixed_noise: bool = False
    opt_steps: int = 250
    opt_restarts: int = 3
    opt_learning_rate: float = 0.05


@dataclass(frozen=True)
class AblationCellResult:
    cell: AblationCell
    replicate_count: int
    metrics: ReferenceComparison
    dataset_root: Path
    x_test: torch.Tensor
    predictive_summary: HyperposteriorPredictiveSummary
    canonical_predictive_summary: HyperposteriorPredictiveSummary | None = None
    canonical_metrics: ReferenceComparison | None = None
    chain_diagnostics: HMCChainDiagnostics | None = None
    multi_chain_diagnostics: dict[str, object] | None = None
    nuts_samples: dict[str, torch.Tensor] | None = None
    nuts_grouped_samples: dict[str, torch.Tensor] | None = None
    canonical_chain_diagnostics: HMCChainDiagnostics | None = None
    canonical_multi_chain_diagnostics: dict[str, object] | None = None
    canonical_nuts_samples: dict[str, torch.Tensor] | None = None
    canonical_nuts_grouped_samples: dict[str, torch.Tensor] | None = None
    artifact_payload: dict[str, object] | None = None


@dataclass(frozen=True)
class AblationStudyResult:
    model: StudyModelConfig
    window_counts: list[int]
    trajectory_fractions: list[float]
    cells: list[AblationCellResult]
    references: ReferenceCurves
    test_grid_mode: str
    window_selection_mode: str
    trajectory_selection_mode: str
    random_seed: int
    device: str
    pmf_alignment: str = "max"


def _select_evenly_spaced_windows(windows: list[UmbrellaWindow], keep_count: int) -> list[UmbrellaWindow]:
    if keep_count <= 0:
        raise ValueError("keep_count must be positive.")
    if keep_count >= len(windows):
        return list(windows)
    indices = np.linspace(0, len(windows) - 1, keep_count, dtype=int)
    indices = np.unique(indices)
    return [windows[idx] for idx in indices.tolist()]


def _select_random_windows(
    windows: list[UmbrellaWindow],
    keep_count: int,
    *,
    rng: np.random.Generator,
) -> list[UmbrellaWindow]:
    if keep_count <= 0:
        raise ValueError("keep_count must be positive.")
    if keep_count >= len(windows):
        return list(windows)
    indices = np.sort(rng.choice(len(windows), size=keep_count, replace=False))
    return [windows[int(idx)] for idx in indices.tolist()]


def _select_windows(
    windows: list[UmbrellaWindow],
    keep_count: int,
    *,
    mode: str,
    rng: np.random.Generator,
) -> list[UmbrellaWindow]:
    if mode == "evenly_spaced":
        return _select_evenly_spaced_windows(windows, keep_count)
    if mode == "random_subset":
        return _select_random_windows(windows, keep_count, rng=rng)
    raise ValueError(f"Unsupported window_selection_mode: {mode}")


def _truncate_window(
    window: UmbrellaWindow,
    *,
    retain_fraction: float,
    n_equilibration: int,
    mode: str,
    rng: np.random.Generator,
) -> UmbrellaWindow:
    if not (0.0 < retain_fraction <= 1.0):
        raise ValueError("retain_fraction must be in (0, 1].")
    time_eq = window.time[n_equilibration:] if len(window.time) > n_equilibration else window.time
    position_eq = window.position[n_equilibration:] if len(window.position) > n_equilibration else window.position
    if position_eq.numel() == 0:
        raise ValueError(f"Window {window.folder} has no usable data after equilibration.")
    n_keep = max(2, int(np.floor(position_eq.numel() * retain_fraction)))
    n_keep = min(int(position_eq.numel()), n_keep)
    if mode == "contiguous":
        keep_indices = torch.arange(n_keep, dtype=torch.long)
    elif mode == "random_subsample":
        chosen = np.sort(rng.choice(int(position_eq.numel()), size=n_keep, replace=False))
        keep_indices = torch.as_tensor(chosen, dtype=torch.long)
    else:
        raise ValueError(f"Unsupported trajectory_selection_mode: {mode}")
    return UmbrellaWindow(
        folder=window.folder,
        folder_number=window.folder_number,
        time=time_eq[keep_indices].clone(),
        position=position_eq[keep_indices].clone(),
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
    window_selection_mode: str = "evenly_spaced",
    trajectory_selection_mode: str = "contiguous",
    rng: np.random.Generator | None = None,
) -> WorkflowBundle:
    rng = rng or np.random.default_rng()
    selected = _select_windows(
        windows,
        cell.window_count,
        mode=window_selection_mode,
        rng=rng,
    )
    truncated = [
        _truncate_window(
            window,
            retain_fraction=cell.trajectory_fraction,
            n_equilibration=n_equilibration,
            mode=trajectory_selection_mode,
            rng=rng,
        )
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


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(val) for val in value]
    return value


def _replicate_plan(
    *,
    effective_replicates: int,
    window_selection_mode: str,
    trajectory_selection_mode: str,
) -> list[dict[str, object]]:
    random_modes_active = (
        window_selection_mode == "random_subset"
        or trajectory_selection_mode == "random_subsample"
    )
    if not random_modes_active:
        return [
            {
                "replicate_index": 0,
                "is_canonical": True,
                "window_selection_mode": window_selection_mode,
                "trajectory_selection_mode": trajectory_selection_mode,
            }
        ]

    plan = [
        {
            "replicate_index": 0,
            "is_canonical": True,
            "window_selection_mode": "evenly_spaced",
            "trajectory_selection_mode": "contiguous",
        }
    ]
    for rep in range(1, effective_replicates):
        plan.append(
            {
                "replicate_index": rep,
                "is_canonical": False,
                "window_selection_mode": window_selection_mode,
                "trajectory_selection_mode": trajectory_selection_mode,
            }
        )
    return plan


def _bundle_artifact_payload(bundle: WorkflowBundle) -> dict[str, object]:
    processed = bundle.processed
    observations = bundle.observations
    return {
        "dataset_root": str(bundle.dataset_root),
        "processed": {
            "folder_numbers": _cpu_clone(processed.folder_numbers),
            "force_constants": _cpu_clone(processed.force_constants),
            "modes": _cpu_clone(processed.modes),
            "variances": _cpu_clone(processed.variances),
            "autocorr_times": _cpu_clone(processed.autocorr_times),
            "n_samples": _cpu_clone(processed.n_samples),
            "restoring_forces": _cpu_clone(processed.restoring_forces),
            "histogram_counts": _cpu_clone(processed.histogram_counts),
            "histogram_probs": _cpu_clone(processed.histogram_probs),
            "histogram_densities": _cpu_clone(processed.histogram_densities),
            "bin_centers_list": _cpu_clone(processed.bin_centers_list),
        },
        "observations": {
            "x_obs": _cpu_clone(observations.x_obs),
            "y_obs": _cpu_clone(observations.y_obs),
            "H_obs": _cpu_clone(observations.H_obs),
            "x_der": _cpu_clone(observations.x_der),
            "dy_der": _cpu_clone(observations.dy_der),
            "noise_func_cov": _cpu_clone(observations.noise_func_cov),
            "noise_deriv_diag": _cpu_clone(observations.noise_deriv_diag),
            "F_list": _cpu_clone(observations.F_list),
        },
        "x_test": _cpu_clone(bundle.x_test),
    }


def _average_metrics(metrics_list: list[ReferenceComparison]) -> ReferenceComparison:
    return ReferenceComparison(
        rmse_wham=float(np.mean([m.rmse_wham for m in metrics_list])),
        rmse_ui=float(np.mean([m.rmse_ui for m in metrics_list])),
        avg_total_std=float(np.mean([m.avg_total_std for m in metrics_list])),
        avg_within_std=float(np.mean([m.avg_within_std for m in metrics_list])),
        avg_between_std=float(np.mean([m.avg_between_std for m in metrics_list])),
        avg_total_variance=float(np.mean([m.avg_total_variance for m in metrics_list])),
        avg_within_variance=float(np.mean([m.avg_within_variance for m in metrics_list])),
        avg_between_variance=float(np.mean([m.avg_between_variance for m in metrics_list])),
    )


def _aggregate_predictive_summaries(
    summaries: list[HyperposteriorPredictiveSummary],
) -> HyperposteriorPredictiveSummary:
    conditional_means = torch.cat([summary.conditional_means for summary in summaries], dim=0)
    conditional_covariances = torch.cat([summary.conditional_covariances for summary in summaries], dim=0)
    mean = conditional_means.mean(dim=0)
    within_cov = conditional_covariances.mean(dim=0)
    centered = conditional_means - mean
    between_cov = centered.T @ centered / conditional_means.shape[0]
    total_cov = within_cov + between_cov
    total_cov = 0.5 * (total_cov + total_cov.T)
    within_cov = 0.5 * (within_cov + within_cov.T)
    between_cov = 0.5 * (between_cov + between_cov.T)
    selected_indices = torch.arange(conditional_means.shape[0], dtype=torch.long)
    return HyperposteriorPredictiveSummary(
        x_test=summaries[0].x_test.clone(),
        mean=mean,
        total_cov=total_cov,
        within_cov=within_cov,
        between_cov=between_cov,
        conditional_means=conditional_means,
        conditional_covariances=conditional_covariances,
        selected_indices=selected_indices,
    )


def _aggregate_chain_diagnostics(
    diagnostics_list: list[HMCChainDiagnostics],
) -> HMCChainDiagnostics:
    labels = diagnostics_list[0].sample_std_by_name.keys()
    sample_std_by_name = {
        label: float(np.mean([diag.sample_std_by_name[label] for diag in diagnostics_list]))
        for label in labels
    }
    return HMCChainDiagnostics(
        step_size=float(np.mean([diag.step_size for diag in diagnostics_list])),
        mean_accept_prob=float(np.mean([diag.mean_accept_prob for diag in diagnostics_list])),
        accept_count=int(np.mean([diag.accept_count for diag in diagnostics_list])),
        divergence_count=int(np.mean([diag.divergence_count for diag in diagnostics_list])),
        sample_std_by_name=sample_std_by_name,
        mean_sample_std=float(np.mean([diag.mean_sample_std for diag in diagnostics_list])),
        max_sample_std=float(np.mean([diag.max_sample_std for diag in diagnostics_list])),
        min_sample_std=float(np.mean([diag.min_sample_std for diag in diagnostics_list])),
        poor_acceptance=bool(any(diag.poor_acceptance for diag in diagnostics_list)),
        looks_stuck=bool(any(diag.looks_stuck for diag in diagnostics_list)),
    )


def _diagnostic_value_to_python(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            scalar = float(value.detach().cpu().item())
            return None if not math.isfinite(scalar) else scalar
        return _diagnostic_value_to_python(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _diagnostic_value_to_python(value.tolist())
    if isinstance(value, dict):
        return {str(key): _diagnostic_value_to_python(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value_to_python(item) for item in value]
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return None if not math.isfinite(scalar) else scalar
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _flatten_numeric_values(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: list[float] = []
        for item in value.values():
            out.extend(_flatten_numeric_values(item))
        return out
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric_values(item))
        return out
    if isinstance(value, (int, float)):
        scalar = float(value)
        return [scalar] if math.isfinite(scalar) else []
    return []


def _divergence_count(value) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(_divergence_count(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return int(value)


def _extract_multi_chain_diagnostics(mcmc) -> dict[str, object]:
    diagnostics = _diagnostic_value_to_python(mcmc.diagnostics())
    parameter_diags = {
        key: value for key, value in diagnostics.items()
        if isinstance(value, dict) and "r_hat" in value and "n_eff" in value
    }
    r_hats: list[float] = []
    n_effs: list[float] = []
    for value in parameter_diags.values():
        r_hats.extend(_flatten_numeric_values(value.get("r_hat")))
        n_effs.extend(_flatten_numeric_values(value.get("n_eff")))
    divergence_map = diagnostics.get("divergences", {})
    divergence_total = _divergence_count(divergence_map)
    return {
        "raw": diagnostics,
        "summary": {
            "max_r_hat": max(r_hats) if r_hats else None,
            "min_r_hat": min(r_hats) if r_hats else None,
            "min_n_eff": min(n_effs) if n_effs else None,
            "max_n_eff": max(n_effs) if n_effs else None,
            "divergence_total": divergence_total,
            "num_diagnostic_parameters": len(parameter_diags),
        },
    }


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


def _optimized_summary(bundle: WorkflowBundle, model: StudyModelConfig):
    if model.kernel != "stationary":
        raise ValueError("optimized_gp currently supports only the stationary kernel.")
    optimized = optimize_stationary_hyperparameters(
        bundle.observations,
        objective=model.objective,
        steps=model.opt_steps,
        learning_rate=model.opt_learning_rate,
        restarts=model.opt_restarts,
        jitter=model.jitter,
    )
    return summarize_fixed_posterior_predictive(optimized.posterior, bundle.x_test)


def _nuts_summary(bundle: WorkflowBundle, model: StudyModelConfig, *, seed: int | None = None):
    config = NUTSConfig(
        num_samples=model.num_samples,
        warmup_steps=model.warmup_steps,
        num_chains=model.num_chains,
        target_accept_prob=model.target_accept_prob,
        max_tree_depth=model.max_tree_depth,
        seed=seed,
        jitter=model.jitter,
        objective=model.objective,
        kernel=model.kernel,
        length_model=model.length_model,
        width_model=model.width_model,
        fixed_noise=model.fixed_noise,
    )
    mcmc, samples = run_hmc_nuts(bundle.observations, config=config)
    grouped_samples = mcmc.get_samples(group_by_chain=True)
    summary = summarize_hyperposterior_predictive(
        bundle.observations,
        samples,
        bundle.x_test,
        config=config,
        max_samples=model.predictive_samples,
    )
    diagnostics = summarize_chain_diagnostics(mcmc, samples, config=config)
    multi_chain_diagnostics = _extract_multi_chain_diagnostics(mcmc)
    return summary, diagnostics, multi_chain_diagnostics, samples, grouped_samples


def _predictive_summary_payload(summary: HyperposteriorPredictiveSummary) -> dict[str, object]:
    return {
        "x_test": _cpu_clone(summary.x_test),
        "mean": _cpu_clone(summary.mean),
        "total_variance": _cpu_clone(summary.total_variance),
        "within_variance": _cpu_clone(summary.within_variance),
        "between_variance": _cpu_clone(summary.between_variance),
        "conditional_means": _cpu_clone(summary.conditional_means),
        "selected_indices": _cpu_clone(summary.selected_indices),
    }


def _replicate_result_payload(
    *,
    replicate_index: int,
    is_canonical: bool,
    window_selection_mode: str,
    trajectory_selection_mode: str,
    random_seed: int,
    hmc_seed: int | None = None,
    bundle: WorkflowBundle,
    summary: HyperposteriorPredictiveSummary,
    metrics: ReferenceComparison,
    diagnostics: HMCChainDiagnostics | None,
    multi_chain_diagnostics: dict[str, object] | None,
    nuts_samples: dict[str, torch.Tensor] | None,
    nuts_grouped_samples: dict[str, torch.Tensor] | None,
) -> dict[str, object]:
    return {
        "replicate_index": replicate_index,
        "is_canonical": is_canonical,
        "window_selection_mode": window_selection_mode,
        "trajectory_selection_mode": trajectory_selection_mode,
        "random_seed": random_seed,
        "hmc_seed": hmc_seed,
        "bundle": _bundle_artifact_payload(bundle),
        "predictive_summary": _predictive_summary_payload(summary),
        "metrics": asdict(metrics),
        "chain_diagnostics": None if diagnostics is None else asdict(diagnostics),
        "multi_chain_diagnostics": multi_chain_diagnostics,
        "nuts_samples": _cpu_clone(nuts_samples),
        "nuts_grouped_samples": _cpu_clone(nuts_grouped_samples),
    }


def run_ablation_study(
    *,
    dataset_root: str,
    project_root: str | None = None,
    reference_wham_path: str | None = None,
    reference_wham_x_units: str = "nm",
    reference_ui_path: str | None = None,
    reference_ui_x_units: str = "nm",
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
    window_selection_mode: str = "evenly_spaced",
    trajectory_selection_mode: str = "contiguous",
    random_seed: int = 0,
    device: str = "cpu",
    pmf_alignment: str = "max",
    checkpoint_dir: Path | str | None = None,
    resume: bool = False,
) -> AblationStudyResult:
    model = model or StudyModelConfig()
    resolved_device = resolve_device(device)
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    windows = load_umbrella_windows(dataset_root_path)
    references = load_reference_curves(
        project_root,
        wham_path=reference_wham_path,
        wham_x_units=reference_wham_x_units,
        ui_path=reference_ui_path,
        ui_x_units=reference_ui_x_units,
    )

    # Align test grid to reference PMF x-range when not explicitly overridden
    if x_min is None and (references.has_wham or references.has_ui):
        x_min = float(min(
            references.wham_x.min() if references.has_wham else float("inf"),
            references.umbrella_x.min() if references.has_ui else float("inf"),
        ))
    if x_max is None and (references.has_wham or references.has_ui):
        x_max = float(max(
            references.wham_x.max() if references.has_wham else float("-inf"),
            references.umbrella_x.max() if references.has_ui else float("-inf"),
        ))

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

    random_modes_active = (
        window_selection_mode == "random_subset"
        or trajectory_selection_mode == "random_subsample"
    )
    effective_replicates = model.selection_replicates
    if effective_replicates is None:
        effective_replicates = 5 if random_modes_active else 1
    effective_replicates = max(1, int(effective_replicates))
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    cells: list[AblationCellResult] = []
    for i, window_count in enumerate(window_counts):
        for j, trajectory_fraction in enumerate(trajectory_fractions):
            cell = AblationCell(window_count=window_count, trajectory_fraction=trajectory_fraction)
            cell_checkpoint = (
                checkpoint_path / f"{_cell_slug(cell)}.pt"
                if checkpoint_path is not None else None
            )
            if resume and cell_checkpoint is not None and cell_checkpoint.exists():
                loaded_cell = torch.load(
                    cell_checkpoint,
                    map_location=resolved_device,
                    weights_only=False,
                )
                if not isinstance(loaded_cell, AblationCellResult):
                    raise TypeError(f"Checkpoint {cell_checkpoint} does not contain an AblationCellResult.")
                cells.append(loaded_cell)
                print(f"resumed checkpoint: {cell_checkpoint}")
                continue

            replicate_summaries: list[HyperposteriorPredictiveSummary] = []
            replicate_metrics: list[ReferenceComparison] = []
            replicate_diagnostics: list[HMCChainDiagnostics] = []
            replicate_nuts_samples: list[dict[str, torch.Tensor]] = []
            replicate_nuts_grouped_samples: list[dict[str, torch.Tensor]] = []
            replicate_artifacts: list[dict[str, object]] = []
            canonical_summary: HyperposteriorPredictiveSummary | None = None
            canonical_metrics: ReferenceComparison | None = None
            canonical_diagnostics: HMCChainDiagnostics | None = None
            canonical_multi_chain_diagnostics: dict[str, object] | None = None
            canonical_nuts_samples: dict[str, torch.Tensor] | None = None
            canonical_nuts_grouped_samples: dict[str, torch.Tensor] | None = None
            cell_replicates = effective_replicates
            if (
                window_selection_mode == "random_subset"
                and trajectory_selection_mode != "random_subsample"
                and window_count >= len(windows)
            ):
                cell_replicates = 1
            replicate_plan = _replicate_plan(
                effective_replicates=cell_replicates,
                window_selection_mode=window_selection_mode,
                trajectory_selection_mode=trajectory_selection_mode,
            )
            for rep_spec in replicate_plan:
                rep = int(rep_spec["replicate_index"])
                cell_seed = int(random_seed + 1009 * i + 9173 * j + 7919 * window_count + 101 * rep)
                cell_rng = np.random.default_rng(cell_seed)
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
                    window_selection_mode=str(rep_spec["window_selection_mode"]),
                    trajectory_selection_mode=str(rep_spec["trajectory_selection_mode"]),
                    rng=cell_rng,
                )
                bundle = move_workflow_bundle(bundle, device=resolved_device)
                if model.method == "fixed_gp":
                    summary = _fixed_summary(bundle, model)
                    diagnostics = None
                    multi_chain_diagnostics = None
                    nuts_samples = None
                    nuts_grouped_samples = None
                elif model.method == "optimized_gp":
                    summary = _optimized_summary(bundle, model)
                    diagnostics = None
                    multi_chain_diagnostics = None
                    nuts_samples = None
                    nuts_grouped_samples = None
                elif model.method == "nuts":
                    hmc_seed = cell_seed + 1_000_003
                    summary, diagnostics, multi_chain_diagnostics, nuts_samples, nuts_grouped_samples = _nuts_summary(
                        bundle,
                        model,
                        seed=hmc_seed,
                    )
                else:
                    raise ValueError(f"Unsupported ablation method: {model.method}")

                metrics = compare_to_reference_curves(summary, references, alignment=pmf_alignment)
                replicate_summaries.append(summary)
                replicate_metrics.append(metrics)
                replicate_artifacts.append(
                    _replicate_result_payload(
                        replicate_index=rep,
                        is_canonical=bool(rep_spec["is_canonical"]),
                        window_selection_mode=str(rep_spec["window_selection_mode"]),
                        trajectory_selection_mode=str(rep_spec["trajectory_selection_mode"]),
                        random_seed=cell_seed,
                        hmc_seed=hmc_seed if model.method == "nuts" else None,
                        bundle=bundle,
                        summary=summary,
                        metrics=metrics,
                        diagnostics=diagnostics,
                        multi_chain_diagnostics=multi_chain_diagnostics,
                        nuts_samples=nuts_samples,
                        nuts_grouped_samples=nuts_grouped_samples,
                    )
                )
                if bool(rep_spec["is_canonical"]):
                    canonical_summary = summary
                    canonical_metrics = metrics
                    canonical_diagnostics = diagnostics
                    canonical_multi_chain_diagnostics = multi_chain_diagnostics
                    canonical_nuts_samples = nuts_samples
                    canonical_nuts_grouped_samples = nuts_grouped_samples
                if diagnostics is not None:
                    replicate_diagnostics.append(diagnostics)
                if nuts_samples is not None:
                    replicate_nuts_samples.append(nuts_samples)
                if nuts_grouped_samples is not None:
                    replicate_nuts_grouped_samples.append(nuts_grouped_samples)

            summary = _aggregate_predictive_summaries(replicate_summaries)
            metrics = _average_metrics(replicate_metrics)
            diagnostics = (
                _aggregate_chain_diagnostics(replicate_diagnostics)
                if replicate_diagnostics else None
            )
            nuts_samples = replicate_nuts_samples[0] if replicate_nuts_samples else None
            cell_result = AblationCellResult(
                cell=cell,
                replicate_count=cell_replicates,
                metrics=metrics,
                dataset_root=dataset_root_path,
                x_test=summary.x_test,
                predictive_summary=summary,
                canonical_predictive_summary=canonical_summary,
                canonical_metrics=canonical_metrics,
                chain_diagnostics=diagnostics,
                multi_chain_diagnostics=canonical_multi_chain_diagnostics,
                nuts_samples=canonical_nuts_samples if canonical_nuts_samples is not None else nuts_samples,
                nuts_grouped_samples=canonical_nuts_grouped_samples if canonical_nuts_grouped_samples is not None else (replicate_nuts_grouped_samples[0] if replicate_nuts_grouped_samples else None),
                canonical_chain_diagnostics=canonical_diagnostics,
                canonical_multi_chain_diagnostics=canonical_multi_chain_diagnostics,
                canonical_nuts_samples=canonical_nuts_samples,
                canonical_nuts_grouped_samples=canonical_nuts_grouped_samples,
                artifact_payload={
                    "selection_replicates": cell_replicates,
                    "canonical_replicate_index": 0 if random_modes_active else 0,
                    "replicates": replicate_artifacts,
                },
            )
            cells.append(cell_result)
            if cell_checkpoint is not None:
                tmp_checkpoint = cell_checkpoint.with_suffix(cell_checkpoint.suffix + ".tmp")
                torch.save(cell_result, tmp_checkpoint)
                tmp_checkpoint.replace(cell_checkpoint)
                print(f"wrote checkpoint: {cell_checkpoint}")

    return AblationStudyResult(
        model=model,
        window_counts=list(window_counts),
        trajectory_fractions=list(trajectory_fractions),
        cells=cells,
        references=references,
        test_grid_mode=test_grid_mode,
        window_selection_mode=window_selection_mode,
        trajectory_selection_mode=trajectory_selection_mode,
        random_seed=random_seed,
        device=resolved_device,
        pmf_alignment=pmf_alignment,
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


def _shift_curve(y: np.ndarray, alignment: str = "max") -> np.ndarray:
    return y - (np.max(y) if alignment == "max" else np.min(y))


def _cell_slug(cell: AblationCell) -> str:
    frac = f"{cell.trajectory_fraction:.2f}".replace(".", "p")
    return f"w{cell.window_count:02d}_f{frac}"


def _plot_predictive_summary(
    ax,
    *,
    cell: AblationCell,
    summary: HyperposteriorPredictiveSummary,
    metrics: ReferenceComparison,
    references: ReferenceCurves,
    title_suffix: str = "",
    alignment: str = "max",
) -> None:
    x_test = summary.x_test.detach().cpu().numpy().reshape(-1)
    pred_mean = summary.mean.detach().cpu().numpy().reshape(-1)
    pred_std = np.sqrt(np.clip(summary.total_variance.detach().cpu().numpy().reshape(-1), a_min=0.0, a_max=None))
    shifted_mean = _shift_curve(pred_mean, alignment)

    ax.plot(x_test, shifted_mean, lw=2, color="royalblue")
    ax.fill_between(
        x_test,
        shifted_mean - 2.0 * pred_std,
        shifted_mean + 2.0 * pred_std,
        color="royalblue",
        alpha=0.2,
    )
    if references.has_wham:
        ax.plot(references.wham_x, _shift_curve(references.wham_f, alignment), color="crimson", alpha=0.7, lw=1.2)
    if references.has_ui:
        ax.plot(references.umbrella_x, _shift_curve(references.umbrella_f, alignment), color="steelblue", alpha=0.7, lw=1.2)
    rmse_parts = []
    if references.has_wham:
        rmse_parts.append(f"RMSE(WHAM)={metrics.rmse_wham:.2f}")
    if references.has_ui:
        rmse_parts.append(f"RMSE(UI)={metrics.rmse_ui:.2f}")
    stats_str = (", ".join(rmse_parts) + ", " if rmse_parts else "") + f"avg std={metrics.avg_total_std:.2f}"
    ax.set_title(
        f"{cell.window_count} windows, {cell.trajectory_fraction:.2f} traj{title_suffix}\n{stats_str}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.15)


def _plot_predictive_cell(
    ax,
    cell_result: AblationCellResult,
    references: ReferenceCurves,
    alignment: str = "max",
) -> None:
    _plot_predictive_summary(
        ax,
        cell=cell_result.cell,
        summary=cell_result.predictive_summary,
        metrics=cell_result.metrics,
        references=references,
        alignment=alignment,
    )


def _plot_canonical_predictive_cell(
    ax,
    cell_result: AblationCellResult,
    references: ReferenceCurves,
    alignment: str = "max",
) -> None:
    if cell_result.canonical_predictive_summary is None or cell_result.canonical_metrics is None:
        _plot_predictive_cell(ax, cell_result, references, alignment=alignment)
        return
    _plot_predictive_summary(
        ax,
        cell=cell_result.cell,
        summary=cell_result.canonical_predictive_summary,
        metrics=cell_result.canonical_metrics,
        references=references,
        title_suffix=" canonical",
        alignment=alignment,
    )


def _save_predictive_figures(
    result: AblationStudyResult,
    figure_dir: Path,
    *,
    y_lim: tuple[float, float] | None = None,
) -> None:
    predictive_dir = figure_dir / "predictive_cells"
    predictive_dir.mkdir(parents=True, exist_ok=True)

    for cell_result in result.cells:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_predictive_cell(ax, cell_result, result.references, alignment=result.pmf_alignment)
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Shifted free energy [kJ/mol]")
        if y_lim is not None:
            ax.set_ylim(y_lim)
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
        sharex=True,
        sharey=True,
    )
    lookup = {
        (cell.cell.window_count, cell.cell.trajectory_fraction): cell
        for cell in result.cells
    }
    for i, window_count in enumerate(result.window_counts):
        for j, trajectory_fraction in enumerate(result.trajectory_fractions):
            ax = axes[i, j]
            cell_result = lookup[(window_count, trajectory_fraction)]
            _plot_predictive_cell(ax, cell_result, result.references, alignment=result.pmf_alignment)
            if i == n_rows - 1:
                ax.set_xlabel("Position [nm]")
            if j == 0:
                ax.set_ylabel("Shifted free energy [kJ/mol]")
    fig.suptitle(
        f"Ablation Predictive Curves ({result.model.method}, {result.model.kernel})",
        fontsize=14,
    )
    if y_lim is not None:
        axes.flat[0].set_ylim(y_lim)
    fig.tight_layout()
    fig.savefig(figure_dir / "ablation_predictive_grid.png", dpi=200)
    plt.close(fig)

    random_modes_active = (
        result.window_selection_mode == "random_subset"
        or result.trajectory_selection_mode == "random_subsample"
    )
    has_canonical = any(cell.canonical_predictive_summary is not None for cell in result.cells)
    if not random_modes_active or not has_canonical:
        return

    canonical_dir = figure_dir / "predictive_cells_canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    for cell_result in result.cells:
        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_canonical_predictive_cell(ax, cell_result, result.references, alignment=result.pmf_alignment)
        ax.set_xlabel("Position [nm]")
        ax.set_ylabel("Shifted free energy [kJ/mol]")
        if y_lim is not None:
            ax.set_ylim(y_lim)
        fig.tight_layout()
        fig.savefig(canonical_dir / f"{_cell_slug(cell_result.cell)}.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.8 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for i, window_count in enumerate(result.window_counts):
        for j, trajectory_fraction in enumerate(result.trajectory_fractions):
            ax = axes[i, j]
            cell_result = lookup[(window_count, trajectory_fraction)]
            _plot_canonical_predictive_cell(ax, cell_result, result.references, alignment=result.pmf_alignment)
            if i == n_rows - 1:
                ax.set_xlabel("Position [nm]")
            if j == 0:
                ax.set_ylabel("Shifted free energy [kJ/mol]")
    fig.suptitle(
        f"Ablation Predictive Curves ({result.model.method}, {result.model.kernel}, canonical replicate)",
        fontsize=14,
    )
    if y_lim is not None:
        axes.flat[0].set_ylim(y_lim)
    fig.tight_layout()
    fig.savefig(figure_dir / "ablation_predictive_grid_canonical.png", dpi=200)
    plt.close(fig)


def _nuts_config_from_model(model: StudyModelConfig) -> NUTSConfig:
    return NUTSConfig(
        num_samples=model.num_samples,
        warmup_steps=model.warmup_steps,
        num_chains=model.num_chains,
        target_accept_prob=model.target_accept_prob,
        max_tree_depth=model.max_tree_depth,
        jitter=model.jitter,
        objective=model.objective,
        kernel=model.kernel,
        length_model=model.length_model,
        width_model=model.width_model,
        fixed_noise=model.fixed_noise,
    )


def _plot_trace_panel(samples: dict[str, torch.Tensor], output_path: Path) -> None:
    names = list(samples.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(10, 3 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        values = samples[name].detach().cpu().numpy()
        if values.ndim == 1:
            chains = values.reshape(1, -1)
        else:
            chains = values.reshape(values.shape[0], values.shape[1], -1)[:, :, 0]
        for chain_idx, chain_values in enumerate(chains):
            label = f"chain {chain_idx}" if chains.shape[0] > 1 else None
            axes[row, 0].plot(chain_values, lw=1.0, alpha=0.9, label=label)
        axes[row, 0].set_title(f"{name} trace")
        for chain_idx, chain_values in enumerate(chains):
            label = f"chain {chain_idx}" if chains.shape[0] > 1 else None
            axes[row, 1].hist(
                chain_values,
                bins=min(20, max(5, len(chain_values))),
                alpha=0.45,
                label=label,
            )
        axes[row, 1].set_title(f"{name} histogram")
        if chains.shape[0] > 1:
            axes[row, 0].legend(fontsize=8)
            axes[row, 1].legend(fontsize=8)
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
        trace_samples = cell_result.nuts_grouped_samples or cell_result.nuts_samples
        _plot_trace_panel(trace_samples, cell_dir / "nuts_traces.png")
        _plot_corner_panel(
            cell_result.nuts_samples,
            cell_dir / "corner.png",
            config=config,
        )
        if cell_result.multi_chain_diagnostics is not None:
            (cell_dir / "chain_diagnostics.json").write_text(
                json.dumps(cell_result.multi_chain_diagnostics, indent=2) + "\n"
            )


def _parameter_display_summary(
    cell_result: AblationCellResult,
    *,
    model: StudyModelConfig,
) -> tuple[list[str], np.ndarray] | None:
    if cell_result.nuts_samples is None:
        return None
    chain, labels = display_samples_for_diagnostics(
        cell_result.nuts_samples,
        config=_nuts_config_from_model(model),
    )
    values = chain.detach().cpu().numpy()
    return labels, np.median(values, axis=0)


def _save_hyperparameter_heatmaps(
    result: AblationStudyResult,
    figure_dir: Path,
    *,
    param_clims: dict | None = None,
) -> None:
    if result.model.method != "nuts":
        return

    summaries = []
    labels: list[str] | None = None
    for cell_result in result.cells:
        display_summary = _parameter_display_summary(cell_result, model=result.model)
        if display_summary is None:
            continue
        cell_labels, medians = display_summary
        labels = cell_labels
        summaries.append((cell_result.cell.window_count, cell_result.cell.trajectory_fraction, medians))
    if not summaries or labels is None:
        return

    n_params = len(labels)
    n_cols = 2 if n_params <= 4 else 3
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.0 * n_rows), squeeze=False)
    axes_flat = axes.ravel()

    for param_index, label in enumerate(labels):
        ax = axes_flat[param_index]
        grid = np.full((len(result.window_counts), len(result.trajectory_fractions)), np.nan, dtype=float)
        lookup = {
            (window_count, trajectory_fraction): medians[param_index]
            for window_count, trajectory_fraction, medians in summaries
        }
        for i, window_count in enumerate(result.window_counts):
            for j, trajectory_fraction in enumerate(result.trajectory_fractions):
                grid[i, j] = lookup[(window_count, trajectory_fraction)]

        _vmin, _vmax = (param_clims or {}).get(label, (None, None))
        image = ax.imshow(grid, aspect="auto", origin="lower", cmap="magma", vmin=_vmin, vmax=_vmax)
        ax.set_xticks(range(len(result.trajectory_fractions)))
        ax.set_xticklabels([f"{value:.2f}" for value in result.trajectory_fractions])
        ax.set_yticks(range(len(result.window_counts)))
        ax.set_yticklabels([str(value) for value in result.window_counts])
        ax.set_xlabel("Trajectory fraction retained")
        ax.set_ylabel("Windows retained")
        ax.set_title(f"{label} posterior median")
        ax.invert_xaxis()
        fig.colorbar(image, ax=ax, shrink=0.85)

    for ax in axes_flat[n_params:]:
        ax.axis("off")

    fig.suptitle(
        f"Hyperparameter Heatmaps ({result.model.objective}, {result.model.kernel})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "hyperparameter_heatmaps.png", dpi=200)
    plt.close(fig)


def _barrier_height_samples(cell_result: AblationCellResult) -> np.ndarray:
    means = cell_result.predictive_summary.conditional_means.detach().cpu().numpy()
    return means.max(axis=1) - means.min(axis=1)


def _reference_barrier_heights(references: ReferenceCurves) -> tuple[float | None, float | None]:
    wham_barrier = float(np.max(references.wham_f) - np.min(references.wham_f)) if references.has_wham else None
    umbrella_barrier = float(np.max(references.umbrella_f) - np.min(references.umbrella_f)) if references.has_ui else None
    return wham_barrier, umbrella_barrier


def _relative_data_retained(result: AblationStudyResult, cell: AblationCell) -> float:
    max_windows = max(result.window_counts) if result.window_counts else 1
    max_trajectory = max(result.trajectory_fractions) if result.trajectory_fractions else 1.0
    return float((cell.window_count / max_windows) * (cell.trajectory_fraction / max_trajectory))


def _global_barrier_bin_edges(result: AblationStudyResult) -> np.ndarray | None:
    all_barriers = []
    for cell_result in result.cells:
        barriers = _barrier_height_samples(cell_result)
        if barriers.size > 0:
            all_barriers.append(barriers)
    if not all_barriers:
        return None
    combined = np.concatenate(all_barriers)
    lo = float(combined.min())
    hi = float(combined.max())
    if np.isclose(lo, hi):
        pad = max(1e-6, 0.05 * max(abs(lo), 1.0))
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, result.model.barrier_bins + 1)


def _save_barrier_histograms(
    result: AblationStudyResult,
    figure_dir: Path,
) -> None:
    if result.model.method != "nuts":
        return

    bin_edges = _global_barrier_bin_edges(result)
    if bin_edges is None:
        return

    cmap = plt.get_cmap("plasma")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    lookup = {
        (cell.cell.window_count, cell.cell.trajectory_fraction): cell
        for cell in result.cells
    }
    wham_barrier, umbrella_barrier = _reference_barrier_heights(result.references)

    def save_family(
        *,
        outer_values: list[float | int],
        inner_values: list[float | int],
        outer_label: str,
        inner_label: str,
        path: Path,
        pick_cell,
    ) -> None:
        n_panels = len(outer_values)
        n_cols = min(3, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), squeeze=False)
        axes_flat = axes.ravel()

        for ax, outer_value in zip(axes_flat, outer_values):
            for inner_value in inner_values:
                cell_result = pick_cell(outer_value, inner_value)
                barriers = _barrier_height_samples(cell_result)
                retained = _relative_data_retained(result, cell_result.cell)
                color = cmap(norm(retained))
                ax.hist(
                    barriers,
                    bins=bin_edges,
                    histtype="step",
                    alpha=0.95,
                    color=color,
                    edgecolor=color,
                    linewidth=2.0,
                    label=f"{inner_label}={inner_value:.2f}" if isinstance(inner_value, float) else f"{inner_label}={inner_value}",
                )
            if wham_barrier is not None:
                ax.axvline(wham_barrier, color="black", linestyle="--", linewidth=1.6, label="WHAM barrier")
            if umbrella_barrier is not None:
                ax.axvline(umbrella_barrier, color="dimgray", linestyle=":", linewidth=1.8, label="UI barrier")
            ax.set_title(f"{outer_label}={outer_value}")
            ax.set_xlabel("Barrier height from GP mean [kJ/mol]")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.15)
            ax.legend(fontsize=8)

        for ax in axes_flat[n_panels:]:
            ax.axis("off")

        fig.suptitle(
            f"Barrier Height Histograms ({result.model.objective}, {result.model.kernel})",
            fontsize=14,
        )
        fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
        fig.savefig(path, dpi=200)
        plt.close(fig)

    save_family(
        outer_values=result.window_counts,
        inner_values=result.trajectory_fractions,
        outer_label="windows",
        inner_label="traj",
        path=figure_dir / "barrier_histograms_by_windows.png",
        pick_cell=lambda window_count, trajectory_fraction: lookup[(window_count, trajectory_fraction)],
    )
    save_family(
        outer_values=result.trajectory_fractions,
        inner_values=result.window_counts,
        outer_label="traj",
        inner_label="windows",
        path=figure_dir / "barrier_histograms_by_trajectory.png",
        pick_cell=lambda trajectory_fraction, window_count: lookup[(window_count, trajectory_fraction)],
    )


def _save_result_artifacts(
    result: AblationStudyResult,
    figure_dir: Path,
) -> None:
    artifacts_dir = figure_dir / "artifacts"
    cell_dir = artifacts_dir / "cells"
    cell_dir.mkdir(parents=True, exist_ok=True)

    if result.references.has_wham or result.references.has_ui:
        save_kwargs: dict[str, object] = {}
        if result.references.has_wham:
            save_kwargs.update(wham_x=result.references.wham_x, wham_f=result.references.wham_f)
            if result.references.wham_e is not None:
                save_kwargs["wham_e"] = result.references.wham_e
        if result.references.has_ui:
            save_kwargs.update(umbrella_x=result.references.umbrella_x, umbrella_f=result.references.umbrella_f)
            if result.references.umbrella_e is not None:
                save_kwargs["umbrella_e"] = result.references.umbrella_e
        np.savez(artifacts_dir / "references.npz", **save_kwargs)

    wham_b, ui_b = _reference_barrier_heights(result.references)
    manifest = {
        "model": asdict(result.model),
        "window_counts": result.window_counts,
        "trajectory_fractions": result.trajectory_fractions,
        "test_grid_mode": result.test_grid_mode,
        "window_selection_mode": result.window_selection_mode,
        "trajectory_selection_mode": result.trajectory_selection_mode,
        "random_seed": result.random_seed,
        "device": result.device,
        "dataset_root": str(result.cells[0].dataset_root) if result.cells else "",
        "reference_barriers": {
            k: v for k, v in [("wham", wham_b), ("ui", ui_b)] if v is not None
        } or None,
        "cells": [],
    }

    for cell_result in result.cells:
        slug = _cell_slug(cell_result.cell)
        payload = {
            "cell": {
                "window_count": cell_result.cell.window_count,
                "trajectory_fraction": cell_result.cell.trajectory_fraction,
            },
            "model": asdict(result.model),
            "metrics": asdict(cell_result.metrics),
            "canonical_metrics": None if cell_result.canonical_metrics is None else asdict(cell_result.canonical_metrics),
            "chain_diagnostics": None if cell_result.chain_diagnostics is None else asdict(cell_result.chain_diagnostics),
            "multi_chain_diagnostics": cell_result.multi_chain_diagnostics,
            "canonical_chain_diagnostics": None if cell_result.canonical_chain_diagnostics is None else asdict(cell_result.canonical_chain_diagnostics),
            "canonical_multi_chain_diagnostics": cell_result.canonical_multi_chain_diagnostics,
            "bundle": cell_result.artifact_payload,
            "predictive_summary": _predictive_summary_payload(cell_result.predictive_summary),
            "canonical_predictive_summary": None
            if cell_result.canonical_predictive_summary is None
            else _predictive_summary_payload(cell_result.canonical_predictive_summary),
            "nuts_samples": _cpu_clone(cell_result.nuts_samples),
            "canonical_nuts_samples": _cpu_clone(cell_result.canonical_nuts_samples),
        }
        torch.save(payload, cell_dir / f"{slug}.pt")
        manifest["cells"].append(
            {
                "slug": slug,
                "window_count": cell_result.cell.window_count,
                "trajectory_fraction": cell_result.cell.trajectory_fraction,
                "replicate_count": cell_result.replicate_count,
                "has_canonical_replicate": cell_result.canonical_predictive_summary is not None,
                "window_selection_mode": result.window_selection_mode,
                "trajectory_selection_mode": result.trajectory_selection_mode,
                "artifact_path": str((cell_dir / f"{slug}.pt").relative_to(figure_dir)),
            }
        )

    (artifacts_dir / "study_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def compute_predictive_y_lim(result: AblationStudyResult) -> tuple[float, float]:
    """Y-axis range covering all cells' shifted mean ± 2σ and any reference curves."""
    ymin, ymax = float("inf"), float("-inf")
    for cell_result in result.cells:
        mean = cell_result.predictive_summary.mean.detach().cpu().numpy().reshape(-1)
        std = np.sqrt(np.clip(cell_result.predictive_summary.total_variance.detach().cpu().numpy().reshape(-1), 0.0, None))
        shifted = _shift_curve(mean, result.pmf_alignment)
        ymin = min(ymin, float((shifted - 2.0 * std).min()))
        ymax = max(ymax, float((shifted + 2.0 * std).max()))
    for arr, has in [(result.references.wham_f, result.references.has_wham),
                     (result.references.umbrella_f, result.references.has_ui)]:
        if has:
            sv = _shift_curve(arr, result.pmf_alignment)
            ymin = min(ymin, float(sv.min()))
            ymax = max(ymax, float(sv.max()))
    pad = 0.05 * max(ymax - ymin, 1.0)
    return ymin - pad, ymax + pad


def compute_metric_clims(result: AblationStudyResult) -> dict[str, tuple[float, float]]:
    """Color limits for each metric heatmap from this result."""
    clims: dict[str, tuple[float, float]] = {}
    for name in ["rmse_wham", "rmse_ui", "avg_total_std"]:
        finite = _metric_grid(result, name)
        finite = finite[np.isfinite(finite)]
        if finite.size > 0:
            clims[name] = (float(finite.min()), float(finite.max()))
    return clims


def compute_param_clims(result: AblationStudyResult) -> dict[str, tuple[float, float]]:
    """Color limits for each hyperparameter heatmap from this result."""
    clims: dict[str, tuple[float, float]] = {}
    if result.model.method != "nuts":
        return clims
    for cell_result in result.cells:
        out = _parameter_display_summary(cell_result, model=result.model)
        if out is None:
            continue
        labels, medians = out
        for label, val in zip(labels, medians):
            v = float(val)
            clims[label] = (min(clims[label][0], v), max(clims[label][1], v)) if label in clims else (v, v)
    return clims


def save_ablation_summary(
    result: AblationStudyResult,
    figure_dir: Path,
    *,
    predictive_y_lim: tuple[float, float] | None = None,
    metric_clims: dict[str, tuple[float, float]] | None = None,
    param_clims: dict[str, tuple[float, float]] | None = None,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    _save_result_artifacts(result, figure_dir)
    _save_predictive_figures(result, figure_dir, y_lim=predictive_y_lim)
    _save_nuts_diagnostics(result, figure_dir)
    _save_hyperparameter_heatmaps(result, figure_dir, param_clims=param_clims)
    _save_barrier_histograms(result, figure_dir)

    metric_specs = [
        ("rmse_wham", "RMSE vs WHAM"),
        ("rmse_ui", "RMSE vs UI"),
        ("avg_total_std", "Average total std (common grid)"),
    ]
    if not result.references.has_wham:
        metric_specs = [(k, v) for k, v in metric_specs if k != "rmse_wham"]
    if not result.references.has_ui:
        metric_specs = [(k, v) for k, v in metric_specs if k != "rmse_ui"]

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(5.5 * len(metric_specs), 4.8))
    if len(metric_specs) == 1:
        axes = [axes]
    for ax, (metric_name, title) in zip(axes, metric_specs):
        grid = _metric_grid(result, metric_name)
        _vmin, _vmax = (metric_clims or {}).get(metric_name, (None, None))
        image = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis", vmin=_vmin, vmax=_vmax)
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
        f"Ablation Summary ({result.model.method}, {result.model.kernel}, {result.model.objective})",
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
                "replicate_count",
                "rmse_wham",
                "rmse_ui",
                "avg_total_std",
                "avg_within_std",
                "avg_between_std",
                "avg_total_variance",
                "avg_within_variance",
                "avg_between_variance",
                "barrier_height_mean",
                "barrier_height_std",
                "hmc_step_size",
                "hmc_mean_accept_prob",
                "hmc_accept_count",
                "hmc_divergence_count",
                "hmc_mean_sample_std",
                "hmc_max_sample_std",
                "hmc_min_sample_std",
                "hmc_poor_acceptance",
                "hmc_looks_stuck",
                "mcmc_max_r_hat",
                "mcmc_min_n_eff",
                "mcmc_divergence_total",
            ]
        )
        for cell in result.cells:
            diagnostics = cell.chain_diagnostics
            multi_chain_diagnostics = cell.multi_chain_diagnostics or {}
            multi_chain_summary = multi_chain_diagnostics.get("summary", {})
            barrier_heights = _barrier_height_samples(cell)
            writer.writerow(
                [
                    cell.cell.window_count,
                    cell.cell.trajectory_fraction,
                    cell.replicate_count,
                    cell.metrics.rmse_wham,
                    cell.metrics.rmse_ui,
                    cell.metrics.avg_total_std,
                    cell.metrics.avg_within_std,
                    cell.metrics.avg_between_std,
                    cell.metrics.avg_total_variance,
                    cell.metrics.avg_within_variance,
                    cell.metrics.avg_between_variance,
                    float(np.mean(barrier_heights)),
                    float(np.std(barrier_heights)),
                    "" if diagnostics is None else diagnostics.step_size,
                    "" if diagnostics is None else diagnostics.mean_accept_prob,
                    "" if diagnostics is None else diagnostics.accept_count,
                    "" if diagnostics is None else diagnostics.divergence_count,
                    "" if diagnostics is None else diagnostics.mean_sample_std,
                    "" if diagnostics is None else diagnostics.max_sample_std,
                    "" if diagnostics is None else diagnostics.min_sample_std,
                    "" if diagnostics is None else diagnostics.poor_acceptance,
                    "" if diagnostics is None else diagnostics.looks_stuck,
                    multi_chain_summary.get("max_r_hat", ""),
                    multi_chain_summary.get("min_n_eff", ""),
                    multi_chain_summary.get("divergence_total", ""),
                ]
            )

        lines = [
            f"method: {result.model.method}",
            f"kernel: {result.model.kernel}",
            f"device: {result.device}",
            f"objective: {result.model.objective}",
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
            f"window_selection_mode: {result.window_selection_mode}",
            f"trajectory_selection_mode: {result.trajectory_selection_mode}",
            f"random_seed: {result.random_seed}",
            f"default_selection_replicates: {result.cells[0].replicate_count if result.cells else 0}",
            "uncertainty_summary: average total standard deviation on the prediction grid",
            f"predictive_cell_dir: {figure_dir / 'predictive_cells'}",
            f"artifacts_dir: {figure_dir / 'artifacts'}",
        ]
    )
    if result.model.method == "nuts":
        stuck_count = sum(
            1 for cell in result.cells
            if cell.chain_diagnostics is not None and cell.chain_diagnostics.looks_stuck
        )
        lines.append(f"warmup_steps: {result.model.warmup_steps}")
        lines.append(f"num_samples: {result.model.num_samples}")
        lines.append(f"num_chains: {result.model.num_chains}")
        lines.append(f"target_accept_prob: {result.model.target_accept_prob}")
        lines.append(f"max_tree_depth: {result.model.max_tree_depth}")
        lines.append(f"stuck_cells: {stuck_count}")
        lines.append(f"nuts_diagnostics_dir: {figure_dir / 'nuts_diagnostics'}")
        lines.append(f"hyperparameter_heatmaps: {figure_dir / 'hyperparameter_heatmaps.png'}")
        lines.append(f"barrier_histograms_by_windows: {figure_dir / 'barrier_histograms_by_windows.png'}")
        lines.append(f"barrier_histograms_by_trajectory: {figure_dir / 'barrier_histograms_by_trajectory.png'}")
        r_hats = [
            cell.multi_chain_diagnostics.get("summary", {}).get("max_r_hat")
            for cell in result.cells
            if cell.multi_chain_diagnostics is not None
        ]
        n_effs = [
            cell.multi_chain_diagnostics.get("summary", {}).get("min_n_eff")
            for cell in result.cells
            if cell.multi_chain_diagnostics is not None
        ]
        r_hats = [float(v) for v in r_hats if v is not None]
        n_effs = [float(v) for v in n_effs if v is not None]
        if r_hats:
            lines.append(f"worst_cell_max_r_hat: {max(r_hats)}")
        if n_effs:
            lines.append(f"worst_cell_min_n_eff: {min(n_effs)}")
    if result.model.method == "optimized_gp":
        lines.append(f"opt_steps: {result.model.opt_steps}")
        lines.append(f"opt_restarts: {result.model.opt_restarts}")
        lines.append(f"opt_learning_rate: {result.model.opt_learning_rate}")
    (figure_dir / "run_summary.txt").write_text("\n".join(lines) + "\n")
