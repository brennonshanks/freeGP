#!/usr/bin/env python3
"""
Replot ablation-grid results from saved artifacts, with axis scales computed
only from a chosen subset of objectives (default: lml and loo, excluding fixed).

Plots are written to a sub-folder named 'replot' inside each objective directory,
e.g. results/NMSM-ablation-10x10-replicates/lml/replot/.

Usage
-----
    # Replot lml and loo with scales from lml+loo (default behaviour)
    python replot_ablation.py --results-dir results/NMSM-ablation-10x10-replicates

    # Only set axis limits from lml+loo, but also replot fixed
    python replot_ablation.py --results-dir results/NMSM-ablation-10x10-replicates \
        --objectives lml loo fixed --scale-from lml loo

    # Use a different sub-folder name for the output
    python replot_ablation.py --results-dir results/NMSM-ablation-10x10-replicates \
        --output-subdir rescaled

Each objective sub-folder must contain:
    artifacts/study_manifest.json
    artifacts/references.npz          (optional, if reference curves were provided)
    artifacts/cells/<slug>.pt         (one per ablation cell)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make sure the freeGP package is importable when run directly from the repo
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from freegp.data import ReferenceCurves
from freegp.hmc import HMCChainDiagnostics, NUTSConfig
from freegp.metrics import ReferenceComparison
from freegp.posterior import HyperposteriorPredictiveSummary
from freegp.studies.ablation import (
    AblationCell,
    AblationCellResult,
    AblationStudyResult,
    StudyModelConfig,
    _metric_grid,
    _save_barrier_histograms,
    _save_hyperparameter_heatmaps,
    _save_predictive_figures,
    compute_metric_clims,
    compute_param_clims,
    compute_predictive_y_lim,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers to reconstruct dataclass objects from the saved .pt payloads
# ---------------------------------------------------------------------------

def _load_references(artifacts_dir: Path) -> ReferenceCurves:
    """Load reference curves from references.npz, or return an empty object."""
    ref_path = artifacts_dir / "references.npz"
    if not ref_path.exists():
        return ReferenceCurves()
    data = np.load(ref_path)
    return ReferenceCurves(
        wham_x=data["wham_x"] if "wham_x" in data else None,
        wham_f=data["wham_f"] if "wham_f" in data else None,
        wham_e=data["wham_e"] if "wham_e" in data else None,
        umbrella_x=data["umbrella_x"] if "umbrella_x" in data else None,
        umbrella_f=data["umbrella_f"] if "umbrella_f" in data else None,
        umbrella_e=data["umbrella_e"] if "umbrella_e" in data else None,
    )


def _load_predictive_summary(payload: dict) -> HyperposteriorPredictiveSummary:
    """Reconstruct HyperposteriorPredictiveSummary from a saved payload dict.

    The .pt files store only the variance diagonals (not full covariance matrices).
    We reconstitute diagonal covariance matrices so all existing properties work.
    conditional_covariances is not stored; we use zeros (it is not used by
    any of the plotting functions).
    """
    x_test = payload["x_test"]
    mean = payload["mean"]
    total_var = payload["total_variance"]
    within_var = payload["within_variance"]
    between_var = payload["between_variance"]
    conditional_means = payload["conditional_means"]
    selected_indices = payload["selected_indices"]

    n = len(total_var)
    k = conditional_means.shape[0]

    return HyperposteriorPredictiveSummary(
        x_test=x_test,
        mean=mean,
        total_cov=torch.diag(total_var),
        within_cov=torch.diag(within_var),
        between_cov=torch.diag(between_var),
        conditional_means=conditional_means,
        conditional_covariances=torch.zeros(k, n, n, dtype=total_var.dtype),
        selected_indices=selected_indices,
    )


def _load_metrics(payload: dict | None) -> ReferenceComparison | None:
    if payload is None:
        return None
    return ReferenceComparison(
        rmse_wham=payload.get("rmse_wham", float("nan")),
        rmse_ui=payload.get("rmse_ui", float("nan")),
        avg_total_std=payload.get("avg_total_std", float("nan")),
        avg_within_std=payload.get("avg_within_std", float("nan")),
        avg_between_std=payload.get("avg_between_std", float("nan")),
        avg_total_variance=payload.get("avg_total_variance", float("nan")),
        avg_within_variance=payload.get("avg_within_variance", float("nan")),
        avg_between_variance=payload.get("avg_between_variance", float("nan")),
    )


def _load_chain_diagnostics(payload: dict | None) -> HMCChainDiagnostics | None:
    if payload is None:
        return None
    return HMCChainDiagnostics(
        step_size=payload["step_size"],
        mean_accept_prob=payload["mean_accept_prob"],
        accept_count=payload["accept_count"],
        divergence_count=payload["divergence_count"],
        sample_std_by_name=payload["sample_std_by_name"],
        mean_sample_std=payload["mean_sample_std"],
        max_sample_std=payload["max_sample_std"],
        min_sample_std=payload["min_sample_std"],
        poor_acceptance=payload["poor_acceptance"],
        looks_stuck=payload["looks_stuck"],
    )


def _load_cell(pt_path: Path) -> tuple[AblationCellResult, dict]:
    """Load a single .pt artifact file and return an AblationCellResult."""
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)

    cell_info = raw["cell"]
    cell = AblationCell(
        window_count=cell_info["window_count"],
        trajectory_fraction=cell_info["trajectory_fraction"],
    )

    metrics = _load_metrics(raw.get("metrics"))
    assert metrics is not None, f"Missing metrics in {pt_path}"

    predictive_summary = _load_predictive_summary(raw["predictive_summary"])

    canonical_summary_raw = raw.get("canonical_predictive_summary")
    canonical_summary = (
        _load_predictive_summary(canonical_summary_raw)
        if canonical_summary_raw is not None
        else None
    )

    canonical_metrics = _load_metrics(raw.get("canonical_metrics"))
    chain_diagnostics = _load_chain_diagnostics(raw.get("chain_diagnostics"))
    canonical_chain_diagnostics = _load_chain_diagnostics(
        raw.get("canonical_chain_diagnostics")
    )

    nuts_samples = raw.get("nuts_samples")
    canonical_nuts_samples = raw.get("canonical_nuts_samples")

    # replicate_count is stored in the manifest, not the cell file; default 1
    replicate_count = raw.get("replicate_count", 1)

    cell_result = AblationCellResult(
        cell=cell,
        replicate_count=replicate_count,
        metrics=metrics,
        dataset_root=Path(raw.get("dataset_root", ".")),
        x_test=predictive_summary.x_test,
        predictive_summary=predictive_summary,
        canonical_predictive_summary=canonical_summary,
        canonical_metrics=canonical_metrics,
        chain_diagnostics=chain_diagnostics,
        nuts_samples=nuts_samples,
        canonical_chain_diagnostics=canonical_chain_diagnostics,
        canonical_nuts_samples=canonical_nuts_samples,
    )
    return cell_result, raw.get("model", {})


def load_ablation_result(objective_dir: Path) -> AblationStudyResult:
    """Reconstruct an AblationStudyResult from a saved objective directory."""
    artifacts_dir = objective_dir / "artifacts"
    manifest_path = artifacts_dir / "study_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"study_manifest.json not found in {artifacts_dir}")

    manifest = json.loads(manifest_path.read_text())

    # Reconstruct StudyModelConfig from manifest
    model_dict = manifest["model"]
    model = StudyModelConfig(**{
        k: v for k, v in model_dict.items()
        if k in StudyModelConfig.__dataclass_fields__
    })

    window_counts: list[int] = manifest["window_counts"]
    trajectory_fractions: list[float] = manifest["trajectory_fractions"]
    test_grid_mode: str = manifest.get("test_grid_mode", "full_dataset")
    window_selection_mode: str = manifest.get("window_selection_mode", "evenly_spaced")
    trajectory_selection_mode: str = manifest.get("trajectory_selection_mode", "contiguous")
    random_seed: int = manifest.get("random_seed", 0)
    pmf_alignment: str = manifest.get("pmf_alignment", "max")

    references = _load_references(artifacts_dir)

    cells: list[AblationCellResult] = []
    cells_dir = artifacts_dir / "cells"

    # Build a lookup from manifest to know replicate_count per cell
    manifest_cell_lookup = {
        (entry["window_count"], entry["trajectory_fraction"]): entry
        for entry in manifest.get("cells", [])
    }

    for entry in manifest.get("cells", []):
        pt_path = objective_dir / entry["artifact_path"]
        if not pt_path.exists():
            print(f"  WARNING: artifact not found: {pt_path}", file=sys.stderr)
            continue
        cell_result, _ = _load_cell(pt_path)

        # Patch replicate_count from the manifest (it's not in the .pt file)
        replicate_count = entry.get("replicate_count", 1)
        cell_result = AblationCellResult(
            cell=cell_result.cell,
            replicate_count=replicate_count,
            metrics=cell_result.metrics,
            dataset_root=cell_result.dataset_root,
            x_test=cell_result.x_test,
            predictive_summary=cell_result.predictive_summary,
            canonical_predictive_summary=cell_result.canonical_predictive_summary,
            canonical_metrics=cell_result.canonical_metrics,
            chain_diagnostics=cell_result.chain_diagnostics,
            nuts_samples=cell_result.nuts_samples,
            canonical_chain_diagnostics=cell_result.canonical_chain_diagnostics,
            canonical_nuts_samples=cell_result.canonical_nuts_samples,
        )
        cells.append(cell_result)

    # Sort cells in the same order as window_counts × trajectory_fractions
    wc_order = {wc: i for i, wc in enumerate(window_counts)}
    tf_order = {tf: j for j, tf in enumerate(trajectory_fractions)}
    cells.sort(key=lambda c: (wc_order.get(c.cell.window_count, 9999),
                               tf_order.get(c.cell.trajectory_fraction, 9999)))

    return AblationStudyResult(
        model=model,
        window_counts=window_counts,
        trajectory_fractions=trajectory_fractions,
        cells=cells,
        references=references,
        test_grid_mode=test_grid_mode,
        window_selection_mode=window_selection_mode,
        trajectory_selection_mode=trajectory_selection_mode,
        random_seed=random_seed,
        pmf_alignment=pmf_alignment,
    )


def _save_figures_only(
    result: AblationStudyResult,
    figure_dir: Path,
    *,
    predictive_y_lim: tuple[float, float] | None = None,
    metric_clims: dict[str, tuple[float, float]] | None = None,
    param_clims: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Save only the figures — no .pt artifacts, no CSV, no NUTS trace plots."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Predictive curves grid + individual cell plots
    _save_predictive_figures(result, figure_dir, y_lim=predictive_y_lim)

    # Hyperparameter posterior heatmaps (NUTS only)
    _save_hyperparameter_heatmaps(result, figure_dir, param_clims=param_clims)

    # Barrier-height histograms (NUTS only)
    _save_barrier_histograms(result, figure_dir)

    # Metric summary heatmap (ablation_summary.png) — replicated from save_ablation_summary
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


def _compute_scales(results: list[AblationStudyResult]) -> dict:
    """Compute unified axis/colour scales across a list of AblationStudyResult objects."""
    y_lims = [compute_predictive_y_lim(r) for r in results]
    y_lim = (min(lo for lo, _ in y_lims), max(hi for _, hi in y_lims))

    all_mc = [compute_metric_clims(r) for r in results]
    metric_clims: dict = {}
    for name in {k for d in all_mc for k in d}:
        vals = [d[name] for d in all_mc if name in d]
        metric_clims[name] = (min(lo for lo, _ in vals), max(hi for _, hi in vals))

    all_pc = [compute_param_clims(r) for r in results]
    param_clims: dict = {}
    for name in {k for d in all_pc for k in d}:
        vals = [d[name] for d in all_pc if name in d]
        param_clims[name] = (min(lo for lo, _ in vals), max(hi for _, hi in vals))

    return {"predictive_y_lim": y_lim, "metric_clims": metric_clims, "param_clims": param_clims}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replot freeGP ablation-grid results from saved artifacts, "
            "with axis limits computed only from a chosen subset of objectives."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=str,
        help=(
            "Root directory that contains the objective sub-folders "
            "(e.g. results/NMSM-ablation-10x10-replicates)."
        ),
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=None,
        help=(
            "Objective sub-folders to replot. Defaults to all sub-folders that "
            "contain an artifacts/study_manifest.json."
        ),
    )
    parser.add_argument(
        "--scale-from",
        nargs="+",
        default=None,
        metavar="OBJECTIVE",
        help=(
            "Subset of objectives used to compute axis/colour limits. "
            "Defaults to all objectives listed under --objectives excluding 'fixed'. "
            "Pass 'all' to include every objective when computing scales."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="replot",
        metavar="NAME",
        help=(
            "Name of the sub-folder created inside each objective directory to "
            "store the replots (default: 'replot'). "
            "E.g. lml/replot/, loo/replot/, fixed/replot/."
        ),
    )
    args = parser.parse_args()

    root = Path(args.results_dir).expanduser().resolve()
    if not root.exists():
        parser.error(f"results-dir not found: {root}")

    # Discover available objectives
    available = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / "artifacts" / "study_manifest.json").exists()
    )
    if not available:
        parser.error(f"No objective sub-folders with artifacts found in {root}")

    objectives = args.objectives or available
    # Validate
    for obj in objectives:
        if obj not in available:
            parser.error(
                f"Objective '{obj}' not found under {root}. "
                f"Available: {available}"
            )

    # Determine which objectives contribute to scale computation
    if args.scale_from is None:
        scale_from = [o for o in objectives if o != "fixed"]
        if not scale_from:
            scale_from = objectives  # fallback: all if only 'fixed' present
    elif args.scale_from == ["all"]:
        scale_from = objectives
    else:
        scale_from = args.scale_from

    print(f"Results root : {root}")
    print(f"Objectives   : {objectives}")
    print(f"Scale from   : {scale_from}")

    # ------------------------------------------------------------------
    # Load all requested results
    # ------------------------------------------------------------------
    results: dict[str, AblationStudyResult] = {}
    for obj in objectives:
        obj_dir = root / obj
        print(f"  Loading '{obj}' from {obj_dir} …", flush=True)
        results[obj] = load_ablation_result(obj_dir)
        print(f"    → {len(results[obj].cells)} cells loaded.")

    # ------------------------------------------------------------------
    # Compute global scales from the scale_from subset
    # ------------------------------------------------------------------
    scale_results = [results[o] for o in scale_from if o in results]
    if not scale_results:
        parser.error(f"None of scale-from objectives {scale_from} were loaded.")

    print(f"\nComputing scales from: {[o for o in scale_from if o in results]}")
    scale_kwargs = _compute_scales(scale_results)
    y_lim = scale_kwargs["predictive_y_lim"]
    print(f"  predictive y-axis : [{y_lim[0]:.3f}, {y_lim[1]:.3f}] kJ/mol")
    for name, (lo, hi) in scale_kwargs["metric_clims"].items():
        print(f"  metric clim '{name}': [{lo:.4g}, {hi:.4g}]")
    for name, (lo, hi) in scale_kwargs.get("param_clims", {}).items():
        print(f"  param clim  '{name}': [{lo:.4g}, {hi:.4g}]")

    # ------------------------------------------------------------------
    # Replot each objective into <obj_dir>/replot/ (or --output-subdir)
    # ------------------------------------------------------------------
    for obj in objectives:
        figure_dir = root / obj / args.output_subdir
        figure_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving plots for '{obj}' → {figure_dir}")
        _save_figures_only(results[obj], figure_dir, **scale_kwargs)
        print(f"  Done.")

    print("\nAll plots saved successfully.")


if __name__ == "__main__":
    main()
