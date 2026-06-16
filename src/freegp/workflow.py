"""High-level helpers that mirror the notebook workflow without notebook state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .data import (
    ReferenceCurves,
    load_reference_curves,
    load_umbrella_windows,
    resolve_dataset_root,
)
from .preprocess import (
    JointObservations,
    ProcessedUmbrellaData,
    build_joint_observations,
    build_test_grid,
    move_joint_observations,
    move_processed_umbrella_data,
    process_umbrella_windows,
)


def _resolve_dataset_root_path(dataset_root: str | None) -> Path:
    """Resolve dataset root the same way the ablation runner does: plain Path resolution,
    then let load_umbrella_windows handle Denis/Katka auto-detection internally."""
    if dataset_root is not None:
        return Path(dataset_root).expanduser().resolve()
    # Fall back to env var via resolve_dataset_root when no explicit path is given.
    return resolve_dataset_root(None)


@dataclass(frozen=True)
class WorkflowBundle:
    processed: ProcessedUmbrellaData
    observations: JointObservations
    x_test: torch.Tensor
    references: ReferenceCurves
    dataset_root: Path


def move_workflow_bundle(
    bundle: WorkflowBundle,
    *,
    device: torch.device | str,
) -> WorkflowBundle:
    """Return a copy of ``bundle`` with tensor-valued fields moved onto ``device``."""
    return replace(
        bundle,
        processed=move_processed_umbrella_data(bundle.processed, device=device),
        observations=move_joint_observations(bundle.observations, device=device),
        x_test=bundle.x_test.to(device=device),
    )


def prepare_gprhd_hmc_inputs(
    *,
    dataset_root: str | None = None,
    project_root: str | None = None,
    reference_wham_path: str | None = None,
    reference_wham_x_units: str = "nm",
    reference_ui_path: str | None = None,
    reference_ui_x_units: str = "nm",
    n_equilibration: int = 40_000,
    num_bins: int = 20,
    num_test_points: int = 400,
    x_min: float | None = None,
    x_max: float | None = None,
    test_grid_source: str = "umbrella_centers",
) -> WorkflowBundle:
    """Load data, preprocess it, and build the objects needed for HMC-NUTS."""
    # Resolve the path directly (matching the ablation runner pattern) so that
    # load_umbrella_windows handles Denis/Katka auto-detection in one place.
    dataset_root_path = _resolve_dataset_root_path(dataset_root)
    windows = load_umbrella_windows(dataset_root_path)
    processed = process_umbrella_windows(
        windows,
        n_equilibration=n_equilibration,
        num_bins=num_bins,
    )
    observations = build_joint_observations(processed)
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

    x_test = build_test_grid(
        processed,
        num_points=num_test_points,
        x_min=x_min,
        x_max=x_max,
        source=test_grid_source,
    )
    return WorkflowBundle(
        processed=processed,
        observations=observations,
        x_test=x_test,
        references=references,
        dataset_root=dataset_root_path,
    )
