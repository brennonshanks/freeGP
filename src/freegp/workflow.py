"""High-level helpers that mirror the notebook workflow without notebook state."""

from __future__ import annotations

from dataclasses import dataclass
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
    process_umbrella_windows,
)


@dataclass(frozen=True)
class WorkflowBundle:
    processed: ProcessedUmbrellaData
    observations: JointObservations
    x_test: torch.Tensor
    references: ReferenceCurves
    dataset_root: Path


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
    resolved_dataset_root = resolve_dataset_root(dataset_root)
    windows = load_umbrella_windows(resolved_dataset_root)
    processed = process_umbrella_windows(
        windows,
        n_equilibration=n_equilibration,
        num_bins=num_bins,
    )
    observations = build_joint_observations(processed)
    x_test = build_test_grid(
        processed,
        num_points=num_test_points,
        x_min=x_min,
        x_max=x_max,
        source=test_grid_source,
    )
    references = load_reference_curves(
        project_root,
        wham_path=reference_wham_path,
        wham_x_units=reference_wham_x_units,
        ui_path=reference_ui_path,
        ui_x_units=reference_ui_x_units,
    )
    return WorkflowBundle(
        processed=processed,
        observations=observations,
        x_test=x_test,
        references=references,
        dataset_root=resolved_dataset_root,
    )
