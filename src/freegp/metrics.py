"""Reference-comparison and uncertainty summary metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import ReferenceCurves
from .posterior import HyperposteriorPredictiveSummary


@dataclass(frozen=True)
class ReferenceComparison:
    rmse_wham: float
    rmse_ui: float
    avg_total_std: float
    avg_within_std: float
    avg_between_std: float
    avg_total_variance: float
    avg_within_variance: float
    avg_between_variance: float


def _shift_curve(y: np.ndarray) -> np.ndarray:
    return y - np.max(y)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compare_to_reference_curves(
    summary: HyperposteriorPredictiveSummary,
    references: ReferenceCurves,
) -> ReferenceComparison:
    x_test = summary.x_test.detach().cpu().numpy().reshape(-1)
    pred = _shift_curve(summary.mean.detach().cpu().numpy().reshape(-1))

    wham_ref = np.interp(x_test, references.wham_x, _shift_curve(references.wham_f))
    ui_ref = np.interp(x_test, references.umbrella_x, _shift_curve(references.umbrella_f))

    total_var = summary.total_variance.detach().cpu().numpy().reshape(-1)
    within_var = summary.within_variance.detach().cpu().numpy().reshape(-1)
    between_var = summary.between_variance.detach().cpu().numpy().reshape(-1)

    return ReferenceComparison(
        rmse_wham=_rmse(wham_ref, pred),
        rmse_ui=_rmse(ui_ref, pred),
        avg_total_std=float(np.sqrt(np.clip(total_var, a_min=0.0, a_max=None)).mean()),
        avg_within_std=float(np.sqrt(np.clip(within_var, a_min=0.0, a_max=None)).mean()),
        avg_between_std=float(np.sqrt(np.clip(between_var, a_min=0.0, a_max=None)).mean()),
        avg_total_variance=float(total_var.mean()),
        avg_within_variance=float(within_var.mean()),
        avg_between_variance=float(between_var.mean()),
    )
