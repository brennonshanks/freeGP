"""Metadynamics preprocessing for the joint histogram + derivative GP model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import ReferenceCurves
from .preprocess import JointObservations, bayes_autocorrelation_time, beta


@dataclass(frozen=True)
class MetadynamicsTrajectory:
    time: np.ndarray
    meta_cv: np.ndarray
    cv: np.ndarray
    bias: np.ndarray | None = None
    lower_boundary_potential: np.ndarray | None = None
    upper_boundary_potential: np.ndarray | None = None


@dataclass(frozen=True)
class MetadynamicsReference:
    x: np.ndarray
    pmf: np.ndarray
    derivative: np.ndarray | None = None


@dataclass(frozen=True)
class MetadynamicsProcessed:
    window_centers: torch.Tensor
    histogram_counts: list[torch.Tensor]
    histogram_probs: list[torch.Tensor]
    bin_centers_list: list[torch.Tensor]
    n_samples_per_window: torch.Tensor
    autocorr_times: torch.Tensor
    derivative_bin_centers: torch.Tensor
    derivative_means: torch.Tensor
    derivative_variances: torch.Tensor
    derivative_sample_counts: torch.Tensor
    derivative_autocorr_times: torch.Tensor


def _load_numeric_table(path: str | Path) -> np.ndarray:
    resolved = Path(path).expanduser().resolve()
    data = np.loadtxt(resolved, comments=("#", "@", ";"))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_colvar(
    path: str | Path,
    *,
    time_col: int = 0,
    meta_cv_col: int = 1,
    cv_col: int = 2,
    bias_col: int | None = 3,
    lower_boundary_col: int | None = 4,
    upper_boundary_col: int | None = 5,
) -> MetadynamicsTrajectory:
    """Load a PLUMED-style COLVAR file.

    The legacy notebooks used columns ``time, MetaCV, CV, bias, lower, upper``.
    The column indices are configurable so other COLVAR layouts can be used
    without rewriting preprocessing code.
    """
    data = _load_numeric_table(path)

    def optional_column(index: int | None) -> np.ndarray | None:
        if index is None or index >= data.shape[1]:
            return None
        return data[:, index].astype(float)

    required = [time_col, meta_cv_col, cv_col]
    if max(required) >= data.shape[1]:
        raise ValueError(f"COLVAR file {path} has only {data.shape[1]} columns.")

    return MetadynamicsTrajectory(
        time=data[:, time_col].astype(float),
        meta_cv=data[:, meta_cv_col].astype(float),
        cv=data[:, cv_col].astype(float),
        bias=optional_column(bias_col),
        lower_boundary_potential=optional_column(lower_boundary_col),
        upper_boundary_potential=optional_column(upper_boundary_col),
    )


def load_fes(
    path: str | Path,
    *,
    x_col: int = 0,
    pmf_col: int = 1,
    derivative_col: int | None = 2,
) -> MetadynamicsReference:
    """Load a metadynamics FES/PMF file with optional derivative column."""
    data = _load_numeric_table(path)
    if max(x_col, pmf_col) >= data.shape[1]:
        raise ValueError(f"FES file {path} has only {data.shape[1]} columns.")
    derivative = (
        data[:, derivative_col].astype(float)
        if derivative_col is not None and derivative_col < data.shape[1]
        else None
    )
    return MetadynamicsReference(
        x=data[:, x_col].astype(float),
        pmf=data[:, pmf_col].astype(float),
        derivative=derivative,
    )


def metad_reference_curves(reference: MetadynamicsReference | None) -> ReferenceCurves:
    """Represent a metadynamics PMF as a generic reference curve."""
    if reference is None:
        return ReferenceCurves()
    return ReferenceCurves(wham_x=reference.x, wham_f=reference.pmf, wham_e=None)


def _filtered_trajectory(
    trajectory: MetadynamicsTrajectory,
    *,
    interval: tuple[float, float],
    time_fraction: float,
) -> MetadynamicsTrajectory:
    if not 0.0 < time_fraction <= 1.0:
        raise ValueError("time_fraction must be in (0, 1].")
    n_keep = max(1, int(np.floor(trajectory.cv.shape[0] * time_fraction)))
    mask = (trajectory.cv[:n_keep] >= interval[0]) & (trajectory.cv[:n_keep] <= interval[1])

    def trim(value: np.ndarray | None) -> np.ndarray | None:
        return None if value is None else value[:n_keep][mask]

    return MetadynamicsTrajectory(
        time=trajectory.time[:n_keep][mask],
        meta_cv=trajectory.meta_cv[:n_keep][mask],
        cv=trajectory.cv[:n_keep][mask],
        bias=trim(trajectory.bias),
        lower_boundary_potential=trim(trajectory.lower_boundary_potential),
        upper_boundary_potential=trim(trajectory.upper_boundary_potential),
    )


def _bin_edges_from_interval(
    interval: tuple[float, float],
    *,
    bin_width: float | None,
    n_bins: int | None,
) -> np.ndarray:
    lower, upper = interval
    if upper <= lower:
        raise ValueError("interval upper bound must exceed lower bound.")
    if bin_width is None and n_bins is None:
        raise ValueError("Provide either bin_width or n_bins.")
    if bin_width is not None:
        if bin_width <= 0.0:
            raise ValueError("bin_width must be positive.")
        n_bins = int(np.ceil((upper - lower) / bin_width))
    assert n_bins is not None
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    return np.linspace(lower, upper, n_bins + 1)


def process_lagrangian_metadynamics(
    trajectory: MetadynamicsTrajectory,
    *,
    force_constant: float,
    interval: tuple[float, float],
    histogram_bin_width: float | None = None,
    n_histogram_windows: int | None = None,
    histogram_radius_bins: float = 5.0,
    derivative_bin_width: float | None = None,
    n_derivative_bins: int | None = None,
    time_fraction: float = 1.0,
    min_window_samples: int = 5,
    min_derivative_samples: int = 5,
    probability_floor: float = 1e-12,
) -> MetadynamicsProcessed:
    """Convert Lagrangian metadynamics trajectories into pseudo-window data.

    Samples are grouped by the metadynamics coordinate ``MetaCV`` to create
    moving harmonic-window histograms. The corresponding derivative
    observations are averages of the restraint force binned by the physical CV.
    """
    if force_constant <= 0.0:
        raise ValueError("force_constant must be positive.")
    if histogram_radius_bins <= 0.0:
        raise ValueError("histogram_radius_bins must be positive.")

    traj = _filtered_trajectory(
        trajectory, interval=interval, time_fraction=time_fraction
    )
    if traj.cv.size == 0:
        raise ValueError("No metadynamics samples remain after filtering.")

    hist_edges = _bin_edges_from_interval(
        interval, bin_width=histogram_bin_width, n_bins=n_histogram_windows
    )
    hist_bin_width = float(np.mean(np.diff(hist_edges)))
    centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])

    histogram_counts: list[torch.Tensor] = []
    histogram_probs: list[torch.Tensor] = []
    bin_centers_list: list[torch.Tensor] = []
    kept_centers: list[float] = []
    n_samples: list[float] = []
    autocorr_times: list[float] = []

    for index, center in enumerate(centers):
        left, right = hist_edges[index], hist_edges[index + 1]
        mask = (traj.meta_cv >= left) & (traj.meta_cv < right)
        radius = histogram_radius_bins * hist_bin_width
        mask &= np.abs(traj.cv - center) <= radius
        cv_window = traj.cv[mask]
        if cv_window.size < min_window_samples:
            continue

        counts_np, _ = np.histogram(cv_window, bins=hist_edges)
        positive = counts_np > 0
        if not np.any(positive):
            continue

        counts = torch.tensor(counts_np[positive], dtype=torch.float64)
        probs = torch.clamp(counts / counts.sum(), min=probability_floor)
        bin_centers = torch.tensor(centers[positive], dtype=torch.float64)

        kept_centers.append(float(center))
        histogram_counts.append(counts)
        histogram_probs.append(probs)
        bin_centers_list.append(bin_centers)
        n_samples.append(float(cv_window.size))
        if cv_window.size >= 3:
            autocorr_times.append(
                max(
                    1e-6,
                    bayes_autocorrelation_time(
                        torch.tensor(cv_window, dtype=torch.float64)
                    ),
                )
            )
        else:
            autocorr_times.append(1.0)

    if not histogram_counts:
        raise ValueError("No pseudo-window histograms passed the sample filters.")

    derivative_edges = _bin_edges_from_interval(
        interval, bin_width=derivative_bin_width, n_bins=n_derivative_bins
    )
    derivative_centers = 0.5 * (derivative_edges[:-1] + derivative_edges[1:])
    restraint_derivative = force_constant * (traj.meta_cv - traj.cv)

    dx: list[float] = []
    dy: list[float] = []
    dy_var: list[float] = []
    dy_counts: list[float] = []
    dy_tau: list[float] = []
    for index, center in enumerate(derivative_centers):
        left, right = derivative_edges[index], derivative_edges[index + 1]
        mask = (traj.cv >= left) & (traj.cv < right)
        values = restraint_derivative[mask]
        if values.size < min_derivative_samples:
            continue
        values_t = torch.tensor(values, dtype=torch.float64)
        tau = max(1e-6, bayes_autocorrelation_time(values_t)) if values.size >= 3 else 1.0
        n_eff = max(values.size / tau, 1.0)
        sample_var = float(values_t.var(unbiased=True).item()) if values.size > 1 else 0.0
        dx.append(float(center))
        dy.append(float(values_t.mean().item()))
        dy_var.append(max(sample_var / n_eff, 1e-12))
        dy_counts.append(float(values.size))
        dy_tau.append(float(tau))

    if not dx:
        raise ValueError("No derivative bins passed the sample filters.")

    return MetadynamicsProcessed(
        window_centers=torch.tensor(kept_centers, dtype=torch.float64),
        histogram_counts=histogram_counts,
        histogram_probs=histogram_probs,
        bin_centers_list=bin_centers_list,
        n_samples_per_window=torch.tensor(n_samples, dtype=torch.float64),
        autocorr_times=torch.tensor(autocorr_times, dtype=torch.float64),
        derivative_bin_centers=torch.tensor(dx, dtype=torch.float64),
        derivative_means=torch.tensor(dy, dtype=torch.float64),
        derivative_variances=torch.tensor(dy_var, dtype=torch.float64),
        derivative_sample_counts=torch.tensor(dy_counts, dtype=torch.float64),
        derivative_autocorr_times=torch.tensor(dy_tau, dtype=torch.float64),
    )


def build_metadynamics_joint_observations(
    processed: MetadynamicsProcessed,
    *,
    force_constant: float,
    probability_floor: float = 1e-12,
    covariance_regularization: float = 1e-8,
    free_energy_offset: float = 0.0,
) -> JointObservations:
    """Build standard joint GP observations from processed metadynamics data."""
    F_list: list[torch.Tensor] = []
    for center, probs, bin_centers in zip(
        processed.window_centers,
        processed.histogram_probs,
        processed.bin_centers_list,
    ):
        probs = torch.clamp(probs, min=probability_floor)
        y = -(1.0 / beta) * torch.log(probs)
        restraint = 0.5 * force_constant * (bin_centers - center) ** 2
        F_list.append(y - restraint + free_energy_offset)

    x_obs = torch.cat([bins.reshape(-1) for bins in processed.bin_centers_list])
    y_obs = torch.cat([values.reshape(-1) for values in F_list])

    bin_counts = [int(bins.numel()) for bins in processed.bin_centers_list]
    n_windows = len(bin_counts)
    window_idx_per_bin = torch.cat(
        [torch.full((count,), i, dtype=torch.long) for i, count in enumerate(bin_counts)]
    )
    n_obs = x_obs.shape[0]
    H_obs = torch.zeros((n_obs, n_windows), dtype=torch.float64)
    H_obs[torch.arange(n_obs), window_idx_per_bin] = 1.0

    n_eff = processed.n_samples_per_window / processed.autocorr_times
    noise_cov_matrix = torch.zeros((n_obs, n_obs), dtype=torch.float64)
    obs_start_idx = 0
    for window_i, n_bins_i in enumerate(bin_counts):
        probs_i = torch.clamp(processed.histogram_probs[window_i], min=probability_floor)
        n_eff_i = max(float(n_eff[window_i].item()), 1.0)
        obs_end_idx = obs_start_idx + n_bins_i
        base_cov = 1.0 / (beta**2 * n_eff_i)
        for i in range(n_bins_i):
            obs_i = obs_start_idx + i
            p_i = float(probs_i[i].item())
            noise_cov_matrix[obs_i, obs_i] = base_cov * ((1.0 / p_i) - 1.0)
            for j in range(i + 1, n_bins_i):
                obs_j = obs_start_idx + j
                noise_cov_matrix[obs_i, obs_j] = -base_cov
                noise_cov_matrix[obs_j, obs_i] = -base_cov
        obs_start_idx = obs_end_idx
    noise_cov_matrix += covariance_regularization * torch.eye(n_obs, dtype=torch.float64)

    return JointObservations(
        x_obs=x_obs.reshape(-1),
        y_obs=y_obs.reshape(-1),
        H_obs=H_obs,
        x_der=processed.derivative_bin_centers.reshape(-1),
        dy_der=processed.derivative_means.reshape(-1),
        noise_func_cov=noise_cov_matrix,
        noise_deriv_diag=processed.derivative_variances.reshape(-1),
        F_list=F_list,
    )
