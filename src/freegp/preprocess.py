"""Preprocessing for the joint histogram + derivative GP model."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import UmbrellaWindow

k_B = 8.3144621e-3
T = 303.15
beta = 1 / (k_B * T)


@dataclass(frozen=True)
class ProcessedUmbrellaData:
    folder_numbers: torch.Tensor
    force_constants: torch.Tensor
    modes: torch.Tensor
    variances: torch.Tensor
    autocorr_times: torch.Tensor
    n_samples: torch.Tensor
    restoring_forces: torch.Tensor
    histogram_counts: list[torch.Tensor]
    histogram_probs: list[torch.Tensor]
    histogram_densities: list[torch.Tensor]
    bin_centers_list: list[torch.Tensor]


@dataclass(frozen=True)
class JointObservations:
    x_obs: torch.Tensor
    y_obs: torch.Tensor
    H_obs: torch.Tensor
    x_der: torch.Tensor
    dy_der: torch.Tensor
    noise_func_cov: torch.Tensor
    noise_deriv_diag: torch.Tensor
    F_list: list[torch.Tensor]


def bayes_autocorrelation_time(
    x: torch.Tensor,
    *,
    prior_mean: float = 0.0,
    prior_precision: float = 1e-2,
    eps: float = 1e-8,
) -> float:
    """AR(1)-style Bayesian autocorrelation-time estimate."""
    x = x.flatten()
    x_centered = x - x.mean()
    x_prev = x_centered[:-1]
    x_curr = x_centered[1:]

    sum_xx = torch.sum(x_prev * x_prev)
    sum_xy = torch.sum(x_prev * x_curr)

    post_precision = prior_precision + sum_xx
    post_mean = (prior_precision * prior_mean + sum_xy) / (post_precision + eps)
    phi_post = torch.tanh(post_mean)
    tau = (1 + phi_post) / (1 - phi_post)
    return tau.item()


def process_umbrella_windows(
    windows: list[UmbrellaWindow],
    *,
    n_equilibration: int = 40_000,
    num_bins: int = 20,
) -> ProcessedUmbrellaData:
    """Turn raw umbrella windows into sorted per-window summary statistics."""
    folder_numbers = []
    force_constants = []
    modes = []
    variances = []
    autocorr_times = []
    n_samples = []
    histogram_counts = []
    histogram_probs = []
    histogram_densities = []
    bin_centers_list = []

    for window in windows:
        position = window.position
        position_eq = position[n_equilibration:] if len(position) > n_equilibration else position
        if position_eq.numel() == 0:
            raise ValueError(f"Window {window.folder} has no usable samples after equilibration.")

        counts, bin_edges = torch.histogram(position_eq, bins=num_bins, density=False)
        counts = counts.to(torch.float64)
        total = counts.sum()
        if total <= 0:
            raise ValueError(f"Window {window.folder} produced an empty histogram.")

        probs = counts / total
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        densities = probs / bin_widths
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        max_idx = torch.argmax(densities)

        folder_numbers.append(window.folder_number)
        force_constants.append(float(window.mdp_last_value))
        modes.append(bin_centers[max_idx].item())
        variances.append(position_eq.var(unbiased=True).item())
        autocorr_times.append(bayes_autocorrelation_time(position_eq))
        n_samples.append(float(position_eq.numel()))
        histogram_counts.append(counts)
        histogram_probs.append(probs)
        histogram_densities.append(densities)
        bin_centers_list.append(bin_centers)

    folder_numbers_t = torch.tensor(folder_numbers, dtype=torch.float64)
    sorted_idx = torch.argsort(folder_numbers_t)

    folder_numbers_t = folder_numbers_t[sorted_idx]
    force_constants_t = torch.tensor(force_constants, dtype=torch.float64)[sorted_idx]
    modes_t = torch.tensor(modes, dtype=torch.float64)[sorted_idx]
    variances_t = torch.tensor(variances, dtype=torch.float64)[sorted_idx]
    autocorr_times_t = torch.tensor(autocorr_times, dtype=torch.float64)[sorted_idx]
    n_samples_t = torch.tensor(n_samples, dtype=torch.float64)[sorted_idx]

    order = sorted_idx.tolist()
    histogram_counts = [histogram_counts[i] for i in order]
    histogram_probs = [histogram_probs[i] for i in order]
    histogram_densities = [histogram_densities[i] for i in order]
    bin_centers_list = [bin_centers_list[i] for i in order]

    restoring_forces = -(modes_t - folder_numbers_t) * force_constants_t

    return ProcessedUmbrellaData(
        folder_numbers=folder_numbers_t,
        force_constants=force_constants_t,
        modes=modes_t,
        variances=variances_t,
        autocorr_times=autocorr_times_t,
        n_samples=n_samples_t,
        restoring_forces=restoring_forces,
        histogram_counts=histogram_counts,
        histogram_probs=histogram_probs,
        histogram_densities=histogram_densities,
        bin_centers_list=bin_centers_list,
    )


def build_joint_observations(
    processed: ProcessedUmbrellaData,
    *,
    probability_floor: float = 1e-12,
    covariance_regularization: float = 1e-8,
) -> JointObservations:
    """Build the flattened GP inputs used by the HMC-NUTS notebook."""
    F_list: list[torch.Tensor] = []
    for i in range(len(processed.histogram_probs)):
        probs = torch.clamp(processed.histogram_probs[i], min=probability_floor)
        y = -(1 / beta) * torch.log(probs)
        w = 0.5 * processed.force_constants[i] * (
            processed.bin_centers_list[i] - processed.folder_numbers[i]
        ) ** 2
        F_list.append(y - w)

    x_obs = torch.cat([bc.reshape(-1) for bc in processed.bin_centers_list]).reshape(-1)
    y_obs = torch.cat([fi.reshape(-1) for fi in F_list]).reshape(-1)

    bin_counts = [int(bc.numel()) for bc in processed.bin_centers_list]
    n_windows = len(bin_counts)
    window_idx_per_bin = torch.cat(
        [torch.full((count,), i, dtype=torch.long) for i, count in enumerate(bin_counts)]
    )

    n_obs = x_obs.shape[0]
    H_obs = torch.zeros((n_obs, n_windows), dtype=torch.float64)
    H_obs[torch.arange(n_obs), window_idx_per_bin] = 1.0

    n_eff = processed.n_samples / processed.autocorr_times
    noise_cov_matrix = torch.zeros((n_obs, n_obs), dtype=torch.float64)

    obs_start_idx = 0
    for window_i, n_bins_i in enumerate(bin_counts):
        probs_i = torch.clamp(processed.histogram_probs[window_i], min=probability_floor)
        n_eff_i = n_eff[window_i].item()
        obs_end_idx = obs_start_idx + n_bins_i
        base_cov = 1.0 / (beta**2 * n_eff_i)

        for i in range(n_bins_i):
            obs_i = obs_start_idx + i
            p_i = probs_i[i].item()
            noise_cov_matrix[obs_i, obs_i] = base_cov * ((1.0 / p_i) - 1.0)
            for j in range(i + 1, n_bins_i):
                obs_j = obs_start_idx + j
                noise_cov_matrix[obs_i, obs_j] = -base_cov
                noise_cov_matrix[obs_j, obs_i] = -base_cov

        obs_start_idx = obs_end_idx

    noise_cov_matrix += covariance_regularization * torch.eye(n_obs, dtype=torch.float64)

    noise_var_deriv = (
        processed.force_constants**2 * processed.variances
    ) / (processed.n_samples / processed.autocorr_times)

    return JointObservations(
        x_obs=x_obs,
        y_obs=y_obs,
        H_obs=H_obs,
        x_der=processed.folder_numbers.clone().reshape(-1),
        dy_der=processed.restoring_forces.clone().reshape(-1),
        noise_func_cov=noise_cov_matrix,
        noise_deriv_diag=noise_var_deriv.reshape(-1),
        F_list=F_list,
    )


def build_test_grid(
    processed: ProcessedUmbrellaData,
    *,
    num_points: int = 400,
    x_min: float | None = None,
    x_max: float | None = None,
    source: str = "umbrella_centers",
) -> torch.Tensor:
    """Create the prediction grid.

    If ``x_min``/``x_max`` are omitted, the grid spans either the umbrella-center
    range or the histogram-support range depending on ``source``.
    """
    if x_min is None:
        if source == "histogram_support":
            x_min = float(min(b.min().item() for b in processed.bin_centers_list))
        else:
            x_min = float(processed.folder_numbers.min().item())
    if x_max is None:
        if source == "histogram_support":
            x_max = float(max(b.max().item() for b in processed.bin_centers_list))
        else:
            x_max = float(processed.folder_numbers.max().item())
    return torch.linspace(x_min, x_max, num_points, dtype=torch.float64)
