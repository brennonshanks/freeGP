"""Preprocessing for the joint histogram + derivative GP model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .data import UmbrellaWindow, UmbrellaWindowND

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


def move_processed_umbrella_data(
    processed: ProcessedUmbrellaData,
    *,
    device: torch.device | str,
) -> ProcessedUmbrellaData:
    """Return a copy of ``processed`` moved onto ``device``."""
    return replace(
        processed,
        folder_numbers=processed.folder_numbers.to(device=device),
        force_constants=processed.force_constants.to(device=device),
        modes=processed.modes.to(device=device),
        variances=processed.variances.to(device=device),
        autocorr_times=processed.autocorr_times.to(device=device),
        n_samples=processed.n_samples.to(device=device),
        restoring_forces=processed.restoring_forces.to(device=device),
        histogram_counts=[value.to(device=device) for value in processed.histogram_counts],
        histogram_probs=[value.to(device=device) for value in processed.histogram_probs],
        histogram_densities=[value.to(device=device) for value in processed.histogram_densities],
        bin_centers_list=[value.to(device=device) for value in processed.bin_centers_list],
    )


def move_joint_observations(
    observations: JointObservations,
    *,
    device: torch.device | str,
) -> JointObservations:
    """Return a copy of ``observations`` moved onto ``device``."""
    return replace(
        observations,
        x_obs=observations.x_obs.to(device=device),
        y_obs=observations.y_obs.to(device=device),
        H_obs=observations.H_obs.to(device=device),
        x_der=observations.x_der.to(device=device),
        dy_der=observations.dy_der.to(device=device),
        noise_func_cov=observations.noise_func_cov.to(device=device),
        noise_deriv_diag=observations.noise_deriv_diag.to(device=device),
        F_list=[value.to(device=device) for value in observations.F_list],
    )


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


# ---------------------------------------------------------------------------
# Multidimensional (ND) preprocessing
#
# Generalizes the 1D pipeline above to windows biased by an isotropic harmonic
# restraint in D collective variables. Each window's histogram becomes a D-way
# grid, and the single restoring-force scalar becomes a D-component gradient
# estimate of the free-energy surface at the window center.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessedUmbrellaDataND:
    n_dim: int
    folder_numbers: torch.Tensor  # (n_windows, D) window centers
    force_constants: torch.Tensor  # (n_windows,)
    modes: torch.Tensor  # (n_windows, D) histogram-mode location per window
    covariances: torch.Tensor  # (n_windows, D, D) sample covariance of position
    autocorr_times: torch.Tensor  # (n_windows,)
    n_samples: torch.Tensor  # (n_windows,)
    restoring_forces: torch.Tensor  # (n_windows, D)
    histogram_counts: list[torch.Tensor]
    histogram_probs: list[torch.Tensor]
    histogram_densities: list[torch.Tensor]
    bin_centers_list: list[torch.Tensor]  # each (n_bins_i, D)


@dataclass(frozen=True)
class JointObservationsND:
    n_dim: int
    x_obs: torch.Tensor  # (n_obs, D)
    y_obs: torch.Tensor  # (n_obs,)
    H_obs: torch.Tensor  # (n_obs, n_windows)
    x_der: torch.Tensor  # (n_windows, D)
    dy_der: torch.Tensor  # (n_windows, D)
    noise_func_cov: torch.Tensor  # (n_obs, n_obs)
    noise_deriv_cov: torch.Tensor  # (n_windows, D, D)
    F_list: list[torch.Tensor]


def move_processed_umbrella_data_nd(
    processed: ProcessedUmbrellaDataND,
    *,
    device: torch.device | str,
) -> ProcessedUmbrellaDataND:
    """Return a copy of ``processed`` moved onto ``device``."""
    return replace(
        processed,
        folder_numbers=processed.folder_numbers.to(device=device),
        force_constants=processed.force_constants.to(device=device),
        modes=processed.modes.to(device=device),
        covariances=processed.covariances.to(device=device),
        autocorr_times=processed.autocorr_times.to(device=device),
        n_samples=processed.n_samples.to(device=device),
        restoring_forces=processed.restoring_forces.to(device=device),
        histogram_counts=[value.to(device=device) for value in processed.histogram_counts],
        histogram_probs=[value.to(device=device) for value in processed.histogram_probs],
        histogram_densities=[value.to(device=device) for value in processed.histogram_densities],
        bin_centers_list=[value.to(device=device) for value in processed.bin_centers_list],
    )


def move_joint_observations_nd(
    observations: JointObservationsND,
    *,
    device: torch.device | str,
) -> JointObservationsND:
    """Return a copy of ``observations`` moved onto ``device``."""
    return replace(
        observations,
        x_obs=observations.x_obs.to(device=device),
        y_obs=observations.y_obs.to(device=device),
        H_obs=observations.H_obs.to(device=device),
        x_der=observations.x_der.to(device=device),
        dy_der=observations.dy_der.to(device=device),
        noise_func_cov=observations.noise_func_cov.to(device=device),
        noise_deriv_cov=observations.noise_deriv_cov.to(device=device),
        F_list=[value.to(device=device) for value in observations.F_list],
    )


def process_umbrella_windows_nd(
    windows: list[UmbrellaWindowND],
    *,
    n_dim: int,
    n_equilibration: int = 0,
    num_bins: int = 6,
) -> ProcessedUmbrellaDataND:
    """Turn raw ND umbrella windows into per-window summary statistics.

    Each window's samples are binned into a D-way histogram (``num_bins`` per
    dimension). The bin with peak density approximates the window's mean
    position (as in the 1D pipeline), from which the D-component restoring
    force -- and hence an estimate of the free-energy gradient -- is derived.
    """
    folder_numbers = []
    force_constants = []
    modes = []
    covariances = []
    autocorr_times = []
    n_samples = []
    histogram_counts = []
    histogram_probs = []
    histogram_densities = []
    bin_centers_list = []

    for window in windows:
        position = window.position
        position_eq = position[n_equilibration:] if position.shape[0] > n_equilibration else position
        if position_eq.shape[0] == 0:
            raise ValueError(f"Window {window.folder} has no usable samples after equilibration.")

        counts, edges = torch.histogramdd(position_eq, bins=num_bins)
        counts = counts.to(torch.float64)
        total = counts.sum()
        if total <= 0:
            raise ValueError(f"Window {window.folder} produced an empty histogram.")

        probs = counts / total
        bin_widths = [e[1:] - e[:-1] for e in edges]
        cell_volume = bin_widths[0]
        for bw in bin_widths[1:]:
            cell_volume = cell_volume[..., None] * bw
        densities = probs / cell_volume

        centers_per_dim = [0.5 * (e[1:] + e[:-1]) for e in edges]
        mesh = torch.meshgrid(*centers_per_dim, indexing="ij")
        bin_centers = torch.stack(mesh, dim=-1).reshape(-1, n_dim)
        flat_densities = densities.reshape(-1)
        flat_probs = probs.reshape(-1)
        flat_counts = counts.reshape(-1)

        max_idx = torch.argmax(flat_densities)

        folder_numbers.append(window.center)
        force_constants.append(window.force_constant)
        modes.append(bin_centers[max_idx])
        cov = (
            torch.cov(position_eq.T)
            if position_eq.shape[0] > 1
            else torch.zeros((n_dim, n_dim), dtype=torch.float64)
        )
        covariances.append(cov.reshape(n_dim, n_dim))
        autocorr_times.append(
            max(bayes_autocorrelation_time(position_eq[:, d]) for d in range(n_dim))
        )
        n_samples.append(float(position_eq.shape[0]))
        histogram_counts.append(flat_counts)
        histogram_probs.append(flat_probs)
        histogram_densities.append(flat_densities)
        bin_centers_list.append(bin_centers)

    folder_numbers_t = torch.stack(folder_numbers, dim=0)
    force_constants_t = torch.tensor(force_constants, dtype=torch.float64)
    modes_t = torch.stack(modes, dim=0)
    covariances_t = torch.stack(covariances, dim=0)
    autocorr_times_t = torch.tensor(autocorr_times, dtype=torch.float64)
    n_samples_t = torch.tensor(n_samples, dtype=torch.float64)

    restoring_forces = -(modes_t - folder_numbers_t) * force_constants_t.reshape(-1, 1)

    return ProcessedUmbrellaDataND(
        n_dim=n_dim,
        folder_numbers=folder_numbers_t,
        force_constants=force_constants_t,
        modes=modes_t,
        covariances=covariances_t,
        autocorr_times=autocorr_times_t,
        n_samples=n_samples_t,
        restoring_forces=restoring_forces,
        histogram_counts=histogram_counts,
        histogram_probs=histogram_probs,
        histogram_densities=histogram_densities,
        bin_centers_list=bin_centers_list,
    )


def build_joint_observations_nd(
    processed: ProcessedUmbrellaDataND,
    *,
    probability_floor: float = 1e-12,
    covariance_regularization: float = 1e-8,
) -> JointObservationsND:
    """Build the flattened ND GP inputs analogous to ``build_joint_observations``."""
    n_dim = processed.n_dim
    F_list: list[torch.Tensor] = []
    for i in range(len(processed.histogram_probs)):
        probs = torch.clamp(processed.histogram_probs[i], min=probability_floor)
        y = -(1 / beta) * torch.log(probs)
        offset = processed.bin_centers_list[i] - processed.folder_numbers[i]
        w = 0.5 * processed.force_constants[i] * (offset**2).sum(dim=-1)
        F_list.append(y - w)

    x_obs = torch.cat(processed.bin_centers_list, dim=0)
    y_obs = torch.cat([fi.reshape(-1) for fi in F_list])

    bin_counts = [int(bc.shape[0]) for bc in processed.bin_centers_list]
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
        base_cov = 1.0 / (beta**2 * n_eff[window_i].item())
        obs_end_idx = obs_start_idx + n_bins_i
        # Multinomial log-probability covariance: diag = base_cov*(1/p_i - 1),
        # off-diag = -base_cov (independent of the pair of bins).
        block = -base_cov * torch.ones((n_bins_i, n_bins_i), dtype=torch.float64)
        block.fill_diagonal_(0.0)
        block += torch.diag(base_cov * (1.0 / probs_i - 1.0))
        noise_cov_matrix[obs_start_idx:obs_end_idx, obs_start_idx:obs_end_idx] = block
        obs_start_idx = obs_end_idx

    noise_cov_matrix += covariance_regularization * torch.eye(n_obs, dtype=torch.float64)

    noise_deriv_cov = (
        processed.force_constants.reshape(-1, 1, 1) ** 2
        * processed.covariances
        / n_eff.reshape(-1, 1, 1)
    )

    return JointObservationsND(
        n_dim=n_dim,
        x_obs=x_obs,
        y_obs=y_obs,
        H_obs=H_obs,
        x_der=processed.folder_numbers.clone(),
        dy_der=processed.restoring_forces.clone(),
        noise_func_cov=noise_cov_matrix,
        noise_deriv_cov=noise_deriv_cov,
        F_list=F_list,
    )


def build_test_grid_nd(
    processed: ProcessedUmbrellaDataND,
    *,
    num_points_per_dim: int = 30,
    x_min: list[float] | None = None,
    x_max: list[float] | None = None,
    source: str = "histogram_support",
) -> torch.Tensor:
    """Create a flattened ND grid of test points, shape (num_points_per_dim**D, D).

    If ``x_min``/``x_max`` are omitted per-dimension, each axis spans either the
    umbrella-center range or the histogram-support range depending on ``source``.
    """
    n_dim = processed.n_dim
    axes = []
    for d in range(n_dim):
        lo = x_min[d] if x_min is not None else None
        hi = x_max[d] if x_max is not None else None
        if lo is None:
            lo = (
                float(min(bc[:, d].min().item() for bc in processed.bin_centers_list))
                if source == "histogram_support"
                else float(processed.folder_numbers[:, d].min().item())
            )
        if hi is None:
            hi = (
                float(max(bc[:, d].max().item() for bc in processed.bin_centers_list))
                if source == "histogram_support"
                else float(processed.folder_numbers[:, d].max().item())
            )
        axes.append(torch.linspace(lo, hi, num_points_per_dim, dtype=torch.float64))
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, n_dim)
