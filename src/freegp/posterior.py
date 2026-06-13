"""Hyperposterior predictive summaries and variance decompositions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .gp import (
    JointGPPosterior,
    predict_derivative,
    predict_function,
    predict_function_mean,
)
from .hmc import (
    HyperPriorConfig,
    NUTSConfig,
    _posterior_for_parameters,
    _sample_site_names,
    _state_to_parameters_and_log_prior,
)
from .preprocess import JointObservations


@dataclass(frozen=True)
class HyperposteriorPredictiveSummary:
    x_test: torch.Tensor
    mean: torch.Tensor
    total_cov: torch.Tensor
    within_cov: torch.Tensor
    between_cov: torch.Tensor
    conditional_means: torch.Tensor
    conditional_covariances: torch.Tensor
    selected_indices: torch.Tensor

    @property
    def total_variance(self) -> torch.Tensor:
        return torch.diagonal(self.total_cov)

    @property
    def within_variance(self) -> torch.Tensor:
        return torch.diagonal(self.within_cov)

    @property
    def between_variance(self) -> torch.Tensor:
        return torch.diagonal(self.between_cov)


def _monte_carlo_indices(n_available: int, max_samples: int | None) -> torch.Tensor:
    if max_samples is None or max_samples >= n_available:
        return torch.arange(n_available, dtype=torch.long)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive or None.")
    if max_samples == 1:
        return torch.tensor([n_available - 1], dtype=torch.long)
    return torch.from_numpy(np.linspace(0, n_available - 1, max_samples, dtype=int)).to(torch.long)


def _summarize_conditionals(
    x_test: torch.Tensor,
    conditional_means: list[torch.Tensor],
    conditional_covariances: list[torch.Tensor],
    selected_indices: torch.Tensor,
) -> HyperposteriorPredictiveSummary:
    conditional_means_t = torch.stack(conditional_means, dim=0)
    conditional_covariances_t = torch.stack(conditional_covariances, dim=0)
    mean = conditional_means_t.mean(dim=0)
    within_cov = conditional_covariances_t.mean(dim=0)
    centered = conditional_means_t - mean
    between_cov = centered.T @ centered / conditional_means_t.shape[0]
    total_cov = within_cov + between_cov
    return HyperposteriorPredictiveSummary(
        x_test=x_test,
        mean=mean,
        total_cov=0.5 * (total_cov + total_cov.T),
        within_cov=0.5 * (within_cov + within_cov.T),
        between_cov=0.5 * (between_cov + between_cov.T),
        conditional_means=conditional_means_t,
        conditional_covariances=conditional_covariances_t,
        selected_indices=selected_indices,
    )


def summarize_hyperposterior_predictive(
    observations: JointObservations,
    samples: dict[str, torch.Tensor],
    x_test: torch.Tensor,
    *,
    priors: HyperPriorConfig | None = None,
    config: NUTSConfig | None = None,
    max_samples: int | None = None,
) -> HyperposteriorPredictiveSummary:
    """Approximate p(f_* | D) with Monte Carlo over the hyperposterior.

    Uses the law of total expectation and law of total variance:

      E[f_* | D]   = E_theta[E[f_* | D, theta]]
      Var[f_* | D] = E_theta[Var(f_* | D, theta)] + Var_theta(E[f_* | D, theta])
    """
    priors = priors or HyperPriorConfig()
    config = config or NUTSConfig()
    site_names = _sample_site_names(config)
    n_available = samples[site_names[0]].shape[0]
    selected_indices = _monte_carlo_indices(n_available, max_samples)

    conditional_means = []
    conditional_covariances = []
    for idx in selected_indices.tolist():
        sample_state = {name: samples[name][idx] for name in site_names}
        params, _ = _state_to_parameters_and_log_prior(
            sample_state,
            observations,
            priors=priors,
            config=config,
        )
        posterior = _posterior_for_parameters(params, observations, config=config)
        pred_mean, pred_cov = predict_function(posterior, x_test)
        conditional_means.append(pred_mean)
        conditional_covariances.append(pred_cov)

    return _summarize_conditionals(
        x_test=x_test,
        conditional_means=conditional_means,
        conditional_covariances=conditional_covariances,
        selected_indices=selected_indices,
    )


def summarize_hyperposterior_derivative(
    observations: JointObservations,
    samples: dict[str, torch.Tensor],
    x_test: torch.Tensor,
    *,
    priors: HyperPriorConfig | None = None,
    config: NUTSConfig | None = None,
    max_samples: int | None = None,
) -> HyperposteriorPredictiveSummary:
    """Approximate the derivative predictive distribution over the hyperposterior."""
    priors = priors or HyperPriorConfig()
    config = config or NUTSConfig()
    site_names = _sample_site_names(config)
    n_available = samples[site_names[0]].shape[0]
    selected_indices = _monte_carlo_indices(n_available, max_samples)

    conditional_means = []
    conditional_covariances = []
    for idx in selected_indices.tolist():
        sample_state = {name: samples[name][idx] for name in site_names}
        params, _ = _state_to_parameters_and_log_prior(
            sample_state,
            observations,
            priors=priors,
            config=config,
        )
        posterior = _posterior_for_parameters(params, observations, config=config)
        pred_mean, pred_cov = predict_derivative(posterior, x_test)
        conditional_means.append(pred_mean)
        conditional_covariances.append(pred_cov)

    return _summarize_conditionals(
        x_test=x_test,
        conditional_means=conditional_means,
        conditional_covariances=conditional_covariances,
        selected_indices=selected_indices,
    )


def hyperposterior_conditional_means(
    observations: JointObservations,
    samples: dict[str, torch.Tensor],
    x_test: torch.Tensor,
    *,
    priors: HyperPriorConfig | None = None,
    config: NUTSConfig | None = None,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate conditional function means without constructing predictive covariances."""
    priors = priors or HyperPriorConfig()
    config = config or NUTSConfig()
    site_names = _sample_site_names(config)
    n_available = samples[site_names[0]].shape[0]
    selected_indices = _monte_carlo_indices(n_available, max_samples)
    conditional_means = []
    for idx in selected_indices.tolist():
        sample_state = {name: samples[name][idx] for name in site_names}
        params, _ = _state_to_parameters_and_log_prior(
            sample_state,
            observations,
            priors=priors,
            config=config,
        )
        posterior = _posterior_for_parameters(params, observations, config=config)
        conditional_means.append(predict_function_mean(posterior, x_test))
    return torch.stack(conditional_means, dim=0), selected_indices


def summarize_fixed_posterior_predictive(
    posterior: JointGPPosterior,
    x_test: torch.Tensor,
) -> HyperposteriorPredictiveSummary:
    """Wrap a single fixed-hyperparameter GP posterior in the same summary interface."""
    mean, cov = predict_function(posterior, x_test)
    zero_cov = torch.zeros_like(cov)
    return HyperposteriorPredictiveSummary(
        x_test=x_test,
        mean=mean,
        total_cov=cov,
        within_cov=cov,
        between_cov=zero_cov,
        conditional_means=mean.reshape(1, -1),
        conditional_covariances=cov.reshape(1, *cov.shape),
        selected_indices=torch.tensor([0], dtype=torch.long),
    )


def summarize_fixed_posterior_derivative(
    posterior: JointGPPosterior,
    x_test: torch.Tensor,
) -> HyperposteriorPredictiveSummary:
    """Wrap a fixed-hyperparameter derivative posterior in the summary interface."""
    mean, cov = predict_derivative(posterior, x_test)
    zero_cov = torch.zeros_like(cov)
    return HyperposteriorPredictiveSummary(
        x_test=x_test,
        mean=mean,
        total_cov=cov,
        within_cov=cov,
        between_cov=zero_cov,
        conditional_means=mean.reshape(1, -1),
        conditional_covariances=cov.reshape(1, *cov.shape),
        selected_indices=torch.tensor([0], dtype=torch.long),
    )


def anchor_predictive_summary(
    summary: HyperposteriorPredictiveSummary,
    *,
    anchor_index: int = 0,
) -> HyperposteriorPredictiveSummary:
    """Convert function predictions to contrasts against one prediction-grid point."""
    n_points = summary.mean.numel()
    if not 0 <= anchor_index < n_points:
        raise IndexError("anchor_index is outside the prediction grid.")
    transform = torch.eye(
        n_points, dtype=summary.mean.dtype, device=summary.mean.device
    )
    transform[:, anchor_index] -= 1.0
    conditional_means = summary.conditional_means @ transform.T
    conditional_covariances = torch.stack(
        [transform @ cov @ transform.T for cov in summary.conditional_covariances],
        dim=0,
    )
    return _summarize_conditionals(
        x_test=summary.x_test,
        conditional_means=list(conditional_means),
        conditional_covariances=list(conditional_covariances),
        selected_indices=summary.selected_indices,
    )
