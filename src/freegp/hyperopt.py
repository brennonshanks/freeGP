"""Deterministic hyperparameter optimization for the stationary joint GP."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .gp import (
    DerivativeGPPosterior,
    JointGPPosterior,
    build_derivative_gp,
    build_joint_gp,
    derivative_log_marginal_likelihood,
    derivative_loo_loglik,
    joint_log_marginal_likelihood,
    joint_loo_loglik,
)
from .hmc import (
    HyperPriorConfig,
    stationary_ell_prior_distribution,
    stationary_log_ell_bounds,
)
from .preprocess import JointObservations


@dataclass(frozen=True)
class HyperparameterOptimizationResult:
    params: dict[str, torch.Tensor]
    posterior: JointGPPosterior | DerivativeGPPosterior
    objective_value: float
    restart: int
    history: list[float]


def _stationary_posterior(
    observations: JointObservations,
    params: dict[str, torch.Tensor],
    *,
    jitter: float,
    fixed_noise: bool = False,
) -> JointGPPosterior:
    dtype = observations.x_obs.dtype
    device = observations.x_obs.device
    if fixed_noise:
        function_noise = observations.noise_func_cov
        derivative_noise = observations.noise_deriv_diag
    else:
        function_noise = params["sigma_f"].square() * torch.eye(
            observations.x_obs.numel(), dtype=dtype, device=device
        )
        derivative_noise = params["sigma_d"].square() * torch.ones(
            observations.x_der.numel(), dtype=dtype, device=device
        )
    return build_joint_gp(
        x_func=observations.x_obs,
        y_func=observations.y_obs,
        x_der=observations.x_der,
        dy_der=observations.dy_der,
        ell=params["ell"],
        w=params["w"],
        noise_func_cov=function_noise,
        noise_deriv_diag=derivative_noise,
        H_func=observations.H_obs,
        jitter=jitter,
    )


def optimize_stationary_hyperparameters(
    observations: JointObservations,
    *,
    objective: str = "lml",
    priors: HyperPriorConfig | None = None,
    use_hyperpriors: bool = True,
    steps: int = 250,
    learning_rate: float = 0.05,
    restarts: int = 3,
    seed: int = 0,
    jitter: float = 1e-6,
    fixed_noise: bool = False,
) -> HyperparameterOptimizationResult:
    """Find a stationary-kernel MAP or ML estimate.

    If ``fixed_noise`` is true, only ``ell`` and ``w`` are optimized and the
    trajectory-derived observation covariance in ``observations`` is used
    directly. Otherwise ``sigma_f`` and ``sigma_d`` are optimized as nuisance
    noise scales. If ``use_hyperpriors`` is false, the bounded optimizer
    maximizes only the selected likelihood objective.
    """
    if objective not in {"lml", "loo"}:
        raise ValueError("objective must be 'lml' or 'loo'.")
    if steps <= 0 or restarts <= 0:
        raise ValueError("steps and restarts must be positive.")

    priors = priors or HyperPriorConfig()
    dtype = observations.x_obs.dtype
    device = observations.x_obs.device
    x_all = torch.cat((observations.x_obs, observations.x_der))
    x_span = max(float((x_all.max() - x_all.min()).item()), 1e-3)
    lower_ell, upper_ell = stationary_log_ell_bounds(observations)
    if fixed_noise:
        lower = torch.tensor([lower_ell, -6.0], dtype=dtype, device=device)
        upper = torch.tensor([upper_ell, 8.0], dtype=dtype, device=device)
        center = torch.tensor(
            [math.log(max(x_span / 3.0, 1e-3)), math.log(4.184)],
            dtype=dtype,
            device=device,
        )
    else:
        lower = torch.tensor(
            [lower_ell, -6.0, -8.0, -8.0],
            dtype=dtype,
            device=device,
        )
        upper = torch.tensor(
            [upper_ell, 8.0, 6.0, 6.0],
            dtype=dtype,
            device=device,
        )
        center = torch.tensor(
            [math.log(max(x_span / 3.0, 1e-3)), math.log(4.184), math.log(0.5), math.log(0.5)],
            dtype=dtype,
            device=device,
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    best: HyperparameterOptimizationResult | None = None

    for restart in range(restarts):
        initial = center.clone()
        if restart:
            initial = initial + 0.5 * torch.randn(
                initial.shape, dtype=dtype, device=device, generator=generator
            )
        theta = torch.nn.Parameter(torch.clamp(initial, min=lower, max=upper))
        optimizer = torch.optim.Adam([theta], lr=learning_rate)
        history: list[float] = []

        for _ in range(steps):
            optimizer.zero_grad()
            values = torch.exp(theta)
            params = {
                "ell": values[0],
                "w": values[1],
            }
            if not fixed_noise:
                params["sigma_f"] = values[2]
                params["sigma_d"] = values[3]
            try:
                posterior = _stationary_posterior(
                    observations,
                    params,
                    jitter=jitter,
                    fixed_noise=fixed_noise,
                )
                likelihood = (
                    joint_loo_loglik(posterior)
                    if objective == "loo"
                    else joint_log_marginal_likelihood(posterior)
                )
                if use_hyperpriors:
                    log_prior = (
                        stationary_ell_prior_distribution(observations, priors).log_prob(theta[0])
                        + torch.distributions.Normal(priors.m_w, priors.s_w).log_prob(theta[1])
                    )
                    if not fixed_noise:
                        log_prior = (
                            log_prior
                            + torch.distributions.Normal(priors.m_sf, priors.s_sf).log_prob(theta[2])
                            + torch.distributions.Normal(priors.m_sd, priors.s_sd).log_prob(theta[3])
                        )
                else:
                    log_prior = likelihood * 0.0
                score = likelihood + log_prior
            except torch.linalg.LinAlgError:
                score = theta.sum() * 0.0 - 1e20

            loss = -score
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_([theta], max_norm=100.0)
                optimizer.step()
                with torch.no_grad():
                    theta.clamp_(min=lower, max=upper)
            history.append(float(score.detach().cpu().item()))

        with torch.no_grad():
            values = torch.exp(theta)
            final_params = {
                "ell": values[0].clone(),
                "w": values[1].clone(),
            }
            if not fixed_noise:
                final_params["sigma_f"] = values[2].clone()
                final_params["sigma_d"] = values[3].clone()
            final_posterior = _stationary_posterior(
                observations,
                final_params,
                jitter=jitter,
                fixed_noise=fixed_noise,
            )
            final_likelihood = (
                joint_loo_loglik(final_posterior)
                if objective == "loo"
                else joint_log_marginal_likelihood(final_posterior)
            )
            if use_hyperpriors:
                final_log_prior = (
                    stationary_ell_prior_distribution(observations, priors).log_prob(theta[0])
                    + torch.distributions.Normal(priors.m_w, priors.s_w).log_prob(theta[1])
                )
                if not fixed_noise:
                    final_log_prior = (
                        final_log_prior
                        + torch.distributions.Normal(priors.m_sf, priors.s_sf).log_prob(theta[2])
                        + torch.distributions.Normal(priors.m_sd, priors.s_sd).log_prob(theta[3])
                    )
            else:
                final_log_prior = final_likelihood * 0.0
            final_score = final_likelihood + final_log_prior
            result = HyperparameterOptimizationResult(
                params=final_params,
                posterior=final_posterior,
                objective_value=float(final_score.cpu().item()),
                restart=restart,
                history=history,
            )
        if best is None or result.objective_value > best.objective_value:
            best = result

    assert best is not None
    return best


def optimize_derivative_hyperparameters(
    *,
    x_der: torch.Tensor,
    dy_der: torch.Tensor,
    objective: str = "lml",
    priors: HyperPriorConfig | None = None,
    steps: int = 250,
    learning_rate: float = 0.05,
    restarts: int = 3,
    seed: int = 0,
    jitter: float = 1e-6,
) -> HyperparameterOptimizationResult:
    """Find a stationary-kernel MAP estimate for derivative-only GPR."""
    if objective not in {"lml", "loo"}:
        raise ValueError("objective must be 'lml' or 'loo'.")
    if steps <= 0 or restarts <= 0:
        raise ValueError("steps and restarts must be positive.")

    priors = priors or HyperPriorConfig()
    x_der = x_der.reshape(-1)
    dy_der = dy_der.reshape(-1)
    dtype = x_der.dtype
    device = x_der.device
    x_span = max(float((x_der.max() - x_der.min()).item()), 1e-3)
    lower_ell = math.log(x_span / 100.0)
    upper_ell = math.log(x_span * 10.0)
    lower = torch.tensor([lower_ell, -6.0, -8.0], dtype=dtype, device=device)
    upper = torch.tensor([upper_ell, 8.0, 6.0], dtype=dtype, device=device)
    center = torch.tensor(
        [math.log(max(x_span / 3.0, 1e-3)), math.log(4.184), math.log(0.5)],
        dtype=dtype,
        device=device,
    )
    ell_prior = (
        torch.distributions.Uniform(lower[0], upper[0])
        if priors.s_ell is None
        else torch.distributions.Normal(priors.m_ell, priors.s_ell)
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    best: HyperparameterOptimizationResult | None = None

    for restart in range(restarts):
        initial = center.clone()
        if restart:
            initial = initial + 0.5 * torch.randn(
                initial.shape, dtype=dtype, device=device, generator=generator
            )
        theta = torch.nn.Parameter(torch.clamp(initial, min=lower, max=upper))
        optimizer = torch.optim.Adam([theta], lr=learning_rate)
        history: list[float] = []

        for _ in range(steps):
            optimizer.zero_grad()
            values = torch.exp(theta)
            try:
                posterior = build_derivative_gp(
                    x_der=x_der,
                    dy_der=dy_der,
                    ell=values[0],
                    w=values[1],
                    noise_deriv_diag=values[2].square()
                    * torch.ones(x_der.numel(), dtype=dtype, device=device),
                    jitter=jitter,
                )
                likelihood = (
                    derivative_loo_loglik(posterior)
                    if objective == "loo"
                    else derivative_log_marginal_likelihood(posterior)
                )
                log_prior = (
                    ell_prior.log_prob(theta[0])
                    + torch.distributions.Normal(priors.m_w, priors.s_w).log_prob(theta[1])
                    + torch.distributions.Normal(priors.m_sd, priors.s_sd).log_prob(theta[2])
                )
                score = likelihood + log_prior
            except torch.linalg.LinAlgError:
                score = theta.sum() * 0.0 - 1e20

            loss = -score
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_([theta], max_norm=100.0)
                optimizer.step()
                with torch.no_grad():
                    theta.clamp_(min=lower, max=upper)
            history.append(float(score.detach().cpu().item()))

        with torch.no_grad():
            values = torch.exp(theta)
            final_params = {
                "ell": values[0].clone(),
                "w": values[1].clone(),
                "sigma_d": values[2].clone(),
            }
            final_posterior = build_derivative_gp(
                x_der=x_der,
                dy_der=dy_der,
                ell=final_params["ell"],
                w=final_params["w"],
                noise_deriv_diag=final_params["sigma_d"].square()
                * torch.ones(x_der.numel(), dtype=dtype, device=device),
                jitter=jitter,
            )
            final_likelihood = (
                derivative_loo_loglik(final_posterior)
                if objective == "loo"
                else derivative_log_marginal_likelihood(final_posterior)
            )
            final_log_prior = (
                ell_prior.log_prob(theta[0])
                + torch.distributions.Normal(priors.m_w, priors.s_w).log_prob(theta[1])
                + torch.distributions.Normal(priors.m_sd, priors.s_sd).log_prob(theta[2])
            )
            final_score = final_likelihood + final_log_prior
            result = HyperparameterOptimizationResult(
                params=final_params,
                posterior=final_posterior,
                objective_value=float(final_score.cpu().item()),
                restart=restart,
                history=history,
            )
        if best is None or result.objective_value > best.objective_value:
            best = result

    assert best is not None
    return best
