"""Pyro HMC-NUTS wrappers for the joint GP model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import pyro.distributions as dist

from .gp import build_joint_gp, joint_log_marginal_likelihood, joint_loo_loglik, predict_function
from .preprocess import JointObservations


@dataclass(frozen=True)
class HyperPriorConfig:
    # Priors are placed in log-parameter space and exponentiated before use.
    # Keep ell weakly informative so NUTS can explore a broad range of
    # physically plausible length scales instead of being pinned near exp(3).
    m_ell: float = math.log(4.0)
    s_ell: float = 1.0
    m_w: float = 1.0
    s_w: float = 0.5
    m_sf: float = 0.5
    s_sf: float = 2.0
    m_sd: float = 0.5
    s_sd: float = 2.0


@dataclass(frozen=True)
class NUTSConfig:
    num_samples: int = 1000
    warmup_steps: int = 2000
    num_chains: int = 1
    target_accept_prob: float = 0.8
    jitter: float = 1e-6
    objective: str = "lml"


def _posterior_for_theta(
    theta_exp: torch.Tensor,
    observations: JointObservations,
    *,
    jitter: float,
):
    ell = torch.as_tensor(theta_exp[0], dtype=torch.float64)
    w = torch.as_tensor(theta_exp[1], dtype=torch.float64)
    sf = torch.as_tensor(theta_exp[2], dtype=torch.float64)
    sd = torch.as_tensor(theta_exp[3], dtype=torch.float64)

    dtype = observations.x_obs.dtype
    device = observations.x_obs.device

    function_noise = (sf**2) * torch.eye(len(observations.x_obs), dtype=dtype, device=device)
    derivative_noise = (sd**2) * torch.ones(
        (len(observations.x_der),),
        dtype=dtype,
        device=device,
    )

    return build_joint_gp(
        x_func=observations.x_obs,
        y_func=observations.y_obs,
        x_der=observations.x_der,
        dy_der=observations.dy_der,
        ell=ell,
        w=w,
        noise_func_cov=function_noise,
        noise_deriv_diag=derivative_noise,
        H_func=observations.H_obs,
        jitter=jitter,
    )


def make_pyro_model(
    observations: JointObservations,
    *,
    priors: HyperPriorConfig | None = None,
    config: NUTSConfig | None = None,
):
    """Return a Pyro model closure for HMC-NUTS."""
    priors = priors or HyperPriorConfig()
    config = config or NUTSConfig()

    import pyro
    import pyro.distributions as dist

    def model():
        theta_ell = pyro.sample("theta_ell", dist.Normal(priors.m_ell, priors.s_ell))
        theta_w = pyro.sample("theta_w", dist.Normal(priors.m_w, priors.s_w))
        theta_sf = pyro.sample("theta_sf", dist.Normal(priors.m_sf, priors.s_sf))
        theta_sd = pyro.sample("theta_sd", dist.Normal(priors.m_sd, priors.s_sd))

        theta = torch.stack([theta_ell, theta_w, theta_sf, theta_sd])
        theta_exp = torch.exp(theta)
        posterior = _posterior_for_theta(theta_exp, observations, jitter=config.jitter)

        if config.objective == "loo":
            likelihood = joint_loo_loglik(posterior)
        else:
            likelihood = joint_log_marginal_likelihood(posterior)

        likelihood = torch.where(
            torch.isfinite(likelihood),
            likelihood,
            torch.tensor(-1e10, dtype=likelihood.dtype, device=likelihood.device),
        )
        pyro.factor("likelihood", likelihood)

    return model


def evaluate_log_posterior(
    theta_log: torch.Tensor,
    observations: JointObservations,
    *,
    priors: HyperPriorConfig | None = None,
    config: NUTSConfig | None = None,
) -> torch.Tensor:
    """Evaluate the sampled log posterior at one log-parameter vector."""
    priors = priors or HyperPriorConfig()
    config = config or NUTSConfig()

    theta_ell = theta_log[0]
    theta_w = theta_log[1]
    theta_sf = theta_log[2]
    theta_sd = theta_log[3]

    log_prior = (
        dist.Normal(priors.m_ell, priors.s_ell).log_prob(theta_ell)
        + dist.Normal(priors.m_w, priors.s_w).log_prob(theta_w)
        + dist.Normal(priors.m_sf, priors.s_sf).log_prob(theta_sf)
        + dist.Normal(priors.m_sd, priors.s_sd).log_prob(theta_sd)
    )

    theta_exp = torch.exp(theta_log)
    posterior = _posterior_for_theta(theta_exp, observations, jitter=config.jitter)
    if config.objective == "loo":
        likelihood = joint_loo_loglik(posterior)
    else:
        likelihood = joint_log_marginal_likelihood(posterior)

    value = log_prior + likelihood
    return torch.where(
        torch.isfinite(value),
        value,
        torch.tensor(float("-inf"), dtype=value.dtype, device=value.device),
    )


def run_hmc_nuts(
    observations: JointObservations,
    *,
    priors: HyperPriorConfig | None = None,
    config: NUTSConfig | None = None,
):
    """Run Pyro NUTS and return the fitted MCMC object and samples."""
    config = config or NUTSConfig()

    import pyro
    from pyro.infer.mcmc import MCMC, NUTS

    pyro.clear_param_store()
    model = make_pyro_model(observations, priors=priors, config=config)
    nuts_kernel = NUTS(
        model,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        target_accept_prob=config.target_accept_prob,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples=config.num_samples,
        warmup_steps=config.warmup_steps,
        num_chains=config.num_chains,
    )
    mcmc.run()
    return mcmc, mcmc.get_samples()


def maximum_a_posteriori_prediction(
    observations: JointObservations,
    samples: dict[str, torch.Tensor],
    x_test: torch.Tensor,
    *,
    priors: HyperPriorConfig | None = None,
    config: NUTSConfig | None = None,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose the highest posterior sample from the chain and predict with it."""
    priors = priors or HyperPriorConfig()
    config = config or NUTSConfig()

    chain_log = torch.stack(
        [
            samples["theta_ell"],
            samples["theta_w"],
            samples["theta_sf"],
            samples["theta_sd"],
        ],
        dim=-1,
    )

    scores = torch.stack(
        [
            evaluate_log_posterior(theta_log, observations, priors=priors, config=config)
            for theta_log in chain_log
        ],
        dim=0,
    )
    best_idx = int(torch.argmax(scores).item())
    best_theta_log = chain_log[best_idx]
    best_theta = torch.exp(best_theta_log)
    posterior = _posterior_for_theta(best_theta, observations, jitter=config.jitter)
    pred_mean, pred_cov = predict_function(posterior, x_test)
    return best_idx, best_theta, scores[best_idx], pred_mean, pred_cov


def sample_posterior_functions(
    observations: JointObservations,
    samples: dict[str, torch.Tensor],
    x_test: torch.Tensor,
    *,
    n_draws: int = 250,
    jitter: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw function samples from the hyperposterior."""
    device = x_test.device
    dtype = x_test.dtype
    chain = torch.stack(
        [
            samples["theta_ell"],
            samples["theta_w"],
            samples["theta_sf"],
            samples["theta_sd"],
        ],
        dim=-1,
    )
    chain = torch.exp(chain).to(device=device, dtype=dtype)

    pred_means = []
    function_draws = []
    for _ in range(n_draws):
        idx = torch.randint(0, chain.shape[0], (1,), device=device)
        theta = chain[idx].squeeze(0)
        posterior = _posterior_for_theta(theta, observations, jitter=jitter)
        pred_mean, pred_cov = predict_function(posterior, x_test)
        chol = torch.linalg.cholesky(
            pred_cov + 1e-10 * torch.eye(len(x_test), dtype=dtype, device=device)
        )
        eps = torch.randn(len(x_test), dtype=dtype, device=device)
        pred_means.append(pred_mean)
        function_draws.append(pred_mean + chol @ eps)

    return torch.stack(pred_means, dim=0), torch.stack(function_draws, dim=0)
