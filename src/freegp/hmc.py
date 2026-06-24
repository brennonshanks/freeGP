"""Pyro HMC-NUTS wrappers for the stationary and Gibbs joint GP models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import pyro.distributions as dist

from .gp import (
    GibbsKernelConfig,
    build_joint_gp,
    build_joint_gp_gibbs,
    joint_log_marginal_likelihood,
    joint_loo_loglik,
    predict_function,
)
from .preprocess import JointObservations


@dataclass(frozen=True)
class HyperPriorConfig:
    # Stationary kernel priors in log-parameter space.
    m_ell: float = math.log(4.0)
    s_ell: float | None = 1.0
    m_w: float = 1.0
    s_w: float = 0.5
    m_sf: float = 0.5
    s_sf: float = 2.0
    m_sd: float = 0.5
    s_sd: float = 2.0

    # Full Gibbs kernel priors. A few centers are data-driven when set to None.
    m_a0: float | None = None
    s_a0: float = 1.0
    m_a1: float = 0.0
    s_a1: float = 0.5
    m_b: float = 0.0
    s_b: float = 1.0
    m_c: float | None = None
    s_c_scale: float = 0.5
    m_log_length_w: float | None = None
    s_log_length_w: float = 1.0
    m_log_s: float = 0.0
    s_log_s: float = 1.0
    m_u: float | None = None
    s_u_scale: float = 0.5
    m_log_width_w: float | None = None
    s_log_width_w: float = 1.0


@dataclass(frozen=True)
class NUTSConfig:
    num_samples: int = 1000
    warmup_steps: int = 2000
    num_chains: int = 1
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    seed: int | None = None
    jitter: float = 1e-6
    objective: str = "lml"
    kernel: str = "stationary"
    length_model: str = "exp_linear_bump"
    width_model: str = "tanh_decay"
    fixed_noise: bool = False  # If True, noise is taken from JointObservations (not sampled)


@dataclass(frozen=True)
class HMCChainDiagnostics:
    step_size: float
    mean_accept_prob: float
    accept_count: int
    divergence_count: int
    sample_std_by_name: dict[str, float]
    mean_sample_std: float
    max_sample_std: float
    min_sample_std: float
    poor_acceptance: bool
    looks_stuck: bool


def _observation_domain(observations: JointObservations) -> tuple[float, float, float]:
    x_all = torch.cat([observations.x_obs.reshape(-1), observations.x_der.reshape(-1)])
    x_min = float(x_all.min().item())
    x_max = float(x_all.max().item())
    x_mid = 0.5 * (x_min + x_max)
    x_span = max(x_max - x_min, 1e-3)
    return x_min, x_mid, x_span


def stationary_log_ell_bounds(
    observations: JointObservations,
) -> tuple[float, float]:
    """Numerical bounds used for a flat prior on log length scale."""
    _, _, x_span = _observation_domain(observations)
    return math.log(x_span / 100.0), math.log(x_span * 10.0)


def stationary_ell_prior_distribution(
    observations: JointObservations,
    priors: HyperPriorConfig,
):
    """Return the configured prior distribution for log length scale."""
    if priors.s_ell is None:
        lower, upper = stationary_log_ell_bounds(observations)
        return dist.Uniform(lower, upper)
    return dist.Normal(priors.m_ell, priors.s_ell)


def _sample_site_names(config: NUTSConfig) -> list[str]:
    if config.kernel == "stationary":
        names = ["theta_ell", "theta_w"]
        if not config.fixed_noise:
            names.extend(["theta_sf", "theta_sd"])
        return names

    names: list[str] = []
    if config.length_model == "exp_linear_bump":
        names.extend(["a0", "a1", "b", "c", "theta_length_w"])
    elif config.length_model == "constant":
        names.append("a0")
    else:
        raise ValueError(f"Unsupported Gibbs length model: {config.length_model}")

    if config.width_model == "tanh_decay":
        names.extend(["theta_s", "u", "theta_width_w"])
    elif config.width_model == "constant":
        names.append("theta_s")
    else:
        raise ValueError(f"Unsupported Gibbs width model: {config.width_model}")

    if not config.fixed_noise:
        names.extend(["theta_sf", "theta_sd"])
    return names


def _display_sample_labels(config: NUTSConfig) -> list[str]:
    if config.kernel == "stationary":
        labels = ["ell", "w"]
        if not config.fixed_noise:
            labels.extend(["sigma_f", "sigma_d"])
        return labels

    labels: list[str] = []
    if config.length_model == "exp_linear_bump":
        labels.extend(["a0", "a1", "b", "c", "length_w"])
    elif config.length_model == "constant":
        labels.append("ell0")

    if config.width_model == "tanh_decay":
        labels.extend(["s", "u", "width_w"])
    elif config.width_model == "constant":
        labels.append("s")

    if not config.fixed_noise:
        labels.extend(["sigma_f", "sigma_d"])
    return labels


def _display_sample_values(sample_state: dict[str, torch.Tensor], config: NUTSConfig) -> list[torch.Tensor]:
    if config.kernel == "stationary":
        values = [
            torch.exp(sample_state["theta_ell"]),
            torch.exp(sample_state["theta_w"]),
        ]
        if not config.fixed_noise:
            values.extend([
                torch.exp(sample_state["theta_sf"]),
                torch.exp(sample_state["theta_sd"]),
            ])
        return values

    values: list[torch.Tensor] = []
    if config.length_model == "exp_linear_bump":
        values.extend(
            [
                sample_state["a0"],
                sample_state["a1"],
                sample_state["b"],
                sample_state["c"],
                torch.exp(sample_state["theta_length_w"]),
            ]
        )
    elif config.length_model == "constant":
        values.append(torch.exp(sample_state["a0"]))

    if config.width_model == "tanh_decay":
        values.extend(
            [
                torch.exp(sample_state["theta_s"]),
                sample_state["u"],
                torch.exp(sample_state["theta_width_w"]),
            ]
        )
    elif config.width_model == "constant":
        values.append(torch.exp(sample_state["theta_s"]))

    if not config.fixed_noise:
        values.extend([torch.exp(sample_state["theta_sf"]), torch.exp(sample_state["theta_sd"])])
    return values


def display_samples_for_diagnostics(
    samples: dict[str, torch.Tensor],
    *,
    config: NUTSConfig | None = None,
) -> tuple[torch.Tensor, list[str]]:
    config = config or NUTSConfig()
    site_names = _sample_site_names(config)
    rows = []
    n_samples = samples[site_names[0]].shape[0]
    for idx in range(n_samples):
        sample_state = {name: samples[name][idx] for name in site_names}
        rows.append(torch.stack(_display_sample_values(sample_state, config)))
    return torch.stack(rows, dim=0), _display_sample_labels(config)


def summarize_chain_diagnostics(
    mcmc,
    samples: dict[str, torch.Tensor],
    *,
    config: NUTSConfig | None = None,
) -> HMCChainDiagnostics:
    config = config or NUTSConfig()
    chain, labels = display_samples_for_diagnostics(samples, config=config)
    stds = chain.std(dim=0, unbiased=False)
    sample_std_by_name = {
        label: float(value.item())
        for label, value in zip(labels, stds)
    }

    step_size = float(getattr(mcmc.kernel, "step_size", float("nan")))
    mean_accept_prob = float(getattr(mcmc.kernel, "_mean_accept_prob", float("nan")))
    accept_count = int(getattr(mcmc.kernel, "_accept_cnt", 0))
    divergences = getattr(mcmc.kernel, "_divergences", [])
    divergence_count = int(len(divergences)) if divergences is not None else 0

    mean_sample_std = float(stds.mean().item())
    max_sample_std = float(stds.max().item())
    min_sample_std = float(stds.min().item())
    poor_acceptance = bool(np.isfinite(mean_accept_prob) and mean_accept_prob < 0.05)
    looks_stuck = bool(max_sample_std < 1e-6)

    return HMCChainDiagnostics(
        step_size=step_size,
        mean_accept_prob=mean_accept_prob,
        accept_count=accept_count,
        divergence_count=divergence_count,
        sample_std_by_name=sample_std_by_name,
        mean_sample_std=mean_sample_std,
        max_sample_std=max_sample_std,
        min_sample_std=min_sample_std,
        poor_acceptance=poor_acceptance,
        looks_stuck=looks_stuck,
    )


def _gibbs_prior_centers(
    observations: JointObservations,
    priors: HyperPriorConfig,
) -> dict[str, float]:
    _, x_mid, x_span = _observation_domain(observations)
    return {
        "a0": priors.m_a0 if priors.m_a0 is not None else math.log(max(x_span / 4.0, 1e-3)),
        "c": priors.m_c if priors.m_c is not None else x_mid,
        "log_length_w": (
            priors.m_log_length_w
            if priors.m_log_length_w is not None
            else math.log(max(x_span / 4.0, 1e-3))
        ),
        "u": priors.m_u if priors.m_u is not None else x_mid,
        "log_width_w": (
            priors.m_log_width_w
            if priors.m_log_width_w is not None
            else math.log(max(x_span / 4.0, 1e-3))
        ),
        "x_span": x_span,
    }


def _sample_kernel_parameters(
    observations: JointObservations,
    *,
    priors: HyperPriorConfig,
    config: NUTSConfig,
) -> dict[str, torch.Tensor]:
    import pyro

    if config.kernel == "stationary":
        theta_ell = pyro.sample(
            "theta_ell", stationary_ell_prior_distribution(observations, priors)
        )
        theta_w = pyro.sample("theta_w", dist.Normal(priors.m_w, priors.s_w))
        if config.fixed_noise:
            return {
                "ell": torch.exp(theta_ell),
                "w": torch.exp(theta_w),
            }
        theta_sf = pyro.sample("theta_sf", dist.Normal(priors.m_sf, priors.s_sf))
        theta_sd = pyro.sample("theta_sd", dist.Normal(priors.m_sd, priors.s_sd))
        return {
            "ell": torch.exp(theta_ell),
            "w": torch.exp(theta_w),
            "sigma_f": torch.exp(theta_sf),
            "sigma_d": torch.exp(theta_sd),
        }

    centers = _gibbs_prior_centers(observations, priors)
    params: dict[str, torch.Tensor] = {}
    if config.length_model == "exp_linear_bump":
        params["a0"] = pyro.sample("a0", dist.Normal(centers["a0"], priors.s_a0))
        params["a1"] = pyro.sample("a1", dist.Normal(priors.m_a1, priors.s_a1))
        params["b"] = pyro.sample("b", dist.Normal(priors.m_b, priors.s_b))
        params["c"] = pyro.sample("c", dist.Normal(centers["c"], priors.s_c_scale * centers["x_span"]))
        params["length_w"] = torch.exp(
            pyro.sample(
                "theta_length_w",
                dist.Normal(centers["log_length_w"], priors.s_log_length_w),
            )
        )
    elif config.length_model == "constant":
        params["a0"] = pyro.sample("a0", dist.Normal(centers["a0"], priors.s_a0))
        params["a1"] = torch.zeros((), dtype=observations.x_obs.dtype, device=observations.x_obs.device)
        params["b"] = torch.zeros((), dtype=observations.x_obs.dtype, device=observations.x_obs.device)
        params["c"] = torch.tensor(centers["c"], dtype=observations.x_obs.dtype, device=observations.x_obs.device)
        params["length_w"] = torch.tensor(
            math.exp(centers["log_length_w"]),
            dtype=observations.x_obs.dtype,
            device=observations.x_obs.device,
        )
    else:
        raise ValueError(f"Unsupported Gibbs length model: {config.length_model}")

    if config.width_model == "tanh_decay":
        params["s"] = torch.exp(pyro.sample("theta_s", dist.Normal(priors.m_log_s, priors.s_log_s)))
        params["u"] = pyro.sample("u", dist.Normal(centers["u"], priors.s_u_scale * centers["x_span"]))
        params["width_w"] = torch.exp(
            pyro.sample(
                "theta_width_w",
                dist.Normal(centers["log_width_w"], priors.s_log_width_w),
            )
        )
    elif config.width_model == "constant":
        params["s"] = torch.exp(pyro.sample("theta_s", dist.Normal(priors.m_log_s, priors.s_log_s)))
        params["u"] = torch.tensor(centers["u"], dtype=observations.x_obs.dtype, device=observations.x_obs.device)
        params["width_w"] = torch.tensor(
            math.exp(centers["log_width_w"]),
            dtype=observations.x_obs.dtype,
            device=observations.x_obs.device,
        )
    else:
        raise ValueError(f"Unsupported Gibbs width model: {config.width_model}")

    if not config.fixed_noise:
        params["sigma_f"] = torch.exp(pyro.sample("theta_sf", dist.Normal(priors.m_sf, priors.s_sf)))
        params["sigma_d"] = torch.exp(pyro.sample("theta_sd", dist.Normal(priors.m_sd, priors.s_sd)))
    return params


def _state_to_parameters_and_log_prior(
    sample_state: dict[str, torch.Tensor],
    observations: JointObservations,
    *,
    priors: HyperPriorConfig,
    config: NUTSConfig,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    dtype = observations.x_obs.dtype
    device = observations.x_obs.device

    if config.kernel == "stationary":
        log_prior = (
            stationary_ell_prior_distribution(observations, priors).log_prob(
                sample_state["theta_ell"]
            )
            + dist.Normal(priors.m_w, priors.s_w).log_prob(sample_state["theta_w"])
        )
        params: dict[str, torch.Tensor] = {
            "ell": torch.exp(sample_state["theta_ell"]),
            "w": torch.exp(sample_state["theta_w"]),
        }
        if not config.fixed_noise:
            log_prior = (
                log_prior
                + dist.Normal(priors.m_sf, priors.s_sf).log_prob(sample_state["theta_sf"])
                + dist.Normal(priors.m_sd, priors.s_sd).log_prob(sample_state["theta_sd"])
            )
            params["sigma_f"] = torch.exp(sample_state["theta_sf"])
            params["sigma_d"] = torch.exp(sample_state["theta_sd"])
        return params, log_prior

    centers = _gibbs_prior_centers(observations, priors)
    zero = torch.zeros((), dtype=dtype, device=device)
    log_prior = zero
    params: dict[str, torch.Tensor] = {}

    if config.length_model == "exp_linear_bump":
        log_prior = log_prior + dist.Normal(centers["a0"], priors.s_a0).log_prob(sample_state["a0"])
        log_prior = log_prior + dist.Normal(priors.m_a1, priors.s_a1).log_prob(sample_state["a1"])
        log_prior = log_prior + dist.Normal(priors.m_b, priors.s_b).log_prob(sample_state["b"])
        log_prior = log_prior + dist.Normal(centers["c"], priors.s_c_scale * centers["x_span"]).log_prob(
            sample_state["c"]
        )
        log_prior = log_prior + dist.Normal(
            centers["log_length_w"], priors.s_log_length_w
        ).log_prob(sample_state["theta_length_w"])
        params["a0"] = sample_state["a0"]
        params["a1"] = sample_state["a1"]
        params["b"] = sample_state["b"]
        params["c"] = sample_state["c"]
        params["length_w"] = torch.exp(sample_state["theta_length_w"])
    elif config.length_model == "constant":
        log_prior = log_prior + dist.Normal(centers["a0"], priors.s_a0).log_prob(sample_state["a0"])
        params["a0"] = sample_state["a0"]
        params["a1"] = zero
        params["b"] = zero
        params["c"] = torch.tensor(centers["c"], dtype=dtype, device=device)
        params["length_w"] = torch.tensor(math.exp(centers["log_length_w"]), dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported Gibbs length model: {config.length_model}")

    if config.width_model == "tanh_decay":
        log_prior = log_prior + dist.Normal(priors.m_log_s, priors.s_log_s).log_prob(sample_state["theta_s"])
        log_prior = log_prior + dist.Normal(centers["u"], priors.s_u_scale * centers["x_span"]).log_prob(
            sample_state["u"]
        )
        log_prior = log_prior + dist.Normal(
            centers["log_width_w"], priors.s_log_width_w
        ).log_prob(sample_state["theta_width_w"])
        params["s"] = torch.exp(sample_state["theta_s"])
        params["u"] = sample_state["u"]
        params["width_w"] = torch.exp(sample_state["theta_width_w"])
    elif config.width_model == "constant":
        log_prior = log_prior + dist.Normal(priors.m_log_s, priors.s_log_s).log_prob(sample_state["theta_s"])
        params["s"] = torch.exp(sample_state["theta_s"])
        params["u"] = torch.tensor(centers["u"], dtype=dtype, device=device)
        params["width_w"] = torch.tensor(math.exp(centers["log_width_w"]), dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported Gibbs width model: {config.width_model}")

    if not config.fixed_noise:
        log_prior = log_prior + dist.Normal(priors.m_sf, priors.s_sf).log_prob(sample_state["theta_sf"])
        log_prior = log_prior + dist.Normal(priors.m_sd, priors.s_sd).log_prob(sample_state["theta_sd"])
        params["sigma_f"] = torch.exp(sample_state["theta_sf"])
        params["sigma_d"] = torch.exp(sample_state["theta_sd"])
    return params, log_prior


def _posterior_for_parameters(
    params: dict[str, torch.Tensor],
    observations: JointObservations,
    *,
    config: NUTSConfig,
):
    dtype = observations.x_obs.dtype
    device = observations.x_obs.device

    if config.fixed_noise:
        function_noise = observations.noise_func_cov
        derivative_noise = observations.noise_deriv_diag
    else:
        function_noise = (params["sigma_f"] ** 2) * torch.eye(len(observations.x_obs), dtype=dtype, device=device)
        derivative_noise = (params["sigma_d"] ** 2) * torch.ones(
            (len(observations.x_der),),
            dtype=dtype,
            device=device,
        )

    if config.kernel == "stationary":
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
            jitter=config.jitter,
        )

    gibbs_config = GibbsKernelConfig(
        length_model=config.length_model,
        width_model=config.width_model,
    )
    return build_joint_gp_gibbs(
        x_func=observations.x_obs,
        y_func=observations.y_obs,
        x_der=observations.x_der,
        dy_der=observations.dy_der,
        a0=params["a0"],
        a1=params["a1"],
        b=params["b"],
        c=params["c"],
        length_w=params["length_w"],
        s=params["s"],
        u=params["u"],
        width_w=params["width_w"],
        noise_func_cov=function_noise,
        noise_deriv_diag=derivative_noise,
        H_func=observations.H_obs,
        config=gibbs_config,
        jitter=config.jitter,
    )


class _PyroModel:
    """Picklable Pyro model for HMC-NUTS (required for spawn-based multiprocessing)."""

    def __init__(
        self,
        observations: JointObservations,
        priors: HyperPriorConfig,
        config: NUTSConfig,
    ) -> None:
        self.observations = observations
        self.priors = priors
        self.config = config

    def __call__(self) -> None:
        import pyro

        params = _sample_kernel_parameters(self.observations, priors=self.priors, config=self.config)
        posterior = _posterior_for_parameters(params, self.observations, config=self.config)
        if self.config.objective == "loo":
            likelihood = joint_loo_loglik(posterior)
        else:
            likelihood = joint_log_marginal_likelihood(posterior)

        likelihood = torch.where(
            torch.isfinite(likelihood),
            likelihood,
            torch.tensor(-1e10, dtype=likelihood.dtype, device=likelihood.device),
        )
        pyro.factor("likelihood", likelihood)


def make_pyro_model(
    observations: JointObservations,
    *,
    priors: HyperPriorConfig | None = None,
    config: NUTSConfig | None = None,
) -> _PyroModel:
    """Return a picklable Pyro model for HMC-NUTS."""
    return _PyroModel(
        observations,
        priors or HyperPriorConfig(),
        config or NUTSConfig(),
    )


def evaluate_log_posterior(
    sample_state: dict[str, torch.Tensor],
    observations: JointObservations,
    *,
    priors: HyperPriorConfig | None = None,
    config: NUTSConfig | None = None,
) -> torch.Tensor:
    """Evaluate the sampled log posterior at one sampler state."""
    priors = priors or HyperPriorConfig()
    config = config or NUTSConfig()
    params, log_prior = _state_to_parameters_and_log_prior(
        sample_state,
        observations,
        priors=priors,
        config=config,
    )
    posterior = _posterior_for_parameters(params, observations, config=config)
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

    if config.seed is not None:
        seed = int(config.seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    model = make_pyro_model(observations, priors=priors, config=config)
    nuts_kernel = NUTS(
        model,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        target_accept_prob=config.target_accept_prob,
        max_tree_depth=config.max_tree_depth,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_samples=config.num_samples,
        warmup_steps=config.warmup_steps,
        num_chains=config.num_chains,
        mp_context="spawn" if config.num_chains > 1 else None,
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
) -> tuple[int, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose the highest posterior sample from the chain and predict with it."""
    priors = priors or HyperPriorConfig()
    config = config or NUTSConfig()
    site_names = _sample_site_names(config)

    scores = []
    params_by_index = []
    n_samples = samples[site_names[0]].shape[0]
    for idx in range(n_samples):
        sample_state = {name: samples[name][idx] for name in site_names}
        params, _ = _state_to_parameters_and_log_prior(
            sample_state,
            observations,
            priors=priors,
            config=config,
        )
        params_by_index.append(params)
        scores.append(
            evaluate_log_posterior(
                sample_state,
                observations,
                priors=priors,
                config=config,
            )
        )

    scores_t = torch.stack(scores, dim=0)
    best_idx = int(torch.argmax(scores_t).item())
    best_params = params_by_index[best_idx]
    posterior = _posterior_for_parameters(best_params, observations, config=config)
    pred_mean, pred_cov = predict_function(posterior, x_test)
    return best_idx, best_params, scores_t[best_idx], pred_mean, pred_cov


def sample_posterior_functions(
    observations: JointObservations,
    samples: dict[str, torch.Tensor],
    x_test: torch.Tensor,
    *,
    n_draws: int = 250,
    config: NUTSConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw function samples from the hyperposterior."""
    config = config or NUTSConfig()
    site_names = _sample_site_names(config)
    device = x_test.device
    dtype = x_test.dtype

    pred_means = []
    function_draws = []
    n_chain = samples[site_names[0]].shape[0]
    for _ in range(n_draws):
        idx = int(torch.randint(0, n_chain, (1,), device=device).item())
        sample_state = {name: samples[name][idx] for name in site_names}
        params, _ = _state_to_parameters_and_log_prior(
            sample_state,
            observations,
            priors=HyperPriorConfig(),
            config=config,
        )
        posterior = _posterior_for_parameters(params, observations, config=config)
        pred_mean, pred_cov = predict_function(posterior, x_test)
        chol = torch.linalg.cholesky(
            pred_cov + 1e-10 * torch.eye(len(x_test), dtype=dtype, device=device)
        )
        eps = torch.randn(len(x_test), dtype=dtype, device=device)
        pred_means.append(pred_mean)
        function_draws.append(pred_mean + chol @ eps)

    return torch.stack(pred_means, dim=0), torch.stack(function_draws, dim=0)
