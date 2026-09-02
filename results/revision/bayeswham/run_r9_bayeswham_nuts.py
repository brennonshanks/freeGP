#!/usr/bin/env python3
"""Sample the BayesWHAM posterior for full R9 effective counts using NUTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyro
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_value
import torch
from torch.distributions import biject_to, constraints


K_B_KJ_MOL_K = 0.0083144621


class BayesWhamModel:
    """Pickle-safe BayesWHAM posterior callable for parallel Pyro chains."""

    def __init__(self, c_il: torch.Tensor, n_i: torch.Tensor, m_l: torch.Tensor):
        self.c_il = c_il
        self.n_i = n_i
        self.m_l = m_l

    def __call__(self) -> None:
        p = pyro.sample(
            "p",
            pyro.distributions.Dirichlet(torch.ones_like(self.m_l)),
        )
        f_i = torch.reciprocal(self.c_il @ p)
        log_likelihood = torch.dot(self.n_i, torch.log(f_i)) + torch.dot(
            self.m_l, torch.log(p)
        )
        pyro.factor("bayeswham_likelihood", log_likelihood)


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=root / "r9_full/effective_counts_ar1")
    p.add_argument(
        "--map-output",
        type=Path,
        default=root / "r9_full/map_validation/effective_counts",
        help="Directory containing p_MAP.txt; the full-data MAP is a valid initializer for ablation cells.",
    )
    p.add_argument("--output", type=Path, default=root / "r9_full/uq/nuts")
    p.add_argument("--temperature-k", type=float, default=303.15)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument(
        "--cell-map",
        action="store_true",
        help="Compute a WHAM MAP initializer from this input instead of loading the full-data MAP.",
    )
    p.add_argument(
        "--init-jitter",
        type=float,
        default=0.0,
        help="Log-probability SD for independent chain initializations around the MAP.",
    )
    return p


def wham_map(c_il: torch.Tensor, n_i: torch.Tensor, m_l: torch.Tensor) -> torch.Tensor:
    """Uniform-prior BayesWHAM MAP from fixed-point WHAM iteration."""
    occupied = m_l > 0
    c = c_il[:, occupied]
    m = m_l[occupied]
    p = torch.full_like(m, 1.0 / m.numel())
    f = torch.reciprocal(c @ p)
    for _ in range(100_000):
        old = p
        p = m / ((n_i * f) @ c)
        p = p / p.sum()
        f = torch.reciprocal(c @ p)
        # This MAP is used only to initialize NUTS. Sparse/disconnected window
        # selections can plateau near machine precision before reaching the
        # stricter tolerance used for a reported point estimate.
        if torch.max(torch.abs(p - old)) <= 1e-8:
            break
    else:
        raise RuntimeError("Cell-specific WHAM MAP did not converge")
    result = torch.full_like(m_l, 1e-12)
    result[occupied] = p
    return result / result.sum()


def main() -> None:
    args = parser().parse_args()
    input_dir = args.input.expanduser().resolve()
    map_dir = args.map_output.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    centers = torch.as_tensor(np.loadtxt(input_dir / "hist_binCenters.txt"), dtype=torch.float64)
    biases = torch.as_tensor(np.loadtxt(input_dir / "harmonic_biases.txt"), dtype=torch.float64)
    counts = torch.stack(
        [
            torch.as_tensor(np.loadtxt(input_dir / "hist" / f"hist_{i}.txt"), dtype=torch.float64)
            for i in range(1, biases.shape[0] + 1)
        ]
    )
    beta = 1.0 / (K_B_KJ_MOL_K * args.temperature_k)
    umbrella_centers = biases[:, 1]
    force_constants = biases[:, 2]
    bias_energy = 0.5 * force_constants[:, None] * (
        centers[None, :] - umbrella_centers[:, None]
    ) ** 2
    c_il = torch.exp(-beta * bias_energy)
    n_i = counts.sum(dim=1)
    m_l = counts.sum(dim=0)
    if args.cell_map:
        p_map = wham_map(c_il, n_i, m_l)
        np.savetxt(output / "p_MAP.txt", p_map.numpy()[None], fmt="%.16g")
    else:
        p_map = torch.as_tensor(np.loadtxt(map_dir / "p_MAP.txt"), dtype=torch.float64)

    model = BayesWhamModel(c_il, n_i, m_l)

    pyro.set_rng_seed(args.seed)
    pyro.enable_validation(True)
    kernel = NUTS(
        model,
        target_accept_prob=args.target_accept,
        max_tree_depth=10,
        init_strategy=init_to_value(values={"p": p_map}),
    )
    initial_params = None
    if args.init_jitter > 0.0:
        generator = torch.Generator().manual_seed(args.seed + 17)
        noise = torch.randn(
            (args.chains, p_map.numel()), dtype=p_map.dtype, generator=generator
        ) * args.init_jitter
        chain_probabilities = torch.softmax(torch.log(p_map)[None, :] + noise, dim=-1)
        initial_params = {
            "p": biject_to(constraints.simplex).inv(chain_probabilities)
        }
    mcmc = MCMC(
        kernel,
        num_samples=args.samples,
        warmup_steps=args.warmup,
        num_chains=args.chains,
        mp_context="spawn",
        initial_params=initial_params,
    )
    mcmc.run()
    grouped = mcmc.get_samples(group_by_chain=True)["p"].detach().cpu().numpy()
    flat = grouped.reshape(-1, grouped.shape[-1])
    bin_widths = np.diff(np.loadtxt(input_dir / "hist_binEdges.txt"))
    beta_f = -np.log(flat / bin_widths[None, :])
    beta_f -= beta_f.mean(axis=1, keepdims=True)
    beta_f_grouped = beta_f.reshape(grouped.shape)
    np.savez_compressed(
        output / "posterior_samples.npz",
        p_grouped=grouped,
        beta_f_grouped=beta_f_grouped,
        bin_centers_nm=centers.numpy(),
    )

    diagnostics = mcmc.diagnostics()
    p_diag = diagnostics["p"]
    r_hat = np.asarray(p_diag["r_hat"].detach().cpu())
    n_eff = np.asarray(p_diag["n_eff"].detach().cpu())
    summary = {
        "probability_model": "BayesWHAM with uniform Dirichlet(1) simplex prior",
        "sampler": "Pyro NUTS",
        "warmup_steps_per_chain": args.warmup,
        "samples_per_chain": args.samples,
        "num_chains": args.chains,
        "target_accept_probability": args.target_accept,
        "seed": args.seed,
        "initializer": "cell-specific WHAM MAP" if args.cell_map else "provided MAP",
        "initial_log_probability_jitter_sd": args.init_jitter,
        "max_r_hat": float(np.nanmax(r_hat)),
        "median_r_hat": float(np.nanmedian(r_hat)),
        "min_effective_sample_size": float(np.nanmin(n_eff)),
        "median_effective_sample_size": float(np.nanmedian(n_eff)),
        "divergences": {key: value for key, value in diagnostics.get("divergences", {}).items()},
        "acceptance_rate": diagnostics.get("acceptance rate", {}),
    }
    (output / "diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
