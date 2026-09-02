#!/usr/bin/env python3
"""Plot full-data BayesWHAM and hierarchical-GP uncertainty against WHAM."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


K_B_KJ_MOL_K = 0.0083144621
TEMPERATURE_K = 303.15


def main() -> None:
    root = Path(__file__).resolve().parent
    uq_dir = root / "r9_full/uq"
    nuts_dir = uq_dir / "nuts"
    output = uq_dir / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    reference_path = Path.home() / "freeGP-v0.1.0/reference_data/wham.dat"
    gp_path = (
        Path.home()
        / "freeGP-v0.1.0/results/membrane-ablation-10x10-5replicates/loo"
        / "artifacts/cells/w25_f1p00.pt"
    )

    reference = np.loadtxt(reference_path, comments="#")
    reference_x = reference[:, 0]
    reference_f = reference[:, 1] - np.max(reference[:, 1])

    posterior = np.load(nuts_dir / "posterior_samples.npz")
    bayes_x = posterior["bin_centers_nm"]
    beta_f = posterior["beta_f_grouped"].reshape(-1, bayes_x.size)
    bayes_draws = beta_f * (K_B_KJ_MOL_K * TEMPERATURE_K)
    bayes_draws -= np.max(bayes_draws, axis=1, keepdims=True)
    bayes_mean = bayes_draws.mean(axis=0)
    bayes_std = bayes_draws.std(axis=0, ddof=1)
    bayes_lower, bayes_upper = np.quantile(bayes_draws, [0.025, 0.975], axis=0)

    artifact = torch.load(gp_path, map_location="cpu", weights_only=False)
    gp = artifact.get("canonical_predictive_summary") or artifact["predictive_summary"]
    gp_x = np.asarray(gp["x_test"], dtype=float)
    gp_mean = np.asarray(gp["mean"], dtype=float)
    gp_std = np.sqrt(np.asarray(gp["total_variance"], dtype=float))
    gp_mean -= np.max(gp_mean)
    gp_lower = gp_mean - 1.96 * gp_std
    gp_upper = gp_mean + 1.96 * gp_std

    reference_on_bayes = np.interp(bayes_x, reference_x, reference_f)
    reference_on_gp = np.interp(gp_x, reference_x, reference_f)
    metrics = {
        "alignment": "each BayesWHAM draw and each displayed mean maximum-aligned",
        "displayed_interval": "BayesWHAM pointwise 95% posterior quantiles",
        "bayeswham_draws": int(bayes_draws.shape[0]),
        "bayeswham_map_rmse_kj_mol": 0.2205978476084517,
        "bayeswham_posterior_mean_rmse_kj_mol": float(
            np.sqrt(np.mean((bayes_mean - reference_on_bayes) ** 2))
        ),
        "bayeswham_mean_pointwise_sd_kj_mol": float(np.mean(bayes_std)),
        "hierarchical_gp_mean_rmse_kj_mol": float(
            np.sqrt(np.mean((gp_mean - reference_on_gp) ** 2))
        ),
        "hierarchical_gp_mean_pointwise_sd_kj_mol": float(np.mean(gp_std)),
        "bayeswham_sampler_diagnostics": json.loads(
            (nuts_dir / "diagnostics.json").read_text()
        ),
    }
    (output / "uq_comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(
        output / "bayeswham_uq_profile.csv",
        np.column_stack((bayes_x, bayes_mean, bayes_std, bayes_lower, bayes_upper)),
        delimiter=",",
        header="coordinate_nm,mean_kj_mol,std_kj_mol,lower_95_kj_mol,upper_95_kj_mol",
        comments="",
    )
    np.savetxt(
        output / "hierarchical_gp_uq_profile.csv",
        np.column_stack((gp_x, gp_mean, gp_std, gp_lower, gp_upper)),
        delimiter=",",
        header="coordinate_nm,mean_kj_mol,std_kj_mol,lower_95_kj_mol,upper_95_kj_mol",
        comments="",
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
    ax.plot(gp_x, gp_mean, color="#0072B2", linewidth=1.7, label="Hierarchical GP mean")
    ax.fill_between(
        bayes_x,
        bayes_lower,
        bayes_upper,
        color="#D55E00",
        alpha=0.30,
        linewidth=0,
        label="BayesWHAM 95% interval",
    )
    ax.plot(
        bayes_x,
        bayes_mean,
        color="#D55E00",
        linewidth=1.7,
        linestyle="--",
        label="BayesWHAM posterior mean",
    )
    ax.plot(reference_x, reference_f, color="black", linewidth=1.8, label="WHAM reference")
    ax.set_xlabel("Membrane--peptide distance [nm]")
    ax.set_ylabel("Free Energy [kJ/mol]")
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "bayeswham_hierarchical_gp_uq_vs_wham.pdf")
    fig.savefig(output / "bayeswham_hierarchical_gp_uq_vs_wham.png", dpi=300)
    plt.close(fig)

    print(json.dumps({key: value for key, value in metrics.items() if key != "bayeswham_sampler_diagnostics"}, indent=2))
    print(f"Wrote UQ comparison to {output}")


if __name__ == "__main__":
    main()
