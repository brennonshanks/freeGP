#!/usr/bin/env python3
"""Plot the three-condition BayesWHAM R9 ablation pilot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KT = 0.0083144621 * 303.15


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "r9_ablation_subset/figures"
    output.mkdir(parents=True, exist_ok=True)
    reference = np.loadtxt(Path.home() / "freeGP-v0.1.0/reference_data/wham.dat")
    reference_f = reference[:, 1] - np.max(reference[:, 1])
    cases = [
        ("Sparse", r"$W=3$, $T=10\%$", 3, root / "r9_ablation_subset/sparse_w03_f0p10/uq/nuts/posterior_samples.npz"),
        ("Intermediate", r"$W=10$, $T=25\%$", 10, root / "r9_ablation_subset/intermediate_w10_f0p25/uq/nuts/posterior_samples.npz"),
        ("Full", r"$W=25$, $T=100\%$", 25, root / "r9_full/uq/nuts/posterior_samples.npz"),
    ]

    summaries = []
    profiles = []
    for name, setting, window_count, path in cases:
        posterior = np.load(path)
        x = posterior["bin_centers_nm"]
        draws = posterior["beta_f_grouped"].reshape(-1, x.size) * KT
        draws -= np.max(draws, axis=1, keepdims=True)
        mean = draws.mean(axis=0)
        sd = draws.std(axis=0, ddof=1)
        lower, upper = np.quantile(draws, [0.025, 0.975], axis=0)
        truth = np.interp(x, reference[:, 0], reference_f)
        rmse = float(np.sqrt(np.mean((mean - truth) ** 2)))
        coverage = float(np.mean((truth >= lower) & (truth <= upper)))
        summaries.append(
            {
                "condition": name,
                "window_count": window_count,
                "rmse_kj_mol": rmse,
                "mean_posterior_sd_kj_mol": float(sd.mean()),
                "pointwise_95_coverage": coverage,
                "posterior_draws": int(draws.shape[0]),
            }
        )
        profiles.append((name, setting, x, mean, lower, upper))

    with (output / "pilot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9.5, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), constrained_layout=True)
    for ax, (name, setting, x, mean, lower, upper) in zip(axes.flat[:3], profiles):
        ax.fill_between(x, lower, upper, color="#D55E00", alpha=0.35, linewidth=0,
                        label="BayesWHAM 95% interval")
        ax.plot(x, mean, color="#D55E00", linewidth=1.5, linestyle="--",
                label="BayesWHAM posterior mean")
        ax.plot(reference[:, 0], reference_f, color="black", linewidth=1.5,
                label="Full-data WHAM reference")
        ax.set_title(f"{name}: {setting}")
        ax.set_xlabel("Membrane--peptide distance [nm]")
        ax.set_ylabel("Free Energy [kJ/mol]")
        ax.set_ylim(-55, 5)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(frameon=False, loc="lower right")

    ax = axes.flat[3]
    uncertainty = np.asarray([x["mean_posterior_sd_kj_mol"] for x in summaries])
    error = np.asarray([x["rmse_kj_mol"] for x in summaries])
    ax.plot([0.04, 25], [0.04, 25], color="0.4", linestyle="--", linewidth=1,
            label=r"RMSE $=$ posterior SD")
    ax.scatter(uncertainty, error, color="#D55E00", s=38, zorder=3)
    for row, xx, yy in zip(summaries, uncertainty, error):
        ax.annotate(row["condition"], (xx, yy), xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.04, 25)
    ax.set_ylim(0.04, 25)
    ax.set_xlabel("Mean posterior SD [kJ/mol]")
    ax.set_ylabel("RMSE [kJ/mol]")
    ax.set_title("Error versus reported uncertainty")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(output / "bayeswham_ablation_pilot.pdf")
    fig.savefig(output / "bayeswham_ablation_pilot.png", dpi=300)
    plt.close(fig)
    print(f"Wrote pilot figure and metrics to {output}")


if __name__ == "__main__":
    main()
