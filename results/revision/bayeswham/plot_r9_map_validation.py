#!/usr/bin/env python3
"""Compare the effective-count BayesWHAM MAP with the paper WHAM reference."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


K_B_KJ_MOL_K = 0.0083144621
TEMPERATURE_K = 303.15


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "r9_full/map_validation/effective_counts"
    reference_path = Path.home() / "freeGP-v0.1.0/reference_data/wham.dat"

    x = np.loadtxt(output / "hist_binCenters.txt")
    beta_f = np.loadtxt(output / "betaF_MAP.txt")
    reference = np.loadtxt(reference_path, comments="#")
    reference_x, reference_f = reference[:, 0], reference[:, 1]
    if not np.allclose(x, reference_x):
        raise ValueError("BayesWHAM and reference grids do not match.")

    bayeswham_f = beta_f * K_B_KJ_MOL_K * TEMPERATURE_K
    finite = np.isfinite(bayeswham_f) & np.isfinite(reference_f)
    bayeswham_aligned = bayeswham_f - np.max(bayeswham_f[finite])
    reference_aligned = reference_f - np.max(reference_f[finite])
    difference = bayeswham_aligned - reference_aligned

    metrics = {
        "alignment": "each profile shifted so its maximum is zero",
        "temperature_k": TEMPERATURE_K,
        "points_compared": int(finite.sum()),
        "rmse_kj_mol": float(np.sqrt(np.mean(difference[finite] ** 2))),
        "mean_difference_kj_mol": float(np.mean(difference[finite])),
        "max_absolute_difference_kj_mol": float(np.max(np.abs(difference[finite]))),
        "bayeswham_barrier_kj_mol": float(np.ptp(bayeswham_aligned[finite])),
        "reference_barrier_kj_mol": float(np.ptp(reference_aligned[finite])),
    }
    (output / "map_validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(
        output / "map_validation_profiles.csv",
        np.column_stack((x, reference_aligned, bayeswham_aligned, difference)),
        delimiter=",",
        header="coordinate_nm,wham_reference_kj_mol,bayeswham_map_kj_mol,difference_kj_mol",
        comments="",
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(4.5, 3.3), constrained_layout=True)
    ax.plot(x, reference_aligned, color="black", linewidth=2.0, label="WHAM reference")
    ax.plot(
        x,
        bayeswham_aligned,
        color="#D55E00",
        linewidth=1.7,
        linestyle="--",
        label="BayesWHAM MAP (AR(1) effective counts)",
    )
    ax.set_xlabel("Membrane--peptide distance [nm]")
    ax.set_ylabel("Free Energy [kJ/mol]")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.04,
        0.72,
        rf"RMSE = {metrics['rmse_kj_mol']:.2f} kJ/mol",
        transform=ax.transAxes,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "bayeswham_map_vs_wham_reference.pdf")
    fig.savefig(output / "bayeswham_map_vs_wham_reference.png", dpi=300)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote validation plot to {output}")


if __name__ == "__main__":
    main()
