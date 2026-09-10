#!/usr/bin/env python3
"""Plot how GP predictive uncertainty shrinks as more umbrella-sampling data is used.

For each ``tutorial_results*`` directory (full, half, quarter, eighth, sixteenth of
the data) this reads ``synthetic_2D_reconstruction.csv`` and computes the mean
predictive std (average of ``fixed_std`` / ``hmc_std`` over the grid) for the Fixed GP
and HMC GP methods, then plots both curves vs. the number of umbrella windows in a
single plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# (result dir, corresponding xvg data dir) ordered from least to most data
DATASETS = [
    ("tutorial_results_32th", "xvg_data_32th"),
    ("tutorial_results_sixteenth", "xvg_data_sixteenth"),
    ("tutorial_results_eight", "xvg_data_eight"),
    ("tutorial_results_quarter", "xvg_data_quarter"),
    ("tutorial_results_half", "xvg_data_half"),
    ("tutorial_results", "xvg_data"),
]

METHODS = [
    ("Fixed GP", "fixed_mean", "fixed_std"),
    ("HMC GP", "hmc_mean", "hmc_std"),
]


def n_windows(xvg_dir: Path) -> int:
    return len(list(xvg_dir.glob("*.xvg")))


def main() -> None:
    n_points = []
    mean_std = {label: [] for label, _, _ in METHODS}

    for result_dir, xvg_dir in DATASETS:
        csv_path = HERE / result_dir / "synthetic_2D_reconstruction.csv"
        data = np.genfromtxt(csv_path, delimiter=",", names=True)

        n_points.append(n_windows(HERE / xvg_dir))
        for label, _, std_key in METHODS:
            mean_std[label].append(np.mean(data[std_key]))

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for label, _, _ in METHODS:
        ax.plot(n_points, mean_std[label], "o-", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Number of umbrella windows")
    ax.set_ylabel("Mean predictive std [kJ/mol]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = HERE / "tutorial_results" / "convergence.png"
    fig.savefig(out, dpi=250)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
