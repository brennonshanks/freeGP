#!/usr/bin/env python3
"""Plot BayesWHAM R9 ablation heatmaps and parity in manuscript style."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from visualization import (  # noqa: E402
    DEFAULT_TRAJ_FRACTIONS,
    DEFAULT_WINDOW_COUNTS,
    PAPER_LABEL_SIZE,
    PAPER_LINE_WIDTH,
    PAPER_TICK_SIZE,
    TwoSlopeLogNorm,
    _add_vertical_colorbar,
    _draw_preview_heatmap,
    configure_main_text_matplotlib,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    analysis = root / "r9_ablation_grid/analysis"
    output = analysis / "figures"
    output.mkdir(parents=True, exist_ok=True)
    with (analysis / "ablation_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    def grid(metric: str) -> np.ndarray:
        values = np.full((len(DEFAULT_WINDOW_COUNTS), len(DEFAULT_TRAJ_FRACTIONS)), np.nan)
        for i, windows in enumerate(DEFAULT_WINDOW_COUNTS):
            for j, fraction in enumerate(DEFAULT_TRAJ_FRACTIONS):
                match = next(
                    row for row in rows
                    if int(row["window_count"]) == windows
                    and np.isclose(float(row["trajectory_fraction"]), fraction)
                )
                values[i, j] = float(match[metric])
        return values

    rmse_grid = grid("rmse_wham")
    sd_grid = grid("avg_total_std")
    rmse_norm = TwoSlopeLogNorm(vmin=0.15, vcenter=5.0, vmax=175.0, power=1.5)
    sd_norm = TwoSlopeLogNorm(vmin=0.06, vcenter=5.0, vmax=50.0, power=1.5)
    cmap = "RdBu_r"
    configure_main_text_matplotlib()

    fig = plt.figure(figsize=(3.33, 7.25), constrained_layout=False)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 0.055],
        wspace=0.12, hspace=0.24,
    )
    axes = []
    for row_index, (values, norm, label, ticks) in enumerate(
        [
            (rmse_grid, rmse_norm, "RMSE [kJ/mol]", [0.15, 0.5, 2, 5, 15, 50, 175]),
            (sd_grid, sd_norm, "Standard Deviation [kJ/mol]", [0.06, 0.3, 1, 5, 10, 25, 50]),
        ]
    ):
        ax = fig.add_subplot(gs[row_index, 0])
        axes.append(ax)
        _draw_preview_heatmap(
            ax, values, norm=norm, cmap=cmap,
            show_xlabels=True, show_ylabels=True,
        )
        ax.set_title("BayesWHAM" if row_index == 0 else "", pad=5)
        _add_vertical_colorbar(
            fig, fig.add_subplot(gs[row_index, 1]), norm=norm, cmap=cmap,
            ticks=ticks, label=label, label_x=4.2,
        )

    parity_ax = fig.add_subplot(gs[2, 0])
    axes.append(parity_ax)
    rmse = np.asarray([float(row["rmse_wham"]) for row in rows])
    sd = np.asarray([float(row["avg_total_std"]) for row in rows])
    windows = np.asarray([float(row["window_count"]) for row in rows])
    crps = float(np.mean([float(row["mean_crps_kj_mol"]) for row in rows]))
    correlation = float(np.corrcoef(rmse, sd)[0, 1])
    threshold = 5.0
    under = int(np.sum((rmse <= threshold) & (sd > threshold)))
    over = int(np.sum((rmse > threshold) & (sd <= threshold)))
    parity_min = 0.045
    parity_max = 28.0
    window_norm = mcolors.Normalize(min(DEFAULT_WINDOW_COUNTS), max(DEFAULT_WINDOW_COUNTS))
    parity_ax.fill_between([parity_min, threshold], threshold, parity_max,
                           color="#E69F00", alpha=0.07, zorder=0)
    parity_ax.fill_between([threshold, parity_max], parity_min, threshold,
                           color="#D55E00", alpha=0.07, zorder=0)
    parity_ax.scatter(rmse, sd, c=windows, cmap="viridis", norm=window_norm,
                      s=9, alpha=0.8, linewidths=0.2, edgecolors="white", rasterized=True)
    parity_ax.plot([parity_min, parity_max], [parity_min, parity_max],
                   color="black", linestyle="--", linewidth=0.55)
    parity_ax.axvline(threshold, color="#666666", linestyle=":", linewidth=0.5)
    parity_ax.axhline(threshold, color="#666666", linestyle=":", linewidth=0.5)
    parity_ax.set_xscale("log")
    parity_ax.set_yscale("log")
    parity_ax.set_xlim(parity_min, parity_max)
    parity_ax.set_ylim(parity_min, parity_max)
    parity_ax.set_box_aspect(1)
    parity_ax.tick_params(axis="both", which="both", direction="in", pad=1)
    parity_ax.text(0.04, 0.96, rf"$r={correlation:.2f}$" + "\n" + rf"CRPS$={crps:.1f}$",
                   transform=parity_ax.transAxes, ha="left", va="top", fontsize=PAPER_TICK_SIZE)
    parity_ax.text(0.96, 0.04, rf"$U={under},\ O={over}$",
                   transform=parity_ax.transAxes, ha="right", va="bottom", fontsize=PAPER_TICK_SIZE)
    cax = fig.add_subplot(gs[2, 1])
    sm = plt.cm.ScalarMappable(norm=window_norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", ticks=DEFAULT_WINDOW_COUNTS)
    cbar.set_label("Umbrella windows", labelpad=8)
    cbar.ax.yaxis.set_label_coords(4.2, 0.5)
    cbar.ax.tick_params(length=2.4, width=PAPER_LINE_WIDTH, pad=1)
    cbar.outline.set_linewidth(PAPER_LINE_WIDTH)

    fig.subplots_adjust(left=0.19, right=0.83, bottom=0.075, top=0.97)
    x_offset = 0.044
    y_offset = 0.13
    for index, ax in enumerate(axes):
        box = ax.get_position()
        fig.text(0.5 * (box.x0 + box.x1), box.y0 - x_offset,
                 "Number of umbrella windows" if index < 2 else "RMSE [kJ/mol]",
                 ha="center", fontsize=PAPER_LABEL_SIZE)
        fig.text(box.x0 - y_offset, 0.5 * (box.y0 + box.y1),
                 "Trajectory length [%]" if index < 2 else "Standard Deviation [kJ/mol]",
                 ha="center", va="center", rotation="vertical", fontsize=PAPER_LABEL_SIZE)
    for fmt in ("pdf", "png"):
        fig.savefig(output / f"bayeswham_ablation_heatmaps_with_parity.{fmt}",
                    dpi=600 if fmt == "png" else None)
    plt.close(fig)
    print(f"Wrote BayesWHAM ablation figure to {output}")


if __name__ == "__main__":
    main()
