#!/usr/bin/env python3
"""Generate alanine-dipeptide paper figures from saved reconstructions.

Usage: python paper_figures.py [--formats pdf png]
Style settings are shared with scripts/visualization.py.

Each point represents one saved reconstruction, not an average over replicate runs.
Errors use the full WHAM reference with periodic bilinear interpolation,
zeroed at its sampled minimum. Saved GP means and SDs are unchanged.
The saved evaluation grids differ between datasets. The parity line is a visual
comparison: mean SD is not RMS SD, so equality is not a strict calibration test.
Pearson r compares the six RMSE/mean-SD pairs per method. CRPS averages Gaussian
marginal scores over grid points, then equally over datasets (including a Gaussian
approximation for HMC). U counts RMSE <= 5 and mean SD > 5; O counts RMSE > 5
and mean SD <= 5, in kJ/mol. These are threshold-based dataset counts.
"""
from __future__ import annotations

import argparse
import sys
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from visualization import configure_main_text_matplotlib, JCTC_DOUBLE_COLUMN_WIDTH_IN
from analyze_ablation_correlations import normal_crps, pearson_r
RESULTS_ROOT = HERE / "data_ablation_results"
DATASETS = [
    ("18_windows", "umbrella_data/xvg_data_32"),
    ("36_windows", "umbrella_data/xvg_data_16"),
    ("72_windows", "umbrella_data/xvg_data_8"),
    ("144_windows", "umbrella_data/xvg_data_quarter"),
    ("288_windows", "umbrella_data/xvg_data_half"),
    ("576_windows", "umbrella_data/xvg_data"),
]
METHODS = [
    ("Fixed GP", "fixed_mean", "fixed_std"),
    ("MAP GP", "map_mean", "map_std"),
    ("HMC GP", "hmc_mean", "hmc_std"),
]


def periodic_reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate across the periodic seam without extrapolating edge slopes."""
    raw = np.loadtxt(HERE / "umbrella_data/xvg_data" / "wham_reference.csv", delimiter=",")
    phi, psi = np.unique(raw[:, 0]), np.unique(raw[:, 1])
    grid = np.full((len(phi), len(psi)), np.nan)
    grid[np.searchsorted(phi, raw[:, 0]), np.searchsorted(psi, raw[:, 1])] = raw[:, 2]
    if not np.isfinite(grid).all():
        raise ValueError("WHAM reference must be a complete finite grid")
    period = 2 * np.pi
    phi_ext = np.r_[phi[-1] - period, phi, phi[0] + period]
    psi_ext = np.r_[psi[-1] - period, psi, psi[0] + period]
    interpolate = RegularGridInterpolator(
        (phi_ext, psi_ext), np.pad(grid, 1, mode="wrap"), bounds_error=True,
    )
    points = np.column_stack(((x + np.pi) % period - np.pi,
                              (y + np.pi) % period - np.pi))
    reference = interpolate(points)
    if reference.min() < grid.min() - 1e-8 or reference.max() > grid.max() + 1e-8:
        raise ValueError("Periodic interpolation exceeded reference energy range")
    return reference - reference.min()


def collect_metrics():
    rows = []
    for result_dir, xvg_dir in DATASETS:
        data = np.genfromtxt(
            RESULTS_ROOT / result_dir / "synthetic_2D_reconstruction.csv",
            delimiter=",", names=True,
        )
        windows = len(list((HERE / xvg_dir).glob("*_xyplane.xvg")))
        if windows == 0:
            raise ValueError(f"No umbrella windows in {xvg_dir}")
        reference = periodic_reference(data["x"], data["y"])
        if not np.isfinite(reference).all() or np.ptp(reference) == 0:
            raise ValueError(f"Missing or invalid reference in {result_dir}")
        for label, mean_key, std_key in METHODS:
            mean, sd = data[mean_key], data[std_key]
            if not (np.isfinite(mean).all() and np.isfinite(sd).all() and (sd >= 0).all()):
                raise ValueError(f"Invalid predictions for {label} in {result_dir}")
            rows.append(dict(
                dataset=result_dir, method=label, n_windows=windows,
                n_grid_points=len(data),
                rmse=float(np.sqrt(np.mean((mean - reference) ** 2))),
                mean_sd=float(np.mean(sd)),
                rms_sd=float(np.sqrt(np.mean(sd ** 2))),
                mean_crps_kj_mol=float(np.mean(normal_crps(mean, sd, reference))),
            ))

    return rows


def plot_si(args):
    rows = collect_metrics()
    with (args.output_dir / "convergence_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig = plt.figure(figsize=(JCTC_DOUBLE_COLUMN_WIDTH_IN, 4.4), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=.12)
    top = outer[0].subgridspec(1, 5, width_ratios=[1, 1, 1, 1, .055])
    bottom = outer[1].subgridspec(1, 3, wspace=.08)
    data = np.genfromtxt(RESULTS_ROOT / args.dataset / "synthetic_2D_reconstruction.csv",
                         delimiter=",", names=True)
    x, y = np.unique(data["x"]), np.unique(data["y"])
    means = [periodic_reference(data["x"], data["y"])] + [data[key] for _, key, _ in METHODS]
    vmax = max(float(np.max(values)) for values in means)
    for i, (label, values) in enumerate(zip(["WHAM"] + [m[0] for m in METHODS], means)):
        grid = np.full((len(y), len(x)), np.nan)
        grid[np.searchsorted(y, data["y"]), np.searchsorted(x, data["x"])] = values
        if not np.isfinite(grid).all():
            raise ValueError("Surface must be a complete finite grid")
        ax = fig.add_subplot(top[0, i])
        im = ax.pcolormesh(x, y, grid, shading="nearest", cmap="viridis",
                          vmin=0, vmax=vmax, rasterized=True)
        ax.set(title=f"({chr(97+i)}) {label}", xlabel=r"$\phi$ [rad]",
               xlim=(x.min(), x.max()), ylim=(y.min(), y.max()), aspect="equal")
        ax.set_xticks([-3, 0, 3])
        ax.set_yticks([-3, 0, 3])
        if i == 0:
            ax.set_ylabel(r"$\psi$ [rad]")
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(direction="in")
    fig.colorbar(im, cax=fig.add_subplot(top[0, 4]), label="Free energy [kJ/mol]")

    lower = 1.0
    limit = 1.1 * max(max(r["rmse"], r["mean_sd"]) for r in rows)
    colors = plt.get_cmap("viridis")(np.linspace(.1, .9, len(DATASETS)))
    markers = ["o", "s", "^", "D", "v", "P"]
    summaries = []
    for i, (label, _, _) in enumerate(METHODS):
        ax = fig.add_subplot(bottom[0, i])
        selected = [r for r in rows if r["method"] == label]
        rmse = np.array([r["rmse"] for r in selected])
        sd = np.array([r["mean_sd"] for r in selected])
        correlation = pearson_r(rmse, sd)
        crps = float(np.mean([r["mean_crps_kj_mol"] for r in selected]))
        under = int(np.sum((rmse <= 5) & (sd > 5)))
        over = int(np.sum((rmse > 5) & (sd <= 5)))
        summaries.append(dict(method=label, n_datasets=len(selected), pearson_r=correlation,
                              mean_crps_kj_mol=crps, threshold_kj_mol=5,
                              underconfident_count=under, overconfident_count=over))
        ax.fill_between([lower, 5], 5, limit, color="#E69F00", alpha=.07, zorder=0)
        ax.fill_between([5, limit], lower, 5, color="#D55E00", alpha=.07, zorder=0)
        ax.axvline(5, color=".4", linestyle=":", linewidth=.5)
        ax.axhline(5, color=".4", linestyle=":", linewidth=.5)
        ax.text(.04, .96, f"$r$ = {correlation:.2f}\nMean CRPS = {crps:.2f} kJ/mol",
                transform=ax.transAxes, va="top", fontsize=5.4)
        ax.text(.96, .04, f"U = {under}, O = {over}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.4)
        ax.plot([lower, limit], [lower, limit], "--", color=".5", linewidth=.5, zorder=1)
        for row, color, marker in zip(selected, colors, markers):
            ax.scatter(row["rmse"], row["mean_sd"], color=color, marker=marker,
                       s=17, linewidths=.3, edgecolors="white", zorder=2,
                       label=str(row["n_windows"]))
        ax.set(title=f"({chr(101+i)}) {label}", xlabel="RMSE [kJ/mol]",
               xscale="log", yscale="log",
               xlim=(lower, limit), ylim=(lower, limit), aspect="equal")
        ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_yticks([1, 2, 5, 10, 20, 40])
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        ax.tick_params(direction="in")
        if i == 0:
            ax.set_ylabel("Mean predictive SD [kJ/mol]")
            ax.legend(title="Umbrella windows", loc="upper left", bbox_to_anchor=(0, .79), ncol=2,
                      frameon=False, fontsize=5.4, title_fontsize=5.4,
                      columnspacing=.5, handletextpad=.3, labelspacing=.3)
        else:
            ax.tick_params(labelleft=False)
    with (args.output_dir / "calibration_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for extension in args.formats:
        path = args.output_dir / f"alanine_dipeptide_si.{extension}"
        fig.savefig(path, dpi=args.dpi)
        print(f"Wrote {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "paper_figures")
    parser.add_argument("--dataset", choices=[d[0] for d in DATASETS], default="576_windows",
                        help="Saved reconstruction used for the heatmaps.")
    parser.add_argument("--formats", nargs="+", choices=["pdf", "svg", "png"], default=["pdf"])
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_main_text_matplotlib()
    plot_si(args)


if __name__ == "__main__":
    main()
