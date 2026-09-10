#!/usr/bin/env python3
"""Rebuild the 2D reconstruction figure from already-computed CSV outputs.

Reads the GP reconstruction grid written by ``run_2D_reconstruction.py``
(``tutorial_results/synthetic_2D_reconstruction.csv``), the WHAM reference
surface (``xvg_data_2_rad/wham_reference.csv``), and the umbrella window
centers (``xvg_data_2_rad/README``) to redraw the mean/std panel figure
without rerunning the (slow) GP fits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def load_reference_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a (x, y, U) reference CSV on a regular grid and shift its minimum to 0."""
    data = np.loadtxt(path, delimiter=",", comments="#")
    x, y, u = data[:, 0], data[:, 1], data[:, 2]
    u = u - u.min()
    x_axis = np.unique(x)
    y_axis = np.unique(y)
    xi = np.searchsorted(x_axis, x)
    yi = np.searchsorted(y_axis, y)
    grid = np.empty((len(y_axis), len(x_axis)))
    grid[yi, xi] = u
    return x_axis, y_axis, grid


def load_window_centers(path: Path) -> np.ndarray:
    """Parse ``x0, y0`` window-center columns out of the dataset README table."""
    centers = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            centers.append((float(parts[1]), float(parts[2])))
    return np.array(centers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruction-csv",
        default=HERE / "tutorial_results" / "synthetic_2D_reconstruction.csv",
        type=Path,
    )
    parser.add_argument(
        "--reference-csv", default=HERE / "xvg_data_2_rad" / "wham_reference.csv", type=Path
    )
    parser.add_argument("--readme", default=HERE / "xvg_data_2_rad" / "README", type=Path)
    parser.add_argument(
        "--out", default=HERE / "tutorial_results" / "synthetic_2D_reconstruction_replot.png", type=Path
    )
    parser.add_argument("--vmin", type=float, default=None, help="Min color scale for the mean plots (shared across panels).")
    parser.add_argument("--vmax", type=float, default=None, help="Max color scale for the mean plots (shared across panels).")
    args = parser.parse_args()

    data = np.genfromtxt(args.reconstruction_csv, delimiter=",", names=True)
    x, y = data["x"], data["y"]
    n = int(round(np.sqrt(len(x))))
    extent = (x.min(), x.max(), y.min(), y.max())

    field_names = set(data.dtype.names)
    mean_panels = []
    if args.reference_csv.exists():
        x_axis, y_axis, ref_grid = load_reference_grid(args.reference_csv)
        mean_panels.append(("WHAM", x_axis, y_axis, ref_grid, None))
    else:
        mean_panels.append(("WHAM", None, None, data["reference"].reshape(n, n).T, None))

    for label, mean_key, std_key in [
        ("Fixed GP", "fixed_mean", "fixed_std"),
        ("MAP GP", "map_mean", "map_std"),
        ("HMC GP", "hmc_mean", "hmc_std"),
    ]:
        if mean_key not in field_names:
            continue
        std_values = data[std_key] if std_key in field_names else None
        mean_panels.append((label, None, None, data[mean_key].reshape(n, n).T, std_values))

    window_centers = load_window_centers(args.readme) if args.readme.exists() else None

    vmin = args.vmin if args.vmin is not None else min(np.nanmin(p[3]) for p in mean_panels)
    vmax = args.vmax if args.vmax is not None else max(np.nanmax(p[3]) for p in mean_panels)

    n_cols = len(mean_panels)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.75 * n_cols, 7.2), constrained_layout=True)
    for col, (title, x_axis, y_axis, grid_or_flat, std_values) in enumerate(mean_panels):
        ax_mean = axes[0, col]
        if x_axis is not None:
            panel_extent = (x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max())
        else:
            panel_extent = extent
        im = ax_mean.imshow(
            grid_or_flat,
            origin="lower",
            extent=panel_extent,
            aspect="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax_mean.set_title(title + " (kJ/mol)", fontsize=10)
        if window_centers is not None:
            ax_mean.scatter(window_centers[:, 0], window_centers[:, 1], s=3, c="white", alpha=0.4, linewidths=0)
        fig.colorbar(im, ax=ax_mean, shrink=0.85)

        ax_std = axes[1, col]
        if std_values is None:
            ax_std.axis("off")
        else:
            im_std = ax_std.imshow(
                std_values.reshape(n, n).T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="magma",
            )
            ax_std.set_title(f"{title} std [{std_values.min():.3g}, {std_values.max():.3g}]", fontsize=10)
            if window_centers is not None:
                ax_std.scatter(window_centers[:, 0], window_centers[:, 1], s=3, c="white", alpha=0.4, linewidths=0)
            fig.colorbar(im_std, ax=ax_std, shrink=0.85)
        ax_std.set_xlabel("$\\phi$ (rad)")
    axes[0, 0].set_ylabel("$\\psi$ (rad)")
    axes[1, 0].set_ylabel("$\\psi$ (rad)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=250)
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
