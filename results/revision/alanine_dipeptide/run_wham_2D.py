#!/usr/bin/env python3
"""Reconstruct the 2D (phi, psi) free-energy surface of alanine dipeptide with WHAM.

Loads umbrella-sampling trajectories from ``xvg_data_2_rad/`` (one ``w###_xyplane.xvg``
file per window, each biased by an isotropic harmonic restraint centered at (x0, y0) with
the given force constant -- see ``xvg_data_2_rad/README``), bins the combined (phi, psi)
samples on a periodic grid, and solves the self-consistent 2D WHAM equations (Kumar et al.
1992) to recover the unbiased free-energy surface. Both collective variables are dihedral
angles, so all differences are taken with the periodic minimum-image convention on
[-pi, pi).

Results are written as a (phi, psi, free_energy) CSV and plotted as a heatmap.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent

# Boltzmann constant in kJ/mol/K and sampling temperature, matching src/freegp/preprocess.py.
K_B = 8.3144621e-3
TEMPERATURE = 303.15
BETA = 1.0 / (K_B * TEMPERATURE)

TWO_PI = 2.0 * np.pi
KJ_PER_KCAL = 4.184


def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    """Wrap angles into [-pi, pi)."""
    return (theta + np.pi) % TWO_PI - np.pi


def load_windows(dataset_root: Path) -> list[dict]:
    """Parse the README (window_id x0 y0 force_constant) and load each window's xvg samples."""
    readme = dataset_root / "README"
    windows = []
    with readme.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            window_id, x0, y0, k = line.split()
            xvg_path = dataset_root / f"{window_id}_xyplane.xvg"
            data = np.loadtxt(xvg_path, comments=("#", "@"))
            phi = wrap_to_pi(data[:, 1])
            psi = wrap_to_pi(data[:, 2])
            windows.append(
                {
                    "id": window_id,
                    "x0": float(x0),
                    "y0": float(y0),
                    "k": float(k),
                    "phi": phi,
                    "psi": psi,
                }
            )
    return windows


def run_wham_2d(
    windows: list[dict],
    num_bins: int,
    tol: float = 1e-7,
    max_iter: int = 5000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the self-consistent 2D WHAM equations on a periodic (phi, psi) grid.

    Returns (phi_centers, psi_centers, free_energy) where free_energy has shape
    (num_bins, num_bins) indexed [psi_bin, phi_bin], in kJ/mol, shifted so its minimum is 0.
    """
    edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    num_windows = len(windows)
    num_samples = np.array([len(w["phi"]) for w in windows], dtype=np.float64)

    # Combined histogram counts n_l across all windows, flattened bin index l = psi_bin*num_bins + phi_bin.
    counts = np.zeros((num_bins, num_bins), dtype=np.float64)
    for w in windows:
        h, _, _ = np.histogram2d(w["psi"], w["phi"], bins=[edges, edges])
        counts += h
    log_n_l = np.full(counts.size, -np.inf)
    nonzero = counts.flatten() > 0
    log_n_l[nonzero] = np.log(counts.flatten()[nonzero])

    # Bias energy U_i(l) = 0.5*k_i*(dphi^2 + dpsi^2) for every (window, bin) pair, periodic minimum image.
    phi_grid, psi_grid = np.meshgrid(centers, centers)  # shape (num_bins, num_bins), [psi, phi]
    phi_flat = phi_grid.flatten()
    psi_flat = psi_grid.flatten()

    x0 = np.array([w["x0"] for w in windows])
    y0 = np.array([w["y0"] for w in windows])
    k = np.array([w["k"] for w in windows])

    dphi = wrap_to_pi(phi_flat[None, :] - x0[:, None])
    dpsi = wrap_to_pi(psi_flat[None, :] - y0[:, None])
    bias_U = 0.5 * k[:, None] * (dphi**2 + dpsi**2)  # shape (num_windows, num_bins*num_bins)
    neg_beta_U = -BETA * bias_U

    log_N_i = np.log(num_samples)
    f_i = np.zeros(num_windows)

    for _ in range(max_iter):
        # log P_l = log(n_l) - logsumexp_i[log(N_i) + beta*f_i - beta*U_i(l)]
        log_denom = logsumexp(log_N_i[:, None] + BETA * f_i[:, None] + neg_beta_U, axis=0)
        log_P_l = log_n_l - log_denom

        # f_i = -1/beta * logsumexp_l[log(P_l) - beta*U_i(l)]
        f_i_new = -(1.0 / BETA) * logsumexp(log_P_l[None, :] + neg_beta_U, axis=1)
        f_i_new -= f_i_new[0]  # fix the arbitrary additive constant in the f_i's

        shift = np.max(np.abs(f_i_new - f_i))
        f_i = f_i_new
        if shift < tol:
            break

    log_denom = logsumexp(log_N_i[:, None] + BETA * f_i[:, None] + neg_beta_U, axis=0)
    log_P_l = log_n_l - log_denom

    free_energy = np.full(counts.size, np.inf)
    finite = np.isfinite(log_P_l)
    free_energy[finite] = -(1.0 / BETA) * log_P_l[finite]
    free_energy -= free_energy[finite].min()
    free_energy = free_energy.reshape(num_bins, num_bins)

    return centers, centers, free_energy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="xvg_data_2_rad")
    parser.add_argument("--results-dir", default="tutorial_results")
    parser.add_argument("--num-bins", type=int, default=36, help="Histogram bins per dimension.")
    parser.add_argument("--vmax", type=float, default=None, help="Max color scale (kcal/mol) for the heatmap.")
    args = parser.parse_args()

    dataset_root = (HERE / args.dataset_root).resolve()
    out = (HERE / args.results_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading umbrella windows from {dataset_root} ...")
    windows = load_windows(dataset_root)
    total_samples = sum(len(w["phi"]) for w in windows)
    print(f"Loaded {len(windows)} windows, {total_samples} total samples.")

    print(f"Running 2D WHAM with {args.num_bins}x{args.num_bins} bins ...")
    phi_centers, psi_centers, free_energy = run_wham_2d(windows, num_bins=args.num_bins)
    free_energy_kcal = free_energy / KJ_PER_KCAL

    phi_grid, psi_grid = np.meshgrid(phi_centers, psi_centers)
    csv_path = out / "alanine_dipeptide_wham_2D.csv"
    # kJ/mol here (not free_energy_kcal, which is display-only for the plot below): this
    # CSV is consumed as the reference surface by run_2D_reconstruction.py, whose GP output
    # is in kJ/mol -- matching BETA (== freegp.preprocess.beta) above. Saving kcal/mol here
    # silently introduced a 4.184x scale mismatch against that GP output.
    rows = np.column_stack([phi_grid.flatten(), psi_grid.flatten(), free_energy.flatten()])
    # Default comments="# " (not ""): run_2D_reconstruction.py's load_2d_reference calls
    # np.loadtxt(..., comments="#") to skip the header row, matching make_dataset.py's
    # ground_truth.csv convention for the synthetic tutorial.
    np.savetxt(csv_path, rows, delimiter=",", header="phi,psi,free_energy_kJ_per_mol")
    print(f"Wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    extent = (phi_centers.min(), phi_centers.max(), psi_centers.min(), psi_centers.max())
    vmax = args.vmax / KJ_PER_KCAL if args.vmax is not None else None
    im = ax.imshow(
        free_energy_kcal,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmax=vmax,
    )
    ax.set_xlabel(r"$\phi$ [rad]")
    ax.set_ylabel(r"$\psi$ [rad]")
    ax.set_title("Alanine dipeptide free-energy surface (2D WHAM)")
    fig.colorbar(im, ax=ax, label="Free energy [kcal/mol]")

    png_path = out / "alanine_dipeptide_wham_2D.png"
    fig.savefig(png_path, dpi=250)
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
