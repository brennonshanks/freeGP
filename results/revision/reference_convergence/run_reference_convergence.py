#!/usr/bin/env python3
"""Check convergence and overlap of the R9 WHAM reference calculation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from freegp.preprocess import bayes_autocorrelation_time


K_B_KJ_MOL_K = 0.0083144621
TEMPERATURE_K = 303.15
EQUILIBRATION_FRAMES = 40_000
N_BLOCKS = 5
CUMULATIVE_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
WINDOW_PATTERN = re.compile(r"d_([0-9]+\.[0-9]+)$")


def read_mdp(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_pullx(path: Path) -> np.ndarray:
    values = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("#", "@")):
                fields = line.split()
                if len(fields) >= 2:
                    values.append(float(fields[1]))
    return np.asarray(values)


def thin_ar1(values: np.ndarray) -> tuple[np.ndarray, float, int]:
    tau = float(bayes_autocorrelation_time(torch.as_tensor(values, dtype=torch.float64)))
    stride = max(1, int(math.ceil(tau)))
    return values[::stride], tau, stride


def wham_map(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    umbrella_centers: np.ndarray,
    force_constants: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, int, float]:
    """Solve the uniform-prior BayesWHAM MAP/standard WHAM equations."""
    beta = 1.0 / (K_B_KJ_MOL_K * TEMPERATURE_K)
    bias = 0.5 * force_constants[:, None] * (
        bin_centers[None, :] - umbrella_centers[:, None]
    ) ** 2
    c_il = np.exp(-beta * bias)
    n_i = counts.sum(axis=1)
    m_l = counts.sum(axis=0)
    occupied = m_l > 0
    if not np.any(occupied):
        raise ValueError("No occupied histogram bins")
    c = c_il[:, occupied]
    m = m_l[occupied]
    p = np.full(m.size, 1.0 / m.size)
    f = 1.0 / (c @ p)
    delta = np.inf
    for iteration in range(1, max_iterations + 1):
        old = p.copy()
        denominator = (n_i * f) @ c
        p = m / denominator
        p /= p.sum()
        f = 1.0 / (c @ p)
        delta = float(np.max(np.abs(p - old)))
        if delta <= tolerance:
            break
    else:
        raise RuntimeError(f"WHAM did not converge in {max_iterations} iterations")
    probability = np.zeros(bin_centers.size)
    probability[occupied] = p
    free_energy = np.full(bin_centers.size, np.nan)
    free_energy[occupied] = -np.log(p) / beta
    free_energy[occupied] -= np.max(free_energy[occupied])
    return free_energy, iteration, delta


def align_rmse(surface: np.ndarray, reference: np.ndarray) -> float:
    mask = np.isfinite(surface) & np.isfinite(reference)
    return float(np.sqrt(np.mean((surface[mask] - reference[mask]) ** 2)))


def main() -> None:
    here = Path(__file__).resolve().parent
    dataset = Path.home() / "freeGP-datasets/membranes/katka"
    reference_path = Path.home() / "freeGP-v0.1.0/reference_data/wham.dat"
    output = here / "outputs"
    output.mkdir(parents=True, exist_ok=True)

    reference_data = np.loadtxt(reference_path)
    x = reference_data[:, 0]
    reference = reference_data[:, 1] - np.max(reference_data[:, 1])
    step = float(np.diff(x).mean())
    edges = np.concatenate(([x[0] - step / 2], x + step / 2))

    windows = []
    for folder in dataset.iterdir():
        match = WINDOW_PATTERN.fullmatch(folder.name)
        if folder.is_dir() and match:
            mdp = read_mdp(folder / "step7_production.mdp")
            positions = read_pullx(folder / "step7_production_pullx.xvg")
            positions = positions[EQUILIBRATION_FRAMES:]
            windows.append(
                (float(match.group(1)), float(mdp["pull-coord1-k"]), positions)
            )
    windows.sort(key=lambda row: row[0])
    if len(windows) != 25:
        raise ValueError(f"Expected 25 windows, found {len(windows)}")
    centers = np.asarray([row[0] for row in windows])
    springs = np.asarray([row[1] for row in windows])

    profile_rows = []
    metric_rows = []
    thinning_rows = []
    surfaces: dict[str, np.ndarray] = {}

    analyses = []
    for block in range(N_BLOCKS):
        segments = []
        for _, _, trajectory in windows:
            bounds = np.linspace(0, trajectory.size, N_BLOCKS + 1, dtype=int)
            segments.append(trajectory[bounds[block] : bounds[block + 1]])
        analyses.append(("block", block + 1, (block + 1) / N_BLOCKS, segments))
    for index, fraction in enumerate(CUMULATIVE_FRACTIONS, start=1):
        segments = [trajectory[: max(2, int(trajectory.size * fraction))] for _, _, trajectory in windows]
        analyses.append(("cumulative", index, fraction, segments))

    full_thinned = None
    full_thinned_samples = None
    for kind, index, fraction, segments in analyses:
        histograms = []
        thinned_segments = []
        for window_index, (center, segment) in enumerate(zip(centers, segments), start=1):
            thinned, tau, stride = thin_ar1(segment)
            thinned_segments.append(thinned)
            histogram, _ = np.histogram(thinned, bins=edges)
            histograms.append(histogram)
            thinning_rows.append(
                {
                    "analysis": kind,
                    "index": index,
                    "fraction": fraction,
                    "window_index": window_index,
                    "window_center_nm": center,
                    "raw_segment_frames": segment.size,
                    "ar1_tau_frames": tau,
                    "thinning_stride": stride,
                    "thinned_frames": thinned.size,
                    "histogrammed_frames": int(histogram.sum()),
                }
            )
        counts = np.asarray(histograms, dtype=float)
        surface, iterations, residual = wham_map(counts, x, centers, springs)
        label = f"{kind}_{index}"
        surfaces[label] = surface
        if kind == "cumulative" and np.isclose(fraction, 1.0):
            full_thinned = counts
            full_thinned_samples = thinned_segments
        metric_rows.append(
            {
                "analysis": kind,
                "index": index,
                "fraction": fraction,
                "rmse_to_reference_kj_mol": align_rmse(surface, reference),
                "wham_iterations": iterations,
                "final_max_probability_change": residual,
                "total_thinned_histogram_count": int(counts.sum()),
            }
        )
        for coordinate, value in zip(x, surface):
            profile_rows.append(
                {"analysis": kind, "index": index, "fraction": fraction,
                 "coordinate_nm": coordinate, "free_energy_kj_mol": value,
                 "between_block_sd_kj_mol": ""}
            )

    if full_thinned is None or full_thinned_samples is None:
        raise RuntimeError("Full-data histogram was not generated")
    overlap_min = min(float(values.min()) for values in full_thinned_samples)
    overlap_max = max(float(values.max()) for values in full_thinned_samples)
    overlap_first_center = step * math.floor(overlap_min / step)
    overlap_last_center = step * math.ceil(overlap_max / step)
    overlap_x = np.arange(overlap_first_center, overlap_last_center + step / 2, step)
    overlap_edges = np.concatenate(
        ([overlap_x[0] - step / 2], overlap_x + step / 2)
    )
    overlap_counts = np.asarray(
        [np.histogram(values, bins=overlap_edges)[0] for values in full_thinned_samples],
        dtype=float,
    )
    probabilities = overlap_counts / np.maximum(overlap_counts.sum(axis=1, keepdims=True), 1)
    adjacent_overlap = np.sum(np.minimum(probabilities[:-1], probabilities[1:]), axis=1)
    aggregate_counts = full_thinned.sum(axis=0)
    occupied_bins = aggregate_counts > 0
    occupied_indices = np.flatnonzero(occupied_bins)
    aggregate_histogram_connected = bool(
        occupied_indices.size > 0
        and np.all(occupied_bins[occupied_indices[0] : occupied_indices[-1] + 1])
    )
    block_stack = np.stack([surfaces[f"block_{i}"] for i in range(1, N_BLOCKS + 1)])
    block_mean = np.nanmean(block_stack, axis=0)
    block_sd = np.nanstd(block_stack, axis=0, ddof=1)
    metric_rows.append(
        {
            "analysis": "block_average",
            "index": 0,
            "fraction": 0.2,
            "rmse_to_reference_kj_mol": align_rmse(block_mean, reference),
            "wham_iterations": "",
            "final_max_probability_change": "",
            "total_thinned_histogram_count": int(sum(row["total_thinned_histogram_count"] for row in metric_rows if row["analysis"] == "block")),
        }
    )
    for coordinate, mean, sd in zip(x, block_mean, block_sd):
        profile_rows.append(
            {"analysis": "block_average", "index": 0, "fraction": 0.2,
             "coordinate_nm": coordinate, "free_energy_kj_mol": mean,
             "between_block_sd_kj_mol": sd}
        )
    overlap_rows = [
        {
            "left_center_nm": centers[i],
            "right_center_nm": centers[i + 1],
            "histogram_overlap_coefficient": adjacent_overlap[i],
        }
        for i in range(centers.size - 1)
    ]

    for filename, rows in (
        ("wham_profiles.csv", profile_rows),
        ("wham_metrics.csv", metric_rows),
        ("autocorrelation_thinning.csv", thinning_rows),
        ("adjacent_histogram_overlap.csv", overlap_rows),
    ):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "dataset": str(dataset),
        "reference": str(reference_path),
        "equilibration_frames_removed": EQUILIBRATION_FRAMES,
        "autocorrelation_treatment": "segment-specific AR(1) tau; deterministic stride ceil(tau)",
        "wham_solver": "uniform-prior BayesWHAM MAP fixed-point equations",
        "blocks": N_BLOCKS,
        "cumulative_fractions": list(CUMULATIVE_FRACTIONS),
        "minimum_adjacent_histogram_overlap": float(adjacent_overlap.min()),
        "median_adjacent_histogram_overlap": float(np.median(adjacent_overlap)),
        "aggregate_histogram_connected": aggregate_histogram_connected,
        "occupied_bins": int(occupied_bins.sum()),
        "total_bins": int(occupied_bins.size),
        "block_average_rmse_to_reference_kj_mol": align_rmse(block_mean, reference),
        "mean_between_block_sd_kj_mol": float(np.nanmean(block_sd)),
        "maximum_between_block_sd_kj_mol": float(np.nanmax(block_sd)),
        "overlap_histogram_range_nm": [float(overlap_edges[0]), float(overlap_edges[-1])],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9.5, "legend.fontsize": 7.2})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    for i, color in enumerate(colors, start=1):
        axes[0].plot(x, surfaces[f"block_{i}"], color=color, linewidth=0.9, alpha=0.65, label=f"Block {i}")
    axes[0].fill_between(x, block_mean - block_sd, block_mean + block_sd,
                         color="#0072B2", alpha=0.18, linewidth=0,
                         label=r"Block mean $\pm$ SD")
    axes[0].plot(x, block_mean, color="#0072B2", linewidth=1.6, label="Block mean")
    axes[0].plot(x, reference, color="black", linewidth=1.6, linestyle="--", label="Published reference")
    axes[0].set_title("Five contiguous trajectory blocks")
    for i, (fraction, color) in enumerate(zip(CUMULATIVE_FRACTIONS, colors), start=1):
        axes[1].plot(x, surfaces[f"cumulative_{i}"], color=color, linewidth=1.1,
                     label=f"{int(100*fraction)}%")
    axes[1].plot(x, reference, color="black", linewidth=1.6, linestyle="--", label="Published reference")
    axes[1].set_title("Cumulative trajectory convergence")
    for ax in axes:
        ax.set_xlabel("Membrane--peptide distance [nm]")
        ax.set_ylabel("Free Energy [kJ/mol]")
        ax.legend(frameon=False, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "wham_reference_convergence.pdf")
    fig.savefig(output / "wham_reference_convergence.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 5.1), constrained_layout=True)
    cmap = plt.cm.viridis
    scale = 0.8
    for i, (center, probability) in enumerate(zip(centers, probabilities)):
        normalized = probability / max(probability.max(), 1e-15)
        baseline = float(i)
        ax.fill_between(overlap_x, baseline, baseline + scale * normalized,
                        color=cmap(i / (centers.size - 1)), alpha=0.65, linewidth=0)
        ax.plot(overlap_x, baseline + scale * normalized,
                color=cmap(i / (centers.size - 1)), linewidth=0.7)
    ax.set_yticks(np.arange(centers.size)[::2], [f"{v:.2f}" for v in centers[::2]])
    ax.set_xlabel("Membrane--peptide distance [nm]")
    ax.set_ylabel("Umbrella center [nm]")
    ax.set_title("AR(1)-thinned full-data window histograms")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "autocorrelation_thinned_histogram_overlap.pdf")
    fig.savefig(output / "autocorrelation_thinned_histogram_overlap.png", dpi=300)
    plt.close(fig)

    # Single-column SI figure combining overlap and both convergence checks.
    fig, axes = plt.subplots(
        3, 1, figsize=(3.33, 7.25),
        gridspec_kw={"height_ratios": [1.25, 1.0, 1.0]},
        constrained_layout=True,
    )
    overlap_ax, block_ax, cumulative_ax = axes
    ridge_scale = 0.8
    for i, (center, probability) in enumerate(zip(centers, probabilities)):
        normalized = probability / max(probability.max(), 1e-15)
        baseline = float(i)
        color = cmap(i / (centers.size - 1))
        overlap_ax.fill_between(
            overlap_x, baseline, baseline + ridge_scale * normalized,
            color=color, alpha=0.65, linewidth=0,
        )
        overlap_ax.plot(
            overlap_x, baseline + ridge_scale * normalized,
            color=color, linewidth=0.55,
        )
    overlap_ax.set_yticks(
        np.arange(centers.size)[::4],
        [f"{value:.2f}" for value in centers[::4]],
    )
    overlap_ax.set_ylabel("Umbrella center [nm]")
    overlap_ax.set_title("AR(1)-thinned window histograms", pad=3)

    for i, color in enumerate(colors, start=1):
        block_ax.plot(
            x, surfaces[f"block_{i}"], color=color, linewidth=0.75,
            alpha=0.65, label=f"Block {i}",
        )
    block_ax.fill_between(
        x, block_mean - block_sd, block_mean + block_sd,
        color="#0072B2", alpha=0.18, linewidth=0,
    )
    block_ax.plot(x, block_mean, color="#0072B2", linewidth=1.25, label="Block mean")
    block_ax.plot(x, reference, color="black", linewidth=1.25, linestyle="--",
                  label="Reference")
    block_ax.set_ylabel("Free Energy [kJ/mol]")
    block_ax.set_title("Five contiguous trajectory blocks", pad=3)
    block_ax.legend(frameon=False, ncol=2, fontsize=6.2, handlelength=1.5,
                    columnspacing=0.8, labelspacing=0.25)

    for i, (fraction, color) in enumerate(zip(CUMULATIVE_FRACTIONS, colors), start=1):
        cumulative_ax.plot(
            x, surfaces[f"cumulative_{i}"], color=color, linewidth=0.9,
            label=f"{int(100 * fraction)}%",
        )
    cumulative_ax.plot(x, reference, color="black", linewidth=1.25,
                       linestyle="--", label="Reference")
    cumulative_ax.set_ylabel("Free Energy [kJ/mol]")
    cumulative_ax.set_title("Cumulative trajectory convergence", pad=3)
    cumulative_ax.legend(frameon=False, ncol=3, fontsize=6.2, handlelength=1.5,
                         columnspacing=0.8, labelspacing=0.25)

    for panel, ax in zip("abc", axes):
        ax.set_xlabel("Membrane--peptide distance [nm]")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="in")
        ax.text(-0.16, 1.04, rf"$\mathbf{{{panel}}}$", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.5)
    for extension in ("pdf", "png"):
        fig.savefig(
            output / f"reference_convergence_column.{extension}",
            dpi=600 if extension == "png" else None,
        )
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Wrote reference-convergence analysis to {output}")


if __name__ == "__main__":
    main()
