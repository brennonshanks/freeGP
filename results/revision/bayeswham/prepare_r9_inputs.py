#!/usr/bin/env python3
"""Prepare full-dataset R9 histograms for the original BayesWHAM interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch

from freegp.preprocess import bayes_autocorrelation_time


WINDOW_PATTERN = re.compile(r"d_([0-9]+\.[0-9]+)$")


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home() / "freeGP-datasets/membranes/katka",
    )
    p.add_argument(
        "--wham-reference",
        type=Path,
        default=Path.home() / "freeGP-v0.1.0/reference_data/wham.dat",
    )
    p.add_argument("--output", type=Path, default=root / "r9_full/raw_counts")
    p.add_argument(
        "--effective-output",
        type=Path,
        default=root / "r9_full/effective_counts_ar1",
    )
    p.add_argument("--equilibration-frames", type=int, default=40_000)
    p.add_argument("--temperature-k", type=float, default=303.15)
    return p


def read_mdp(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if line and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_pullx(path: Path) -> np.ndarray:
    positions: list[float] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(("#", "@")):
                continue
            fields = line.split()
            if len(fields) >= 2:
                positions.append(float(fields[1]))
    return np.asarray(positions, dtype=float)


def reference_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    centers = np.loadtxt(path, comments="#", usecols=0)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError(f"Invalid WHAM reference grid: {path}")
    steps = np.diff(centers)
    if not np.allclose(steps, steps[0]):
        raise ValueError("The R9 WHAM reference grid is not uniformly spaced.")
    half_step = steps[0] / 2.0
    edges = np.concatenate(([centers[0] - half_step], centers + half_step))
    return centers, edges


def main() -> None:
    args = parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    reference = args.wham_reference.expanduser().resolve()
    output = args.output.expanduser().resolve()
    effective_output = args.effective_output.expanduser().resolve()
    hist_dir = output / "hist"
    effective_hist_dir = effective_output / "hist"
    hist_dir.mkdir(parents=True, exist_ok=True)
    effective_hist_dir.mkdir(parents=True, exist_ok=True)

    centers, edges = reference_grid(reference)
    window_dirs: list[tuple[float, Path]] = []
    for path in dataset_root.iterdir():
        match = WINDOW_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            window_dirs.append((float(match.group(1)), path))
    window_dirs.sort()
    if len(window_dirs) != 25:
        raise ValueError(f"Expected 25 R9 windows, found {len(window_dirs)}.")

    bias_rows: list[tuple[int, float, float]] = []
    window_records: list[dict[str, object]] = []
    aggregate = np.zeros(centers.size, dtype=np.int64)
    effective_aggregate = np.zeros(centers.size, dtype=float)

    for index, (folder_center, window_dir) in enumerate(window_dirs, start=1):
        mdp_path = window_dir / "step7_production.mdp"
        pullx_path = window_dir / "step7_production_pullx.xvg"
        mdp = read_mdp(mdp_path)
        mdp_center = float(mdp["pull-coord1-init"])
        force_constant = float(mdp["pull-coord1-k"])
        temperature_values = [float(x) for x in mdp["ref_t"].split()]
        if not np.isclose(folder_center, mdp_center):
            raise ValueError(f"Center mismatch in {window_dir.name}.")
        if not all(np.isclose(x, args.temperature_k) for x in temperature_values):
            raise ValueError(f"Temperature mismatch in {window_dir.name}.")

        raw_positions = read_pullx(pullx_path)
        if raw_positions.size <= args.equilibration_frames:
            raise ValueError(f"Not enough frames in {pullx_path}.")
        positions = raw_positions[args.equilibration_frames :]
        counts, _ = np.histogram(positions, bins=edges)
        autocorrelation_time = bayes_autocorrelation_time(
            torch.as_tensor(positions, dtype=torch.float64)
        )
        effective_counts = counts.astype(float) / autocorrelation_time
        included = int(counts.sum())
        excluded = int(positions.size - included)
        aggregate += counts
        effective_aggregate += effective_counts
        np.savetxt(hist_dir / f"hist_{index}.txt", counts[None, :], fmt="%d")
        np.savetxt(
            effective_hist_dir / f"hist_{index}.txt",
            effective_counts[None, :],
            fmt="%.12g",
        )
        bias_rows.append((index, mdp_center, force_constant))
        window_records.append(
            {
                "index": index,
                "directory": window_dir.name,
                "center_nm": mdp_center,
                "force_constant_kj_mol_nm2": force_constant,
                "raw_frames": int(raw_positions.size),
                "equilibration_frames_removed": args.equilibration_frames,
                "retained_frames": int(positions.size),
                "histogrammed_frames": included,
                "outside_histogram_range": excluded,
                "ar1_autocorrelation_time_frames": autocorrelation_time,
                "effective_histogrammed_frames": float(effective_counts.sum()),
            }
        )

    np.savetxt(
        output / "harmonic_biases.txt",
        np.asarray(bias_rows),
        fmt=["%d", "%.8f", "%.8f"],
    )
    np.savetxt(output / "hist_binEdges.txt", edges[None, :], fmt="%.8f")
    np.savetxt(output / "hist_binCenters.txt", centers[None, :], fmt="%.8f")
    np.savetxt(output / "aggregate_histogram.txt", aggregate[None, :], fmt="%d")
    np.savetxt(
        effective_output / "harmonic_biases.txt",
        np.asarray(bias_rows),
        fmt=["%d", "%.8f", "%.8f"],
    )
    np.savetxt(effective_output / "hist_binEdges.txt", edges[None, :], fmt="%.8f")
    np.savetxt(effective_output / "hist_binCenters.txt", centers[None, :], fmt="%.8f")
    np.savetxt(
        effective_output / "aggregate_histogram.txt",
        effective_aggregate[None, :],
        fmt="%.12g",
    )

    manifest = {
        "description": "Raw-count BayesWHAM inputs for the full 25-window R9 dataset",
        "dataset_root": str(dataset_root),
        "wham_reference": str(reference),
        "dimensionality": 1,
        "periodicity": [0],
        "temperature_k": args.temperature_k,
        "equilibration_frames_removed": args.equilibration_frames,
        "histogram_bins": int(centers.size),
        "histogram_range_nm": [float(edges[0]), float(edges[-1])],
        "histogram_centers_match_wham_reference": True,
        "count_treatment": "raw correlated trajectory frames; suitable for MAP validation, not final UQ",
        "windows": window_records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    effective_manifest = dict(manifest)
    effective_manifest.update(
        {
            "description": "AR(1) effective-count BayesWHAM inputs for the full 25-window R9 dataset",
            "count_treatment": (
                "per-bin raw counts divided by the window-specific AR(1) "
                "autocorrelation time from freegp.preprocess.bayes_autocorrelation_time"
            ),
            "autocorrelation_estimator": "freegp.preprocess.bayes_autocorrelation_time",
            "effective_histogrammed_frames_total": float(effective_aggregate.sum()),
        }
    )
    (effective_output / "manifest.json").write_text(
        json.dumps(effective_manifest, indent=2) + "\n"
    )

    print(f"Wrote {len(window_records)} windows and {centers.size} bins to {output}")
    print(f"Histogrammed frames: {int(aggregate.sum()):,}")
    print(f"Frames outside grid: {sum(int(x['outside_histogram_range']) for x in window_records):,}")
    print(f"Wrote AR(1) effective counts to {effective_output}")
    print(f"Effective histogrammed frames: {float(effective_aggregate.sum()):,.1f}")


if __name__ == "__main__":
    main()
