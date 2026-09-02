#!/usr/bin/env python3
"""Prepare three canonical R9 ablation cells for a BayesWHAM UQ pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from freegp.preprocess import bayes_autocorrelation_time
from prepare_r9_inputs import read_mdp, read_pullx, reference_grid


CELLS = {
    "sparse_w03_f0p10": "w03_f0p10.pt",
    "intermediate_w10_f0p25": "w10_f0p25.pt",
    "full_w25_f1p00": "w25_f1p00.pt",
}
EQUILIBRATION_FRAMES = 40_000


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset = Path.home() / "freeGP-datasets/membranes/katka"
    archive = (
        Path.home()
        / "freeGP-v0.1.0/results/membrane-ablation-10x10-5replicates/loo/artifacts/cells"
    )
    reference = Path.home() / "freeGP-v0.1.0/reference_data/wham.dat"
    centers, edges = reference_grid(reference)

    for label, artifact_name in CELLS.items():
        artifact = torch.load(archive / artifact_name, map_location="cpu", weights_only=False)
        replicate = artifact["bundle"]["replicates"][0]
        if not replicate["is_canonical"]:
            raise ValueError(f"First replicate is not canonical for {artifact_name}")
        processed = replicate["bundle"]["processed"]
        selected_centers = [float(x) for x in processed["folder_numbers"]]
        sample_counts = [int(x) for x in processed["n_samples"]]

        output = root / "r9_ablation_subset" / label / "effective_counts_ar1"
        hist_dir = output / "hist"
        hist_dir.mkdir(parents=True, exist_ok=True)
        bias_rows = []
        aggregate = np.zeros(centers.size, dtype=float)
        records = []

        for index, (window_center, frame_count) in enumerate(
            zip(selected_centers, sample_counts), start=1
        ):
            window_dir = dataset / f"d_{window_center:.2f}"
            mdp = read_mdp(window_dir / "step7_production.mdp")
            force_constant = float(mdp["pull-coord1-k"])
            positions = read_pullx(window_dir / "step7_production_pullx.xvg")
            positions = positions[
                EQUILIBRATION_FRAMES : EQUILIBRATION_FRAMES + frame_count
            ]
            if positions.size != frame_count:
                raise ValueError(f"Expected {frame_count} frames for {window_dir.name}")
            counts, _ = np.histogram(positions, bins=edges)
            tau = float(
                bayes_autocorrelation_time(
                    torch.as_tensor(positions, dtype=torch.float64)
                )
            )
            effective_counts = counts.astype(float) / tau
            aggregate += effective_counts
            np.savetxt(hist_dir / f"hist_{index}.txt", effective_counts[None], fmt="%.12g")
            bias_rows.append((index, window_center, force_constant))
            records.append(
                {
                    "index": index,
                    "center_nm": window_center,
                    "force_constant_kj_mol_nm2": force_constant,
                    "retained_frames": frame_count,
                    "histogrammed_frames": int(counts.sum()),
                    "ar1_autocorrelation_time_frames": tau,
                    "effective_histogrammed_frames": float(effective_counts.sum()),
                }
            )

        np.savetxt(output / "harmonic_biases.txt", bias_rows, fmt=["%d", "%.8f", "%.8f"])
        np.savetxt(output / "hist_binEdges.txt", edges[None], fmt="%.8f")
        np.savetxt(output / "hist_binCenters.txt", centers[None], fmt="%.8f")
        np.savetxt(output / "aggregate_histogram.txt", aggregate[None], fmt="%.12g")
        manifest = {
            "condition": label,
            "archived_cell": artifact_name,
            "replicate": "canonical evenly spaced selection (replicate 0)",
            "trajectory_selection": "first requested fraction after 40000-frame equilibration removal",
            "count_treatment": "histogram counts divided by per-window AR(1) autocorrelation time",
            "temperature_k": 303.15,
            "windows": records,
            "effective_histogrammed_frames_total": float(aggregate.sum()),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{label}: {len(records)} windows, {aggregate.sum():,.1f} effective counts")


if __name__ == "__main__":
    main()
