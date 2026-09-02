#!/usr/bin/env python3
"""Prepare BayesWHAM inputs for every saved R9 ablation replicate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from prepare_r9_inputs import read_mdp, read_pullx, reference_grid


EQUILIBRATION_FRAMES = 40_000


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset = Path.home() / "freeGP-datasets/membranes/katka"
    archive = (
        Path.home()
        / "freeGP-v0.1.0/results/membrane-ablation-10x10-5replicates/loo/artifacts/cells"
    )
    reference = Path.home() / "freeGP-v0.1.0/reference_data/wham.dat"
    output_root = root / "r9_ablation_grid"
    centers, edges = reference_grid(reference)

    trajectories: dict[float, np.ndarray] = {}
    metadata: dict[float, float] = {}
    for window_dir in sorted(dataset.glob("d_*")):
        center = float(window_dir.name.removeprefix("d_"))
        trajectories[center] = read_pullx(
            window_dir / "step7_production_pullx.xvg"
        )[EQUILIBRATION_FRAMES:]
        metadata[center] = float(read_mdp(window_dir / "step7_production.mdp")["pull-coord1-k"])

    prepared = 0
    for artifact_path in sorted(archive.glob("w*_f*.pt")):
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        for replicate in artifact["bundle"]["replicates"]:
            processed = replicate["bundle"]["processed"]
            selected_centers = [float(x) for x in processed["folder_numbers"]]
            sample_counts = [int(x) for x in processed["n_samples"]]
            taus = [float(x) for x in processed["autocorr_times"]]
            rep_index = int(replicate["replicate_index"])
            output = output_root / artifact_path.stem / f"replicate_{rep_index + 1}" / "input"
            if (output / "manifest.json").exists():
                prepared += 1
                continue
            hist_dir = output / "hist"
            hist_dir.mkdir(parents=True, exist_ok=True)
            aggregate = np.zeros(centers.size, dtype=float)
            bias_rows = []
            records = []

            for index, (window_center, frame_count, tau) in enumerate(
                zip(selected_centers, sample_counts, taus), start=1
            ):
                positions = trajectories[window_center][:frame_count]
                if positions.size != frame_count:
                    raise ValueError(f"Insufficient frames for {window_center:.2f}")
                counts, _ = np.histogram(positions, bins=edges)
                effective_counts = counts.astype(float) / tau
                aggregate += effective_counts
                np.savetxt(hist_dir / f"hist_{index}.txt", effective_counts[None], fmt="%.12g")
                bias_rows.append((index, window_center, metadata[window_center]))
                records.append(
                    {
                        "index": index,
                        "center_nm": window_center,
                        "retained_frames": frame_count,
                        "ar1_autocorrelation_time_frames": tau,
                        "effective_histogrammed_frames": float(effective_counts.sum()),
                    }
                )

            np.savetxt(output / "harmonic_biases.txt", bias_rows, fmt=["%d", "%.8f", "%.8f"])
            np.savetxt(output / "hist_binEdges.txt", edges[None], fmt="%.8f")
            np.savetxt(output / "hist_binCenters.txt", centers[None], fmt="%.8f")
            np.savetxt(output / "aggregate_histogram.txt", aggregate[None], fmt="%.12g")
            manifest = {
                "archived_cell": artifact_path.name,
                "replicate_index": rep_index,
                "is_canonical": bool(replicate["is_canonical"]),
                "window_selection_mode": replicate["window_selection_mode"],
                "trajectory_selection_mode": replicate["trajectory_selection_mode"],
                "random_seed": int(replicate["random_seed"]),
                "count_treatment": "counts divided by archived per-window AR(1) autocorrelation times",
                "windows": records,
                "effective_histogrammed_frames_total": float(aggregate.sum()),
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            prepared += 1
    print(f"Prepared {prepared} archived ablation replicates under {output_root}")


if __name__ == "__main__":
    main()
