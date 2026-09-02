#!/usr/bin/env python3
"""Aggregate the completed R9 BayesWHAM grid into manuscript metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch


KT = 0.0083144621 * 303.15


def empirical_crps(draws: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pointwise ensemble CRPS without forming pairwise sample matrices."""
    ordered = np.sort(draws, axis=0)
    n = ordered.shape[0]
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    pair_term = np.sum(weights[:, None] * ordered, axis=0) / (n * n)
    return np.mean(np.abs(draws - truth[None, :]), axis=0) - pair_term


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def window_overlap_connected(input_dir: Path) -> bool:
    biases = np.loadtxt(input_dir / "harmonic_biases.txt")
    histograms = np.stack(
        [
            np.loadtxt(input_dir / "hist" / f"hist_{index}.txt")
            for index in range(1, len(biases) + 1)
        ]
    )
    occupied = histograms > 0
    adjacency = occupied.astype(int) @ occupied.astype(int).T > 0
    seen = {0}
    stack = [0]
    while stack:
        index = stack.pop()
        for neighbor in np.flatnonzero(adjacency[index]):
            neighbor = int(neighbor)
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(histograms)


def main() -> None:
    root = Path(__file__).resolve().parent
    grid = root / "r9_ablation_grid"
    archive = (
        Path.home()
        / "freeGP-v0.1.0/results/membrane-ablation-10x10-5replicates/loo/artifacts/cells"
    )
    reference = np.loadtxt(Path.home() / "freeGP-v0.1.0/reference_data/wham.dat")
    reference_f = reference[:, 1] - np.max(reference[:, 1])
    output = grid / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    cell_rows = []
    replicate_rows = []
    for cell_dir in sorted(grid.glob("w*_f*")):
        archived = torch.load(archive / f"{cell_dir.name}.pt", map_location="cpu", weights_only=False)
        window_count = int(archived["cell"]["window_count"])
        fraction = float(archived["cell"]["trajectory_fraction"])
        per_rep = []
        for replicate_dir in sorted(cell_dir.glob("replicate_*")):
            connected = window_overlap_connected(replicate_dir / "input")
            candidates = []
            for candidate_name in ("nuts_refined_fast", "nuts"):
                candidate_diagnostics = replicate_dir / "uq" / candidate_name / "diagnostics.json"
                if candidate_diagnostics.is_file():
                    candidate = json.loads(candidate_diagnostics.read_text())
                    candidate_divergences = sum(
                        len(value) for value in candidate["divergences"].values()
                    )
                    candidate_passes = (
                        float(candidate["max_r_hat"]) <= 1.05
                        and float(candidate["min_effective_sample_size"]) >= 100.0
                        and candidate_divergences == 0
                    )
                    candidates.append((candidate_name, candidate_passes))
            passing_name = next((name for name, passes in candidates if passes), None)
            result_name = passing_name or candidates[0][0]
            # Inclusion is based only on the predeclared MCMC diagnostics.
            # Window overlap is retained as descriptive metadata and does not
            # determine whether a replicate enters the analysis.
            included = passing_name is not None
            posterior_path = replicate_dir / "uq" / result_name / "posterior_samples.npz"
            diagnostics_path = replicate_dir / "uq" / result_name / "diagnostics.json"
            if not posterior_path.is_file() or not diagnostics_path.is_file():
                raise FileNotFoundError(f"Incomplete grid result: {replicate_dir}")
            posterior = np.load(posterior_path)
            x = np.asarray(posterior["bin_centers_nm"], dtype=float)
            draws = posterior["beta_f_grouped"].reshape(-1, x.size) * KT
            draws -= np.max(draws, axis=1, keepdims=True)
            mean = draws.mean(axis=0)
            sd = draws.std(axis=0, ddof=1)
            truth = np.interp(x, reference[:, 0], reference_f)
            lower68, upper68 = np.quantile(draws, [0.16, 0.84], axis=0)
            lower95, upper95 = np.quantile(draws, [0.025, 0.975], axis=0)
            diagnostics = json.loads(diagnostics_path.read_text())
            values = {
                "rmse_wham": float(np.sqrt(np.mean((mean - truth) ** 2))),
                "avg_total_std": float(np.mean(sd)),
                "avg_total_variance": float(np.mean(sd**2)),
                "mean_crps_kj_mol": float(np.mean(empirical_crps(draws, truth))),
                "wham_inclusion_68": float(np.mean((truth >= lower68) & (truth <= upper68))),
                "wham_inclusion_95": float(np.mean((truth >= lower95) & (truth <= upper95))),
                "max_r_hat": float(diagnostics["max_r_hat"]),
                "min_effective_sample_size": float(diagnostics["min_effective_sample_size"]),
                "divergence_count": int(sum(len(v) for v in diagnostics["divergences"].values())),
            }
            rep_index = int(replicate_dir.name.split("_")[-1])
            replicate_rows.append(
                {"window_count": window_count, "trajectory_fraction": fraction,
                 "replicate": rep_index, "result_source": result_name,
                 "window_overlap_connected": connected,
                 "included_in_valid_only_analysis": included,
                 "exclusion_reason": "" if included else "failed_sampler_diagnostics",
                 **values}
            )
            if included:
                per_rep.append(values)

        row = {
            "window_count": window_count,
            "trajectory_fraction": fraction,
            "replicate_count": len(archived["bundle"]["replicates"]),
            "valid_replicate_count": len(per_rep),
        }
        for metric in (
            "rmse_wham", "avg_total_std", "avg_total_variance", "mean_crps_kj_mol",
            "wham_inclusion_68", "wham_inclusion_95",
        ):
            row[metric] = float(np.mean([rep[metric] for rep in per_rep])) if per_rep else float("nan")
        row["mcmc_max_r_hat"] = float(max(rep["max_r_hat"] for rep in per_rep)) if per_rep else float("nan")
        row["mcmc_min_n_eff"] = float(min(rep["min_effective_sample_size"] for rep in per_rep)) if per_rep else float("nan")
        row["mcmc_divergence_total"] = int(sum(rep["divergence_count"] for rep in per_rep)) if per_rep else 0
        cell_rows.append(row)

    cell_rows.sort(key=lambda row: (int(row["window_count"]), float(row["trajectory_fraction"])))
    replicate_rows.sort(key=lambda row: (int(row["window_count"]), float(row["trajectory_fraction"]), int(row["replicate"])))
    write_csv(output / "ablation_metrics.csv", cell_rows)
    write_csv(output / "replicate_metrics.csv", replicate_rows)
    diagnostic_failures = []
    failure_counts: dict[tuple[int, float], int] = {}
    for row in replicate_rows:
        bad_r_hat = float(row["max_r_hat"]) > 1.05
        bad_ess = float(row["min_effective_sample_size"]) < 100.0
        if not row["included_in_valid_only_analysis"]:
            diagnostic_failures.append(
                {
                    **row,
                    "r_hat_above_1p05": bad_r_hat,
                    "ess_below_100": bad_ess,
                    "failure_reason": row["exclusion_reason"],
                }
            )
            key = (int(row["window_count"]), float(row["trajectory_fraction"]))
            failure_counts[key] = failure_counts.get(key, 0) + 1
    failure_cells = [
        {
            "window_count": row["window_count"],
            "trajectory_fraction": row["trajectory_fraction"],
            "replicate_count": row["replicate_count"],
            "flagged_replicates": failure_counts.get(
                (int(row["window_count"]), float(row["trajectory_fraction"])), 0
            ),
            "cell_max_r_hat": row["mcmc_max_r_hat"],
            "cell_min_effective_sample_size": row["mcmc_min_n_eff"],
        }
        for row in cell_rows
        if failure_counts.get(
            (int(row["window_count"]), float(row["trajectory_fraction"])), 0
        )
        > 0
    ]
    write_csv(output / "diagnostic_failures.csv", diagnostic_failures)
    write_csv(output / "diagnostic_failures_by_cell.csv", failure_cells)

    rmse = np.asarray([row["rmse_wham"] for row in cell_rows])
    uncertainty = np.asarray([row["avg_total_std"] for row in cell_rows])
    finite = np.isfinite(rmse) & np.isfinite(uncertainty)
    threshold = 5.0
    summary = {
        "cells": len(cell_rows),
        "replicate_calculations": len(replicate_rows),
        "cells_with_valid_replicates": int(finite.sum()),
        "cells_masked_no_valid_replicates": int((~finite).sum()),
        "valid_replicate_calculations": int(sum(row["valid_replicate_count"] for row in cell_rows)),
        "pearson_r_rmse_vs_mean_sd": float(np.corrcoef(rmse[finite], uncertainty[finite])[0, 1]),
        "mean_crps_kj_mol": float(np.nanmean([row["mean_crps_kj_mol"] for row in cell_rows])),
        "mean_wham_inclusion_68_percent": 100.0 * float(np.nanmean([row["wham_inclusion_68"] for row in cell_rows])),
        "mean_wham_inclusion_95_percent": 100.0 * float(np.nanmean([row["wham_inclusion_95"] for row in cell_rows])),
        "underconfident_cells_rmse_le_5_sd_gt_5": int(np.sum(finite & (rmse <= threshold) & (uncertainty > threshold))),
        "overconfident_cells_rmse_gt_5_sd_le_5": int(np.sum(finite & (rmse > threshold) & (uncertainty <= threshold))),
        "maximum_r_hat": float(np.nanmax([row["mcmc_max_r_hat"] for row in cell_rows])),
        "minimum_effective_sample_size": float(np.nanmin([row["mcmc_min_n_eff"] for row in cell_rows])),
        "total_divergences": int(sum(row["mcmc_divergence_total"] for row in cell_rows)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
