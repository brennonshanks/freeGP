#!/usr/bin/env python3
"""Analyze agreement between predicted uncertainty and reconstruction error.

The script reads saved ``ablation_metrics.csv`` files; it does not rerun any
free-energy calculations.  For each method it computes Pearson's correlation
between average predictive standard deviation (``avg_total_std``) and RMSE to
the WHAM reference (``rmse_wham``):

1. separately at each window count, across trajectory fractions; and
2. across all window-count/trajectory-fraction cells.

It writes the numerical results to CSV and produces a correlation-versus-window
plot and an all-cell scatter plot.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MethodSpec:
    label: str
    relative_csv: Path
    color: str
    marker: str


METHODS = {
    "ui": MethodSpec(
        "UI + block averaging",
        Path("ui/ui/ablation_metrics.csv"),
        "#999999",
        "o",
    ),
    "fixed": MethodSpec("Fixed GP", Path("fixed/ablation_metrics.csv"), "#0072B2", "s"),
    "loo_map": MethodSpec(
        "Optimized GP (LOO)", Path("loo_map/ablation_metrics.csv"), "#E69F00", "^"
    ),
    "loo": MethodSpec(
        "Hierarchical GP (LOO)", Path("loo/ablation_metrics.csv"), "#009E73", "D"
    ),
    "lml_map": MethodSpec(
        "Optimized GP (LML)", Path("lml_map/ablation_metrics.csv"), "#CC79A7", "v"
    ),
    "lml": MethodSpec(
        "Hierarchical GP (LML)", Path("lml/ablation_metrics.csv"), "#56B4E9", "P"
    ),
}
DEFAULT_METHODS = ["ui", "fixed", "loo_map", "loo"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/membrane-ablation-10x10-5replicates"),
        help="Root directory containing the method-specific ablation CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revision/coverage/ablation_pearson"),
        help="Directory for the summary CSV and figures.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS),
        default=DEFAULT_METHODS,
        help="Ablation methods to analyze (defaults to the four main-text methods).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=5.0,
        help="RMSE and average-SD threshold used for convergence quadrants [kJ/mol].",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Figure formats to save.",
    )
    return parser


def read_metrics(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Ablation metrics not found: {path}")
    rows: list[dict[str, float]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"window_count", "trajectory_fraction", "rmse_wham", "avg_total_std"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for raw in reader:
            row = {name: float(raw[name]) for name in required}
            if all(np.isfinite(list(row.values()))):
                rows.append(row)
    if not rows:
        raise ValueError(f"No finite ablation metrics found in {path}")
    return rows


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson's r, or NaN when fewer than two variable pairs exist."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def normal_crps(mean: np.ndarray, std: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """CRPS for univariate Gaussian marginals, with deterministic-limit handling."""
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    observation = np.asarray(observation, dtype=float)
    result = np.abs(mean - observation)
    positive = np.isfinite(std) & (std > 0.0)
    if np.any(positive):
        z = (observation[positive] - mean[positive]) / std[positive]
        phi = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        result[positive] = std[positive] * (
            z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi)
        )
    return result


def cell_stem(window_count: float, trajectory_fraction: float) -> str:
    fraction = f"{trajectory_fraction:.2f}".replace(".", "p")
    return f"w{int(window_count):02d}_f{fraction}"


def add_probabilistic_scores(
    data: dict[str, list[dict[str, float]]], results_dir: Path
) -> list[dict[str, object]]:
    """Compute CRPS and WHAM-reference interval inclusion for every cell."""
    reference_path = results_dir / "loo" / "artifacts" / "references.npz"
    if not reference_path.is_file():
        reference_path = results_dir / "fixed" / "artifacts" / "references.npz"
    references = np.load(reference_path)
    wham_x = np.asarray(references["wham_x"], dtype=float).reshape(-1)
    wham_f = np.asarray(references["wham_f"], dtype=float).reshape(-1)
    wham_f = wham_f - np.max(wham_f)

    crps_rows: list[dict[str, object]] = []
    for method, rows in data.items():
        for row in rows:
            stem = cell_stem(row["window_count"], row["trajectory_fraction"])
            if method == "ui":
                curve_path = results_dir / METHODS[method].relative_csv.parent / "curves" / f"{stem}.csv"
                with curve_path.open(newline="") as handle:
                    curve = list(csv.DictReader(handle))
                x = np.asarray([float(item["x_nm"]) for item in curve])
                mean = np.asarray([float(item["pmf_mean_kJmol"]) for item in curve])
                std = np.asarray([float(item["pmf_std_kJmol"]) for item in curve])
            else:
                import torch

                artifact_path = results_dir / METHODS[method].relative_csv.parent / "artifacts" / "cells" / f"{stem}.pt"
                artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
                summary = artifact["predictive_summary"]
                x = summary["x_test"].detach().cpu().numpy().reshape(-1)
                mean = summary["mean"].detach().cpu().numpy().reshape(-1)
                variance = summary["total_variance"].detach().cpu().numpy().reshape(-1)
                std = np.sqrt(np.clip(variance, 0.0, None))

            mean = mean - np.max(mean)
            truth = np.interp(x, wham_x, wham_f)
            value = float(np.mean(normal_crps(mean, std, truth)))
            inclusion_68 = float(np.mean(np.abs(truth - mean) <= std))
            inclusion_95 = float(np.mean(np.abs(truth - mean) <= 2.0 * std))
            row["mean_crps"] = value
            row["inclusion_68"] = inclusion_68
            row["inclusion_95"] = inclusion_95
            crps_rows.append(
                {
                    "method": method,
                    "method_label": METHODS[method].label,
                    "window_count": int(row["window_count"]),
                    "trajectory_fraction": row["trajectory_fraction"],
                    "mean_crps_kj_mol": value,
                    "wham_inclusion_68": inclusion_68,
                    "wham_inclusion_95": inclusion_95,
                    "profile_points": int(x.size),
                }
            )
    return crps_rows


def calculate_correlations(
    data: dict[str, list[dict[str, float]]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for method, rows in data.items():
        windows = sorted({int(row["window_count"]) for row in rows})
        for window_count in windows:
            subset = [row for row in rows if int(row["window_count"]) == window_count]
            rmse = np.asarray([row["rmse_wham"] for row in subset])
            avg_std = np.asarray([row["avg_total_std"] for row in subset])
            summaries.append(
                {
                    "method": method,
                    "method_label": METHODS[method].label,
                    "scope": "window_count",
                    "window_count": window_count,
                    "n": len(subset),
                    "pearson_r": pearson_r(avg_std, rmse),
                }
            )

        rmse = np.asarray([row["rmse_wham"] for row in rows])
        avg_std = np.asarray([row["avg_total_std"] for row in rows])
        summaries.append(
            {
                "method": method,
                "method_label": METHODS[method].label,
                "scope": "all_cells",
                "window_count": "",
                "n": len(rows),
                "pearson_r": pearson_r(avg_std, rmse),
            }
        )
    return summaries


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "method_label", "scope", "window_count", "n", "pearson_r"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_probabilistic_scores(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "method_label",
                "window_count",
                "trajectory_fraction",
                "mean_crps_kj_mol",
                "wham_inclusion_68",
                "wham_inclusion_95",
                "profile_points",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_method_summary(
    data: dict[str, list[dict[str, float]]], path: Path
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for method, rows in data.items():
        summaries.append(
            {
                "method": method,
                "method_label": METHODS[method].label,
                "ablation_cells": len(rows),
                "wham_inclusion_68_percent": 100.0
                * float(np.mean([row["inclusion_68"] for row in rows])),
                "wham_inclusion_95_percent": 100.0
                * float(np.mean([row["inclusion_95"] for row in rows])),
                "mean_crps_kj_mol": float(np.mean([row["mean_crps"] for row in rows])),
                "median_crps_kj_mol": float(np.median([row["mean_crps"] for row in rows])),
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt.lstrip('.')}", dpi=dpi, bbox_inches="tight")


def plot_by_window(
    summaries: list[dict[str, object]],
    methods: list[str],
    output_dir: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for method in methods:
        spec = METHODS[method]
        rows = [
            row
            for row in summaries
            if row["method"] == method and row["scope"] == "window_count"
        ]
        rows.sort(key=lambda row: int(row["window_count"]))
        ax.plot(
            [int(row["window_count"]) for row in rows],
            [float(row["pearson_r"]) for row in rows],
            color=spec.color,
            marker=spec.marker,
            markersize=4.5,
            linewidth=1.4,
            label=spec.label,
        )
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.7)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Number of umbrella windows")
    ax.set_ylabel("Pearson $r$ (average SD vs RMSE)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    save_figure(fig, output_dir, "pearson_by_window_count", formats, dpi)
    plt.close(fig)


def plot_all_cells(
    data: dict[str, list[dict[str, float]]],
    summaries: list[dict[str, object]],
    methods: list[str],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    accuracy_threshold: float,
) -> None:
    ncols = 2
    nrows = (len(methods) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.5, 3.2 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    all_window_counts = sorted(
        {int(row["window_count"]) for rows in data.values() for row in rows}
    )
    norm = plt.Normalize(min(all_window_counts), max(all_window_counts))
    all_values = np.asarray(
        [
            value
            for rows in data.values()
            for row in rows
            for value in (row["rmse_wham"], row["avg_total_std"])
        ],
        dtype=float,
    )
    positive_values = all_values[np.isfinite(all_values) & (all_values > 0.0)]
    if positive_values.size == 0:
        raise ValueError("Parity plots require at least one positive RMSE or SD value.")
    axis_min = 0.8 * float(np.nanmin(positive_values))
    axis_max = 1.25 * float(np.nanmax(positive_values))
    scatter = None
    for ax, method in zip(axes.flat, methods):
        rows = data[method]
        rmse = np.asarray([row["rmse_wham"] for row in rows])
        avg_std = np.asarray([row["avg_total_std"] for row in rows])
        window_count = np.asarray([row["window_count"] for row in rows])
        scatter = ax.scatter(
            rmse,
            avg_std,
            c=window_count,
            cmap="viridis",
            norm=norm,
            s=18,
            alpha=0.8,
            linewidths=0.2,
            edgecolors="white",
        )
        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.8,
            label="$y=x$",
        )
        ax.axvline(
            accuracy_threshold,
            color="#666666",
            linestyle=":",
            linewidth=0.8,
        )
        ax.axhline(
            accuracy_threshold,
            color="#666666",
            linestyle=":",
            linewidth=0.8,
        )
        ax.fill_between(
            [axis_min, accuracy_threshold],
            accuracy_threshold,
            axis_max,
            color="#E69F00",
            alpha=0.07,
            zorder=0,
        )
        ax.fill_between(
            [accuracy_threshold, axis_max],
            axis_min,
            accuracy_threshold,
            color="#D55E00",
            alpha=0.07,
            zorder=0,
        )
        summary = next(
            row
            for row in summaries
            if row["method"] == method and row["scope"] == "all_cells"
        )
        mean_crps = float(np.mean([row["mean_crps"] for row in rows]))
        underconfident = sum(
            row["rmse_wham"] <= accuracy_threshold
            and row["avg_total_std"] > accuracy_threshold
            for row in rows
        )
        overconfident = sum(
            row["rmse_wham"] > accuracy_threshold
            and row["avg_total_std"] <= accuracy_threshold
            for row in rows
        )
        ax.set_title(
            f"{METHODS[method].label} ($r$ = {float(summary['pearson_r']):.3f})\n"
            f"mean CRPS = {mean_crps:.2f} kJ/mol",
            fontsize=10,
        )
        ax.text(
            0.03,
            0.97,
            f"Accurate, SD > {accuracy_threshold:g}: {underconfident}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#A55F00",
        )
        ax.text(
            0.97,
            0.03,
            f"Inaccurate, SD ≤ {accuracy_threshold:g}: {overconfident}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#A64000",
        )
        ax.set_xlabel("RMSE to WHAM [kJ/mol]")
        ax.set_ylabel("Average predicted SD [kJ/mol]")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2, linewidth=0.5)
    for ax in axes.flat[len(methods) :]:
        ax.set_visible(False)
    if scatter is not None:
        fig.colorbar(
            scatter,
            ax=list(axes.flat[: len(methods)]),
            label="Umbrella windows",
            shrink=0.82,
            pad=0.02,
        )
    save_figure(fig, output_dir, "rmse_vs_average_sd_all_cells", formats, dpi)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        method: read_metrics(results_dir / METHODS[method].relative_csv)
        for method in args.methods
    }
    summaries = calculate_correlations(data)
    score_rows = add_probabilistic_scores(data, results_dir)
    summary_path = output_dir / "pearson_correlations.csv"
    write_summary(summaries, summary_path)
    scores_path = output_dir / "probabilistic_scores_by_cell.csv"
    write_probabilistic_scores(score_rows, scores_path)
    method_summary_path = output_dir / "probabilistic_method_summary.csv"
    method_summaries = write_method_summary(data, method_summary_path)
    plot_by_window(summaries, args.methods, output_dir, args.formats, args.dpi)
    plot_all_cells(
        data,
        summaries,
        args.methods,
        output_dir,
        args.formats,
        args.dpi,
        args.accuracy_threshold,
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {scores_path}")
    print(f"Wrote {method_summary_path}")
    for row in method_summaries:
        print(
            f"{row['method_label']}: 68% inclusion = "
            f"{row['wham_inclusion_68_percent']:.2f}%, 95% inclusion = "
            f"{row['wham_inclusion_95_percent']:.2f}%, mean CRPS = "
            f"{row['mean_crps_kj_mol']:.3f} kJ/mol"
        )
    for method in args.methods:
        row = next(
            item
            for item in summaries
            if item["method"] == method and item["scope"] == "all_cells"
        )
        print(f"{METHODS[method].label}: r = {float(row['pearson_r']):.6f} (n = {row['n']})")


if __name__ == "__main__":
    main()
