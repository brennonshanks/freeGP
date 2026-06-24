#!/usr/bin/env python3
"""Data-efficiency and failure-suppression analysis for ablation benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METHODS = [
    ("ui", "Umbrella Integration", "ui/ui/ablation_metrics.csv"),
    ("fixed", "Fixed Hyperparameters", "fixed/ablation_metrics.csv"),
    ("lml", "Hierarchical GP LML", "lml/ablation_metrics.csv"),
    ("loo", "Hierarchical GP LOO", "loo/ablation_metrics.csv"),
]

COLORS = {
    "Umbrella Integration": "tab:blue",
    "Fixed Hyperparameters": "tab:orange",
    "Hierarchical GP LML": "tab:red",
    "Hierarchical GP LOO": "tab:green",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify data efficiency and suppression of catastrophic RMSE "
            "failures across an ablation grid."
        )
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=5.0,
        help="RMSE threshold for an acceptable reconstruction.",
    )
    parser.add_argument(
        "--bad-rmse-threshold",
        type=float,
        default=20.0,
        help="RMSE threshold used to count catastrophic failures.",
    )
    parser.add_argument(
        "--acceptable-fraction-target",
        type=float,
        default=0.9,
        help=(
            "Target fraction of higher-data cells that must satisfy the RMSE "
            "criterion when estimating the practical data requirement."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to RESULTS_DIR/data_efficiency_failure.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method specs: tag:Label:relative_csv_path.",
    )
    return parser


def _parse_methods(values: list[str] | None) -> list[tuple[str, str, Path]]:
    if values is None:
        return [(tag, label, Path(path)) for tag, label, path in DEFAULT_METHODS]
    parsed = []
    for spec in values:
        pieces = spec.split(":", 2)
        if len(pieces) != 3:
            raise ValueError("Method specs must be tag:Label:relative_csv_path")
        parsed.append((pieces[0], pieces[1], Path(pieces[2])))
    return parsed


def _load_data(results_dir: Path, methods: list[tuple[str, str, Path]]) -> pd.DataFrame:
    frames = []
    for tag, label, rel_path in methods:
        csv_path = results_dir / rel_path
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        required = {"window_count", "trajectory_fraction", "rmse_wham", "avg_total_std"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")
        out = df[["window_count", "trajectory_fraction", "rmse_wham", "avg_total_std"]].copy()
        out["method_tag"] = tag
        out["method"] = label
        out["data_amount"] = (
            out["window_count"].to_numpy(dtype=float)
            * out["trajectory_fraction"].to_numpy(dtype=float)
        )
        out["cell_label"] = [
            f"w{int(w):02d}_f{float(f):.3g}"
            for w, f in zip(out["window_count"], out["trajectory_fraction"])
        ]
        frames.append(out)
    if not frames:
        raise FileNotFoundError("No ablation_metrics.csv files were found.")
    combined = pd.concat(frames, ignore_index=True)
    combined["data_fraction"] = combined["data_amount"] / combined["data_amount"].max()
    return combined.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["rmse_wham", "avg_total_std", "data_fraction"]
    )


def _threshold_summary(df: pd.DataFrame, *, epsilon: float, bad_rmse_threshold: float) -> pd.DataFrame:
    thresholds = np.array(sorted(df["data_fraction"].unique()))
    rows = []
    for method, method_df in df.groupby("method", sort=False):
        for threshold in thresholds:
            subset = method_df[method_df["data_fraction"] >= threshold]
            if subset.empty:
                continue
            rmse = subset["rmse_wham"].to_numpy(dtype=float)
            rows.append(
                {
                    "method": method,
                    "min_data_fraction": float(threshold),
                    "n_cells": int(len(subset)),
                    "acceptable_fraction": float(np.mean(rmse <= epsilon)),
                    "catastrophic_fraction": float(np.mean(rmse > bad_rmse_threshold)),
                    "mean_rmse": float(np.mean(rmse)),
                    "median_rmse": float(np.median(rmse)),
                    "p90_rmse": float(np.percentile(rmse, 90)),
                    "max_rmse": float(np.max(rmse)),
                    "mean_std": float(np.mean(subset["avg_total_std"])),
                    "median_std": float(np.median(subset["avg_total_std"])),
                }
            )
    return pd.DataFrame(rows)


def _method_summary(df: pd.DataFrame, *, epsilon: float, bad_rmse_threshold: float) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method", sort=False):
        rmse = group["rmse_wham"].to_numpy(dtype=float)
        acceptable = group[group["rmse_wham"] <= epsilon].sort_values(
            ["data_fraction", "window_count", "trajectory_fraction"]
        )
        first = acceptable.iloc[0] if not acceptable.empty else None
        rows.append(
            {
                "method": method,
                "n_cells": int(len(group)),
                "acceptable_fraction": float(np.mean(rmse <= epsilon)),
                "catastrophic_fraction": float(np.mean(rmse > bad_rmse_threshold)),
                "mean_rmse": float(np.mean(rmse)),
                "median_rmse": float(np.median(rmse)),
                "p90_rmse": float(np.percentile(rmse, 90)),
                "p95_rmse": float(np.percentile(rmse, 95)),
                "max_rmse": float(np.max(rmse)),
                "first_acceptable_data_fraction": np.nan if first is None else float(first["data_fraction"]),
                "first_acceptable_cell": "" if first is None else str(first["cell_label"]),
                "first_acceptable_rmse": np.nan if first is None else float(first["rmse_wham"]),
            }
        )
    return pd.DataFrame(rows)


def _low_data_summary(df: pd.DataFrame, *, epsilon: float, bad_rmse_threshold: float) -> pd.DataFrame:
    rows = []
    cutoffs = [0.05, 0.1, 0.2, 0.4]
    for method, group in df.groupby("method", sort=False):
        for cutoff in cutoffs:
            subset = group[group["data_fraction"] <= cutoff]
            if subset.empty:
                continue
            rmse = subset["rmse_wham"].to_numpy(dtype=float)
            rows.append(
                {
                    "method": method,
                    "max_data_fraction": cutoff,
                    "n_cells": int(len(subset)),
                    "acceptable_fraction": float(np.mean(rmse <= epsilon)),
                    "catastrophic_fraction": float(np.mean(rmse > bad_rmse_threshold)),
                    "median_rmse": float(np.median(rmse)),
                    "p90_rmse": float(np.percentile(rmse, 90)),
                    "max_rmse": float(np.max(rmse)),
                }
            )
    return pd.DataFrame(rows)


def _data_requirement_summary(
    threshold_summary: pd.DataFrame,
    *,
    epsilon: float,
    acceptable_fraction_target: float,
) -> pd.DataFrame:
    rows = []
    for method, group in threshold_summary.groupby("method", sort=False):
        group = group.sort_values("min_data_fraction")
        mostly_acceptable = group[group["acceptable_fraction"] >= acceptable_fraction_target]
        no_catastrophic = group[group["catastrophic_fraction"] == 0.0]
        p90_acceptable = group[group["p90_rmse"] <= epsilon]
        all_acceptable = group[group["max_rmse"] <= epsilon]
        rows.append(
            {
                "method": method,
                "target_acceptable_fraction": acceptable_fraction_target,
                "first_data_fraction_for_target_acceptable_fraction": (
                    np.nan
                    if mostly_acceptable.empty
                    else float(mostly_acceptable["min_data_fraction"].iloc[0])
                ),
                "first_data_fraction_with_no_catastrophic_failures": (
                    np.nan
                    if no_catastrophic.empty
                    else float(no_catastrophic["min_data_fraction"].iloc[0])
                ),
                "first_data_fraction_with_p90_rmse_below_epsilon": (
                    np.nan
                    if p90_acceptable.empty
                    else float(p90_acceptable["min_data_fraction"].iloc[0])
                ),
                "first_data_fraction_with_all_rmse_below_epsilon": (
                    np.nan if all_acceptable.empty else float(all_acceptable["min_data_fraction"].iloc[0])
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_efficiency(summary: pd.DataFrame, output_path: Path, *, epsilon: float, bad_rmse_threshold: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), sharex=True)
    specs = [
        ("acceptable_fraction", "Fraction RMSE <= epsilon", (0.0, 1.03)),
        ("catastrophic_fraction", f"Fraction RMSE > {bad_rmse_threshold:g}", (0.0, 1.03)),
        ("p90_rmse", "90th percentile RMSE", None),
        ("max_rmse", "Maximum RMSE", None),
    ]
    for ax, (column, label, ylim) in zip(axes.flat, specs):
        for method, group in summary.groupby("method", sort=False):
            ax.plot(
                group["min_data_fraction"],
                group[column],
                marker="o",
                ms=3.2,
                lw=1.8,
                color=COLORS.get(method),
                label=method,
            )
        if column in {"p90_rmse", "max_rmse"}:
            ax.axhline(epsilon, color="0.25", lw=1.0, ls=":")
        ax.set_ylabel(label)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("Minimum retained data fraction")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Data efficiency and failure suppression", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_rmse_distribution(df: pd.DataFrame, output_path: Path, *, epsilon: float, bad_rmse_threshold: float) -> None:
    methods = list(df["method"].drop_duplicates())
    values = [df.loc[df["method"] == method, "rmse_wham"].to_numpy(dtype=float) for method in methods]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    box = ax.boxplot(values, tick_labels=methods, patch_artist=True, showfliers=True)
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(COLORS.get(method, "0.6"))
        patch.set_alpha(0.55)
    ax.axhline(epsilon, color="0.25", lw=1.0, ls=":", label="epsilon")
    ax.axhline(bad_rmse_threshold, color="tab:red", lw=1.0, ls="--", label="catastrophic threshold")
    ax.set_ylabel("RMSE vs WHAM (kJ/mol)")
    ax.set_title("RMSE distribution over ablation grid")
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(df: pd.DataFrame, output_path: Path, *, epsilon: float) -> None:
    methods = list(df["method"].drop_duplicates())
    window_counts = sorted(df["window_count"].unique())
    traj_fractions = sorted(df["trajectory_fraction"].unique(), reverse=True)
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(3.7 * len(methods), 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        group = df[df["method"] == method]
        grid = np.full((len(traj_fractions), len(window_counts)), np.nan)
        for i, frac in enumerate(traj_fractions):
            for j, windows in enumerate(window_counts):
                rows = group[(group["trajectory_fraction"] == frac) & (group["window_count"] == windows)]
                if not rows.empty:
                    grid[i, j] = float(rows["rmse_wham"].iloc[0])
        im = ax.imshow(grid, aspect="auto", origin="upper", cmap="viridis", vmin=0.0)
        acceptable = grid <= epsilon
        if np.nanmin(acceptable.astype(float)) < 0.5 < np.nanmax(acceptable.astype(float)):
            ax.contour(acceptable, levels=[0.5], colors="white", linewidths=1.0)
        ax.set_title(method)
        ax.set_xticks(range(len(window_counts)))
        ax.set_xticklabels([str(int(w)) for w in window_counts], rotation=45, ha="right")
        ax.set_yticks(range(len(traj_fractions)))
        ax.set_yticklabels([f"{float(f) * 100:.3g}" for f in traj_fractions])
        ax.set_xlabel("Windows")
    axes[0].set_ylabel("Trajectory length (%)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("RMSE vs WHAM (kJ/mol)")
    fig.suptitle("Acceptable-region boundary over ablation grid")
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "data_efficiency_failure"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = _parse_methods(args.methods)
    data = _load_data(results_dir, methods)
    threshold_summary = _threshold_summary(
        data,
        epsilon=args.epsilon,
        bad_rmse_threshold=args.bad_rmse_threshold,
    )
    method_summary = _method_summary(
        data,
        epsilon=args.epsilon,
        bad_rmse_threshold=args.bad_rmse_threshold,
    )
    low_data_summary = _low_data_summary(
        data,
        epsilon=args.epsilon,
        bad_rmse_threshold=args.bad_rmse_threshold,
    )
    data_requirement_summary = _data_requirement_summary(
        threshold_summary,
        epsilon=args.epsilon,
        acceptable_fraction_target=args.acceptable_fraction_target,
    )

    data.to_csv(output_dir / "data_efficiency_by_cell.csv", index=False)
    threshold_summary.to_csv(output_dir / "data_efficiency_by_min_data.csv", index=False)
    method_summary.to_csv(output_dir / "data_efficiency_method_summary.csv", index=False)
    low_data_summary.to_csv(output_dir / "low_data_failure_summary.csv", index=False)
    data_requirement_summary.to_csv(output_dir / "data_requirement_summary.csv", index=False)
    _plot_efficiency(
        threshold_summary,
        output_dir / "data_efficiency_curves.png",
        epsilon=args.epsilon,
        bad_rmse_threshold=args.bad_rmse_threshold,
    )
    _plot_rmse_distribution(
        data,
        output_dir / "rmse_distribution.png",
        epsilon=args.epsilon,
        bad_rmse_threshold=args.bad_rmse_threshold,
    )
    _plot_heatmaps(
        data,
        output_dir / "rmse_grid_acceptability.png",
        epsilon=args.epsilon,
    )

    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
