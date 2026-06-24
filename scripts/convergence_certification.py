#!/usr/bin/env python3
"""Convergence-certification analysis for free-energy UQ benchmarks.

This is not a calibration test. It asks whether convergence of the reported
uncertainty to its final scale certifies convergence of the predictive mean.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


DEFAULT_DELTA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5]
DEFAULT_METHODS = [
    ("ui", "Umbrella Integration", "ui/ui/ablation_metrics.csv"),
    ("fixed", "Fixed Hyperparameters", "fixed/ablation_metrics.csv"),
    ("lml", "Hierarchical GP LML", "lml/ablation_metrics.csv"),
    ("loo", "Hierarchical GP LOO", "loo/ablation_metrics.csv"),
]
CATEGORY_LABELS = {
    0: "neither",
    1: "accurate_only",
    2: "u_scale_converged_only",
    3: "accurate_and_u_scale_converged",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether uncertainty convergence certifies predictive-mean "
            "convergence for an ablation benchmark."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Ablation result directory containing method subdirectories.",
    )
    parser.add_argument(
        "--epsilon",
        required=True,
        help=(
            "Accuracy threshold in kJ/mol, or 'kBT'. This is intentionally "
            "required so the scientific criterion is explicit."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature used when --epsilon kBT is requested.",
    )
    parser.add_argument(
        "--delta-u",
        type=float,
        default=0.2,
        help="Relative uncertainty convergence tolerance.",
    )
    parser.add_argument(
        "--delta-u-grid",
        default=",".join(str(v) for v in DEFAULT_DELTA_GRID),
        help="Comma-separated delta_U values for sensitivity analysis.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional method specs: tag:Label:relative_csv_path. Defaults to "
            "UI, fixed, LML, and LOO when present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to RESULTS_DIR/convergence_certification.",
    )
    return parser


def _parse_epsilon(value: str, *, temperature: float) -> float:
    if value.lower() == "kbt":
        return 8.3144621e-3 * temperature
    epsilon = float(value)
    if epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    return epsilon


def _parse_delta_grid(value: str) -> list[float]:
    deltas = [float(piece) for piece in value.split(",") if piece.strip()]
    if not deltas:
        raise ValueError("--delta-u-grid must contain at least one value")
    if any(delta <= 0 for delta in deltas):
        raise ValueError("All delta_U values must be positive")
    return deltas


def _parse_methods(values: list[str] | None) -> list[tuple[str, str, Path]]:
    if values is None:
        return [(tag, label, Path(rel)) for tag, label, rel in DEFAULT_METHODS]

    parsed: list[tuple[str, str, Path]] = []
    for spec in values:
        pieces = spec.split(":", 2)
        if len(pieces) != 3:
            raise ValueError(
                "Method specs must have form tag:Label:relative_csv_path"
            )
        tag, label, rel_path = pieces
        parsed.append((tag, label, Path(rel_path)))
    return parsed


def _load_method_data(
    results_dir: Path,
    methods: list[tuple[str, str, Path]],
) -> pd.DataFrame:
    frames = []
    for tag, label, rel_path in methods:
        csv_path = results_dir / rel_path
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        required = {"window_count", "trajectory_fraction", "rmse_wham"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")
        if "avg_total_std" in df.columns:
            uncertainty = df["avg_total_std"].to_numpy(dtype=float)
        elif "avg_total_variance" in df.columns:
            uncertainty = np.sqrt(
                np.clip(df["avg_total_variance"].to_numpy(dtype=float), 0.0, None)
            )
        else:
            raise ValueError(
                f"{csv_path} must contain avg_total_std or avg_total_variance"
            )

        out = df[["window_count", "trajectory_fraction", "rmse_wham"]].copy()
        out["method_tag"] = tag
        out["method"] = label
        out["data_amount"] = (
            out["window_count"].to_numpy(dtype=float)
            * out["trajectory_fraction"].to_numpy(dtype=float)
        )
        out["uncertainty"] = uncertainty
        out["cell_label"] = [
            f"w{int(w):02d}_f{float(f):.3g}"
            for w, f in zip(out["window_count"], out["trajectory_fraction"])
        ]
        frames.append(out)

    if not frames:
        raise FileNotFoundError("No method metrics CSVs were found.")

    combined = pd.concat(frames, ignore_index=True)
    max_data = float(combined["data_amount"].max())
    combined["data_fraction"] = combined["data_amount"] / max_data
    return combined.replace([np.inf, -np.inf], np.nan)


def _add_convergence_columns(
    df: pd.DataFrame,
    *,
    epsilon: float,
    delta_u: float,
) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method", sort=False):
        ordered = group.sort_values(
            ["data_amount", "window_count", "trajectory_fraction"],
            ascending=[True, True, True],
        ).copy()
        conv_candidates = ordered.sort_values(
            ["data_amount", "window_count", "trajectory_fraction"],
            ascending=[False, False, False],
        )
        conv_uncertainty = float(conv_candidates["uncertainty"].iloc[0])
        if not np.isfinite(conv_uncertainty) or conv_uncertainty <= 0:
            ordered["U_conv"] = np.nan
            ordered["R_U"] = np.nan
            ordered["C_U"] = False
        else:
            ordered["U_conv"] = conv_uncertainty
            ordered["R_U"] = ordered["uncertainty"] / conv_uncertainty
            ordered["C_U"] = np.abs(ordered["R_U"] - 1.0) < delta_u
        ordered["C_E"] = ordered["rmse_wham"] < epsilon
        ordered["category_code"] = (
            ordered["C_E"].astype(int) + 2 * ordered["C_U"].astype(int)
        )
        ordered["category"] = ordered["category_code"].map(CATEGORY_LABELS)
        rows.append(ordered)
    return pd.concat(rows, ignore_index=True)


def _first_true(group: pd.DataFrame, column: str) -> pd.Series | None:
    hits = group[group[column]].sort_values(
        ["data_amount", "window_count", "trajectory_fraction"],
        ascending=[True, True, True],
    )
    if hits.empty:
        return None
    return hits.iloc[0]


def _metrics_for_group(group: pd.DataFrame) -> dict[str, float | str]:
    first_u = _first_true(group, "C_U")
    first_e = _first_true(group, "C_E")
    cu_count = int(group["C_U"].sum())
    ce_count = int(group["C_E"].sum())
    both_count = int((group["C_U"] & group["C_E"]).sum())
    false_count = int((group["C_U"] & ~group["C_E"]).sum())
    missed_count = int((~group["C_U"] & group["C_E"]).sum())
    n = int(len(group))

    return {
        "method": str(group["method"].iloc[0]),
        "n_t": n,
        "U_conv": float(group["U_conv"].dropna().iloc[0]) if group["U_conv"].notna().any() else np.nan,
        "first_uncertainty_converged_t": np.nan if first_u is None else float(first_u["data_fraction"]),
        "first_uncertainty_converged_label": "" if first_u is None else str(first_u["cell_label"]),
        "first_accuracy_converged_t": np.nan if first_e is None else float(first_e["data_fraction"]),
        "first_accuracy_converged_label": "" if first_e is None else str(first_e["cell_label"]),
        "stopping_time_error": np.nan if first_u is None else float(first_u["rmse_wham"]),
        "stopping_time_uncertainty": np.nan if first_u is None else float(first_u["uncertainty"]),
        "false_convergence": bool(false_count > 0),
        "false_convergence_rate": false_count / n if n else np.nan,
        "reliability": both_count / cu_count if cu_count else np.nan,
        "missed_convergence_rate": missed_count / n if n else np.nan,
        "n_uncertainty_converged": cu_count,
        "n_accuracy_converged": ce_count,
    }


def _compute_method_metrics(by_t: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _metrics_for_group(group)
        for _, group in by_t.groupby("method", sort=False)
    ]
    return pd.DataFrame(rows)


def _save_diagnostic_plots(
    by_t: pd.DataFrame,
    output_dir: Path,
    *,
    epsilon: float,
) -> None:
    for method, group in by_t.groupby("method", sort=False):
        ordered = group.sort_values("data_fraction")
        tag = str(ordered["method_tag"].iloc[0])
        fig, ax_left = plt.subplots(figsize=(6.4, 4.0))
        ax_right = ax_left.twinx()

        ax_left.plot(
            ordered["data_fraction"],
            ordered["R_U"],
            marker="o",
            lw=1.8,
            color="tab:blue",
            label="Relative uncertainty",
        )
        ax_right.plot(
            ordered["data_fraction"],
            ordered["rmse_wham"] / epsilon,
            marker="s",
            lw=1.5,
            color="tab:red",
            label="RMSE / epsilon",
        )

        ax_left.axhline(1.0, color="tab:blue", lw=1.0, ls=":")
        ax_right.axhline(1.0, color="tab:red", lw=1.0, ls=":")
        ax_left.set_xlabel("Relative data amount")
        ax_left.set_ylabel("U(t) / U_conv", color="tab:blue")
        ax_right.set_ylabel("E(t) / epsilon", color="tab:red")
        ax_left.set_title(f"Convergence certification: {method}")
        ax_left.grid(alpha=0.25)
        ax_left.tick_params(axis="y", labelcolor="tab:blue")
        ax_right.tick_params(axis="y", labelcolor="tab:red")
        fig.tight_layout()
        fig.savefig(output_dir / f"convergence_diagnostic_{tag}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def _save_stopping_time_plot(
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    epsilon: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    values = metrics["stopping_time_error"].to_numpy(dtype=float)
    labels = metrics["method"].tolist()
    colors = ["0.7" if np.isnan(value) else "tab:blue" for value in values]
    ax.bar(range(len(labels)), values, color=colors)
    ax.axhline(epsilon, color="tab:red", lw=1.2, ls="--", label="epsilon")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("E(t_U) RMSE vs reference (kJ/mol)")
    ax.set_title("Stopping-time error from uncertainty-scale convergence")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "stopping_time_error.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _save_false_convergence_matrix(by_t: pd.DataFrame, output_dir: Path) -> None:
    ordered_labels = (
        by_t[["cell_label", "data_fraction", "window_count", "trajectory_fraction"]]
        .drop_duplicates()
        .sort_values(["data_fraction", "window_count", "trajectory_fraction"])
    )
    methods = list(by_t["method"].drop_duplicates())
    matrix = np.full((len(methods), len(ordered_labels)), np.nan)
    for i, method in enumerate(methods):
        group = by_t[by_t["method"] == method].set_index("cell_label")
        for j, label in enumerate(ordered_labels["cell_label"]):
            if label in group.index:
                matrix[i, j] = int(group.loc[label, "category_code"])

    matrix_df = pd.DataFrame(matrix, index=methods, columns=ordered_labels["cell_label"])
    matrix_df.to_csv(output_dir / "false_convergence_matrix.csv")

    cmap = ListedColormap(["#d9d9d9", "#4c78a8", "#e45756", "#54a24b"])
    fig_width = max(8.0, min(24.0, 0.22 * len(ordered_labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 0.75 * len(methods) + 2.0))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=3.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    tick_step = max(1, int(np.ceil(len(ordered_labels) / 20)))
    tick_idx = np.arange(0, len(ordered_labels), tick_step)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(ordered_labels["cell_label"].iloc[tick_idx], rotation=60, ha="right")
    ax.set_xlabel("Data amount t (ablation cell, sorted low to high)")
    ax.set_title("Convergence-certification categories")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.025, pad=0.02)
    cbar.ax.set_yticklabels(
        [
            "neither",
            "accurate only",
            "U scale only",
            "accurate + U scale",
        ]
    )
    fig.tight_layout()
    fig.savefig(output_dir / "false_convergence_matrix.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _compute_sensitivity(
    df: pd.DataFrame,
    *,
    epsilon: float,
    delta_grid: list[float],
) -> pd.DataFrame:
    rows = []
    for delta in delta_grid:
        by_t = _add_convergence_columns(df, epsilon=epsilon, delta_u=delta)
        metrics = _compute_method_metrics(by_t)
        for _, row in metrics.iterrows():
            rows.append(
                {
                    "delta_U": delta,
                    "method": row["method"],
                    "reliability": row["reliability"],
                    "false_convergence_rate": row["false_convergence_rate"],
                    "missed_convergence_rate": row["missed_convergence_rate"],
                    "n_uncertainty_converged": row["n_uncertainty_converged"],
                }
            )
    return pd.DataFrame(rows)


def _save_sensitivity_plot(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True, sharey=True)
    for method, group in sensitivity.groupby("method", sort=False):
        axes[0].plot(group["delta_U"], group["reliability"], marker="o", lw=1.8, label=method)
        axes[1].plot(group["delta_U"], group["false_convergence_rate"], marker="o", lw=1.8, label=method)
    axes[0].set_title("Reliability P(C_E | C_U)")
    axes[1].set_title("False convergence rate")
    for ax in axes:
        ax.set_xlabel("delta_U")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Rate")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle("Convergence-certification sensitivity")
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_sensitivity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "convergence_certification"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon = _parse_epsilon(args.epsilon, temperature=args.temperature)
    delta_grid = _parse_delta_grid(args.delta_u_grid)
    methods = _parse_methods(args.methods)
    raw = _load_method_data(results_dir, methods)
    by_t = _add_convergence_columns(raw, epsilon=epsilon, delta_u=args.delta_u)
    metrics = _compute_method_metrics(by_t)
    sensitivity = _compute_sensitivity(raw, epsilon=epsilon, delta_grid=delta_grid)

    metrics.to_csv(output_dir / "convergence_certification_metrics.csv", index=False)
    by_t.to_csv(output_dir / "convergence_certification_by_t.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity_deltaU.csv", index=False)
    _save_diagnostic_plots(by_t, output_dir, epsilon=epsilon)
    _save_stopping_time_plot(metrics, output_dir, epsilon=epsilon)
    _save_false_convergence_matrix(by_t, output_dir)
    _save_sensitivity_plot(sensitivity, output_dir)

    print(f"epsilon: {epsilon:.6g} kJ/mol")
    print(f"delta_U: {args.delta_u:.6g}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
