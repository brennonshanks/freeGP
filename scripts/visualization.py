#!/usr/bin/env python3
"""Paper-ready visualization utilities for freeGP benchmark results.

The current entry point builds ablation-grid heatmaps from existing
``ablation_metrics.csv`` files. It intentionally reads saved summaries only;
it does not rerun any GP, UI, or HMC calculations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np


DEFAULT_RESULTS_DIR = Path("results/membrane-ablation-10x10-5replicates")
DEFAULT_FIGURES = ["ablation_heatmaps"]
AVAILABLE_FIGURES = [
    "ablation_heatmaps",
    "main_text_ablation_grids",
    "si_lml_loo_ablation",
    "si_map_sampled_difference",
]
DEFAULT_WINDOW_COUNTS = [3, 4, 6, 8, 10, 13, 16, 19, 22, 25]
DEFAULT_TRAJ_FRACTIONS = [1.0, 0.9, 0.8, 0.6, 0.4, 0.25, 0.16, 0.1, 0.063, 0.04]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    relative_csv: Path


METHOD_SPECS = {
    "ui": MethodSpec("ui", "Umbrella integration", Path("ui/ui/ablation_metrics.csv")),
    "fixed": MethodSpec("fixed", "Fixed-hyperparameter GP", Path("fixed/ablation_metrics.csv")),
    "lml_map": MethodSpec("lml_map", "Optimized GP (LML)", Path("lml_map/ablation_metrics.csv")),
    "lml": MethodSpec("lml", "Hierarchical GP (LML)", Path("lml/ablation_metrics.csv")),
    "loo_map": MethodSpec("loo_map", "Optimized GP (LOO)", Path("loo_map/ablation_metrics.csv")),
    "loo": MethodSpec("loo", "Hierarchical GP (LOO)", Path("loo/ablation_metrics.csv")),
}
DEFAULT_METHODS = ["ui", "fixed", "lml_map", "lml", "loo_map", "loo"]
MAIN_TEXT_ABLATION_METHODS = ["ui", "fixed", "loo_map", "loo"]
SI_LML_LOO_METHODS = ["lml", "loo"]
MAP_SAMPLED_PAIRS = [
    ("lml", "lml_map", "LML"),
    ("loo", "loo_map", "LOO"),
]


class TwoSlopeLogNorm(mcolors.Normalize):
    """Log normalization that maps a selected positive value to white/center."""

    def __init__(
        self,
        *,
        vcenter: float,
        vmin: float,
        vmax: float,
        power: float = 1.0,
    ) -> None:
        if not (0.0 < vmin < vcenter < vmax):
            raise ValueError("TwoSlopeLogNorm requires 0 < vmin < vcenter < vmax.")
        self.vcenter = float(vcenter)
        self.power = float(power)
        super().__init__(vmin=float(vmin), vmax=float(vmax))

    def __call__(self, value, clip=None):
        arr = np.ma.asarray(value, dtype=float)
        filled = np.ma.filled(arr, np.nan)
        log_values = np.log(np.where(filled > 0.0, filled, np.nan))
        log_vmin = np.log(self.vmin)
        log_center = np.log(self.vcenter)
        log_vmax = np.log(self.vmax)
        lower = np.clip((log_values - log_vmin) / (log_center - log_vmin), 0.0, 1.0)
        upper = np.clip((log_vmax - log_values) / (log_vmax - log_center), 0.0, 1.0)
        mapped = np.where(
            log_values <= log_center,
            0.5 * lower**self.power,
            1.0 - 0.5 * upper**self.power,
        )
        return np.ma.masked_invalid(mapped)

    def inverse(self, value):
        value = np.asarray(value)
        log_vmin = np.log(self.vmin)
        log_center = np.log(self.vcenter)
        log_vmax = np.log(self.vmax)
        return np.where(
            value <= 0.5,
            np.exp(log_vmin + (log_center - log_vmin) * (2.0 * value) ** (1.0 / self.power)),
            np.exp(log_vmax - (log_vmax - log_center) * (2.0 * (1.0 - value)) ** (1.0 / self.power)),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for paper figures. Each figure module gets its own subfolder.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=AVAILABLE_FIGURES,
        default=DEFAULT_FIGURES,
        help="Figure modules to generate.",
    )
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_SPECS), default=DEFAULT_METHODS)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", nargs="+", default=["svg", "png"], help="Figure formats to write.")
    parser.add_argument("--rmse-center", type=float, default=5.0)
    parser.add_argument("--std-center", type=float, default=5.0)
    parser.add_argument("--rmse-vmax", type=float, default=None)
    parser.add_argument("--std-vmax", type=float, default=None)
    parser.add_argument("--std-power", type=float, default=1.4)
    parser.add_argument("--cmap", default="seismic")
    parser.add_argument("--main-text-vmin", type=float, default=0.25)
    parser.add_argument("--main-text-center", type=float, default=5.0)
    parser.add_argument("--main-text-vmax", type=float, default=175.0)
    parser.add_argument("--main-text-std-vmin", type=float, default=0.3)
    parser.add_argument("--main-text-std-vmax", type=float, default=50.0)
    return parser


def paper_figure_root(args: argparse.Namespace) -> Path:
    results_dir = args.results_dir.resolve()
    return (args.output_dir or results_dir / "paper_figures").resolve()


def figure_output_dir(args: argparse.Namespace, module_name: str) -> Path:
    out = paper_figure_root(args) / module_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_value(row: dict[str, str], metric: str) -> float:
    if metric == "avg_total_std" and not row.get("avg_total_std"):
        variance = float(row["avg_total_variance"])
        return float(np.sqrt(max(variance, 0.0)))
    return float(row[metric])


def load_metric_grid(
    csv_path: Path,
    *,
    metric: str,
    window_counts: list[int],
    trajectory_fractions: list[float],
) -> np.ndarray:
    rows = _read_csv_rows(csv_path)
    grid = np.full((len(window_counts), len(trajectory_fractions)), np.nan, dtype=float)
    for i, window_count in enumerate(window_counts):
        for j, trajectory_fraction in enumerate(trajectory_fractions):
            matches = [
                row
                for row in rows
                if int(float(row["window_count"])) == window_count
                and np.isclose(float(row["trajectory_fraction"]), trajectory_fraction)
            ]
            if matches:
                grid[i, j] = _metric_value(matches[0], metric)
    return grid


def _positive_finite_values(grids: list[np.ndarray]) -> np.ndarray:
    values = np.concatenate([grid.ravel() for grid in grids])
    return values[np.isfinite(values) & (values > 0.0)]


def make_norm(
    grids: list[np.ndarray],
    *,
    center: float,
    vmax: float | None = None,
    power: float = 1.0,
) -> mcolors.Normalize:
    finite = _positive_finite_values(grids)
    if finite.size == 0:
        raise ValueError("No finite positive values available for color normalization.")
    vmin = float(np.nanmin(finite))
    vmax_value = float(np.nanmax(finite) if vmax is None else vmax)
    if vmin < center < vmax_value:
        return TwoSlopeLogNorm(vmin=vmin, vcenter=center, vmax=vmax_value, power=power)
    return mcolors.LogNorm(vmin=vmin, vmax=vmax_value)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def configure_main_text_matplotlib() -> None:
    rc("font", **{"family": "sans-serif", "sans-serif": ["DejaVu Sans"], "size": 10})
    rc("mathtext", **{"default": "regular"})
    plt.rcParams.update(
        {
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_heatmap(
    ax,
    grid: np.ndarray,
    *,
    norm: mcolors.Normalize,
    cmap: str,
    window_counts: list[int],
    trajectory_fractions: list[float],
    show_xlabels: bool,
    show_ylabels: bool,
):
    im = ax.imshow(
        grid.T,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xticks(np.arange(len(window_counts)))
    ax.set_xticklabels([str(value) for value in window_counts] if show_xlabels else [])
    ax.set_yticks(np.arange(len(trajectory_fractions)))
    ax.set_yticklabels(
        [f"{100.0 * value:.3g}" for value in trajectory_fractions] if show_ylabels else []
    )
    return im


def draw_main_text_heatmap(
    ax,
    grid: np.ndarray,
    *,
    norm: mcolors.Normalize,
    cmap: str,
):
    im = ax.imshow(
        grid.T,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xticks(np.arange(len(DEFAULT_WINDOW_COUNTS)))
    ax.set_xticklabels(
        [str(value) for value in DEFAULT_WINDOW_COUNTS],
        rotation=0,
        ha="center",
    )
    ax.set_yticks(np.arange(len(DEFAULT_TRAJ_FRACTIONS)))
    ax.set_yticklabels([f"{100.0 * value:.3g}" for value in DEFAULT_TRAJ_FRACTIONS])
    ax.set_xlabel("Umbrella windows", labelpad=5)
    ax.set_ylabel("Trajectory length (%)", labelpad=6)
    return im


def _set_colorbar_ticks(cbar, ticks: list[float]) -> None:
    ticks = [tick for tick in ticks if cbar.norm.vmin <= tick <= cbar.norm.vmax]
    if not ticks:
        return
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{tick:g}" for tick in ticks])


def plot_ablation_heatmaps(args: argparse.Namespace) -> None:
    results_dir = args.results_dir.resolve()
    output_dir = figure_output_dir(args, "ablation_heatmaps")

    methods = []
    for key in args.methods:
        spec = METHOD_SPECS[key]
        csv_path = results_dir / spec.relative_csv
        if csv_path.exists():
            methods.append((spec, csv_path))
        else:
            print(f"Skipping {key}: missing {csv_path}")
    if not methods:
        raise FileNotFoundError(f"No requested ablation metric files found in {results_dir}")

    rmse_grids = [
        load_metric_grid(
            csv_path,
            metric="rmse_wham",
            window_counts=DEFAULT_WINDOW_COUNTS,
            trajectory_fractions=DEFAULT_TRAJ_FRACTIONS,
        )
        for _, csv_path in methods
    ]
    std_grids = [
        load_metric_grid(
            csv_path,
            metric="avg_total_std",
            window_counts=DEFAULT_WINDOW_COUNTS,
            trajectory_fractions=DEFAULT_TRAJ_FRACTIONS,
        )
        for _, csv_path in methods
    ]

    rmse_norm = make_norm(rmse_grids, center=args.rmse_center, vmax=args.rmse_vmax)
    std_norm = make_norm(std_grids, center=args.std_center, vmax=args.std_vmax, power=args.std_power)

    configure_matplotlib()
    n_cols = len(methods)
    fig_width = max(7.0, 1.55 * n_cols + 1.0)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(fig_width, 4.7),
        squeeze=False,
        constrained_layout=True,
    )

    row_specs = [
        (rmse_grids, rmse_norm, r"RMSE vs WHAM (kJ mol$^{-1}$)", [1.0, 2.0, 5.0, 10.0, 100.0]),
        (std_grids, std_norm, r"Average predictive SD (kJ mol$^{-1}$)", [1.0, 2.0, 5.0, 10.0, 100.0]),
    ]
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    for row, (grids, norm, cbar_label, cbar_ticks) in enumerate(row_specs):
        last_im = None
        for col, ((spec, _csv_path), grid) in enumerate(zip(methods, grids)):
            ax = axes[row, col]
            last_im = draw_heatmap(
                ax,
                grid,
                norm=norm,
                cmap=args.cmap,
                window_counts=DEFAULT_WINDOW_COUNTS,
                trajectory_fractions=DEFAULT_TRAJ_FRACTIONS,
                show_xlabels=row == 1,
                show_ylabels=col == 0,
            )
            if row == 0:
                ax.set_title(f"({panel_letters[col]}) {spec.label}", pad=4)
            if row == 1:
                ax.set_xlabel("Umbrella windows")
            if col == 0:
                ax.set_ylabel("Trajectory length (%)")
        assert last_im is not None
        cbar = fig.colorbar(last_im, ax=axes[row, :], fraction=0.035, pad=0.015)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(length=2.5, width=0.6)
        cbar.outline.set_linewidth(0.6)
        _set_colorbar_ticks(cbar, cbar_ticks)

    for fmt in args.formats:
        out = output_dir / f"ablation_heatmaps_paper.{fmt}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def _save_figure(fig, output_dir: Path, stem: str, formats: list[str], dpi: int) -> None:
    for fmt in formats:
        out = output_dir / f"{stem}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out}")


def _main_text_color_limits(
    grids: dict[str, np.ndarray],
    *,
    center: float,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    values = _positive_finite_values(list(grids.values()))
    if values.size == 0:
        raise ValueError("No finite positive RMSE values available for main-text color scaling.")
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    lower = float(np.floor(data_min)) if vmin is None else float(vmin)
    upper = float(np.ceil(data_max)) if vmax is None else float(vmax)
    if lower >= center:
        lower = max(center * 0.5, center - 1.0)
    if upper <= center:
        upper = center + 1.0
    return lower, upper


def _nice_tick_label(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _even_visual_ticks(norm: mcolors.Normalize, *, n_ticks: int = 7) -> tuple[np.ndarray, list[str]]:
    """Return data ticks that are evenly spaced in colorbar coordinates."""
    positions = np.linspace(0.0, 1.0, n_ticks)
    values = np.asarray(norm.inverse(positions), dtype=float)
    labels = [_nice_tick_label(value) for value in values]
    return values, labels


def _main_text_methods(results_dir: Path) -> list[tuple[MethodSpec, Path]]:
    methods: list[tuple[MethodSpec, Path]] = []
    for key in MAIN_TEXT_ABLATION_METHODS:
        spec = METHOD_SPECS[key]
        csv_path = results_dir / spec.relative_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required main-text metric file: {csv_path}")
        methods.append((spec, csv_path))
    return methods


def _required_methods(results_dir: Path, keys: list[str]) -> list[tuple[MethodSpec, Path]]:
    methods: list[tuple[MethodSpec, Path]] = []
    for key in keys:
        spec = METHOD_SPECS[key]
        csv_path = results_dir / spec.relative_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required metric file: {csv_path}")
        methods.append((spec, csv_path))
    return methods


def _load_main_text_metric_grids(
    methods: list[tuple[MethodSpec, Path]],
    *,
    metric: str,
) -> dict[str, np.ndarray]:
    return {
        spec.key: load_metric_grid(
            csv_path,
            metric=metric,
            window_counts=DEFAULT_WINDOW_COUNTS,
            trajectory_fractions=DEFAULT_TRAJ_FRACTIONS,
        )
        for spec, csv_path in methods
    }


def _main_text_norm(
    grids: dict[str, np.ndarray],
    *,
    center: float,
    vmin: float | None,
    vmax: float | None,
) -> TwoSlopeLogNorm:
    lower, upper = _main_text_color_limits(grids, center=center, vmin=vmin, vmax=vmax)
    if lower <= 0.0:
        lower = max(1e-3, 0.5 * float(np.nanmin(_positive_finite_values(list(grids.values())))))
    return TwoSlopeLogNorm(vmin=lower, vcenter=center, vmax=upper, power=1.0)


def _plot_main_text_metric_panels(
    *,
    methods: list[tuple[MethodSpec, Path]],
    grids: dict[str, np.ndarray],
    norm: mcolors.Normalize,
    output_dir: Path,
    file_prefix: str,
    cbar_label: str,
    cbar_ticks: list[float],
    args: argparse.Namespace,
) -> None:
    for spec, _csv_path in methods:
        fig, ax = plt.subplots(figsize=(3.45, 2.25))
        draw_main_text_heatmap(ax, grids[spec.key], norm=norm, cmap=args.cmap)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.20, top=0.98)
        _save_figure(fig, output_dir, f"{file_prefix}_{spec.key}", args.formats, args.dpi)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(0.42, 2.65))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=args.cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        cax=ax,
        orientation="vertical",
        extend="both",
        ticks=cbar_ticks,
    )
    cbar.set_ticklabels([_nice_tick_label(value) for value in cbar_ticks])
    cbar.set_label(cbar_label, labelpad=4)
    cbar.ax.tick_params(length=2.8, width=0.7, pad=1)
    cbar.outline.set_linewidth(0.7)
    _save_figure(fig, output_dir, f"shared_{file_prefix}_colorbar", args.formats, args.dpi)
    plt.close(fig)


def _draw_preview_heatmap(
    ax,
    grid: np.ndarray,
    *,
    norm: mcolors.Normalize,
    cmap: str,
    show_xlabels: bool,
    show_ylabels: bool,
):
    im = ax.imshow(
        grid.T,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    ax.set_xticks(np.arange(len(DEFAULT_WINDOW_COUNTS)))
    ax.set_xticklabels([str(value) for value in DEFAULT_WINDOW_COUNTS] if show_xlabels else [])
    ax.set_yticks(np.arange(len(DEFAULT_TRAJ_FRACTIONS)))
    ax.set_yticklabels(
        [f"{100.0 * value:.3g}" for value in DEFAULT_TRAJ_FRACTIONS] if show_ylabels else []
    )
    return im


def _add_vertical_colorbar(
    fig,
    cax,
    *,
    norm: mcolors.Normalize,
    cmap: str,
    ticks: list[float],
    label: str,
) -> None:
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", extend="both", ticks=ticks)
    cbar.set_ticklabels([_nice_tick_label(value) for value in ticks])
    cbar.set_label(label, labelpad=4)
    cbar.ax.tick_params(length=2.8, width=0.7, pad=1)
    cbar.outline.set_linewidth(0.7)


def _plot_main_text_combined_preview(
    *,
    methods: list[tuple[MethodSpec, Path]],
    rmse_grids: dict[str, np.ndarray],
    std_grids: dict[str, np.ndarray],
    rmse_norm: mcolors.Normalize,
    std_norm: mcolors.Normalize,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    fig = plt.figure(figsize=(12.8, 5.25), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        len(methods) + 1,
        width_ratios=[1.0] * len(methods) + [0.055],
        wspace=0.08,
        hspace=0.10,
    )
    axes = np.empty((2, len(methods)), dtype=object)
    for row in range(2):
        for col, (spec, _csv_path) in enumerate(methods):
            ax = fig.add_subplot(gs[row, col])
            axes[row, col] = ax
            grids = rmse_grids if row == 0 else std_grids
            norm = rmse_norm if row == 0 else std_norm
            _draw_preview_heatmap(
                ax,
                grids[spec.key],
                norm=norm,
                cmap=args.cmap,
                show_xlabels=row == 1,
                show_ylabels=col == 0,
            )
            if row == 0:
                ax.set_title(spec.label, pad=4)

    _add_vertical_colorbar(
        fig,
        fig.add_subplot(gs[0, -1]),
        norm=rmse_norm,
        cmap=args.cmap,
        ticks=[0.25, 0.75, 2.0, 5.0, 15.0, 50.0, 175.0],
        label="RMSE vs WHAM (kJ/mol)",
    )
    _add_vertical_colorbar(
        fig,
        fig.add_subplot(gs[1, -1]),
        norm=std_norm,
        cmap=args.cmap,
        ticks=[0.3, 0.75, 2.0, 5.0, 10.0, 25.0, 50.0],
        label="Average predictive SD (kJ/mol)",
    )
    fig.supxlabel("Umbrella windows", y=0.025)
    fig.supylabel("Trajectory length (%)", x=0.025)
    fig.subplots_adjust(left=0.070, right=0.94, bottom=0.11, top=0.92)
    _save_figure(fig, output_dir, "main_text_ablation_grids_preview", args.formats, args.dpi)
    plt.close(fig)


def _plot_two_metric_grid_preview(
    *,
    methods: list[tuple[MethodSpec, Path]],
    rmse_grids: dict[str, np.ndarray],
    std_grids: dict[str, np.ndarray],
    rmse_norm: mcolors.Normalize,
    std_norm: mcolors.Normalize,
    output_dir: Path,
    stem: str,
    args: argparse.Namespace,
) -> None:
    fig = plt.figure(figsize=(6.8, 5.25), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        len(methods) + 1,
        width_ratios=[1.0] * len(methods) + [0.07],
        wspace=0.10,
        hspace=0.10,
    )
    for row in range(2):
        for col, (spec, _csv_path) in enumerate(methods):
            ax = fig.add_subplot(gs[row, col])
            grids = rmse_grids if row == 0 else std_grids
            norm = rmse_norm if row == 0 else std_norm
            _draw_preview_heatmap(
                ax,
                grids[spec.key],
                norm=norm,
                cmap=args.cmap,
                show_xlabels=row == 1,
                show_ylabels=col == 0,
            )
            if row == 0:
                ax.set_title(spec.label, pad=4)

    _add_vertical_colorbar(
        fig,
        fig.add_subplot(gs[0, -1]),
        norm=rmse_norm,
        cmap=args.cmap,
        ticks=[0.25, 0.75, 2.0, 5.0, 15.0, 50.0, 175.0],
        label="RMSE vs WHAM (kJ/mol)",
    )
    _add_vertical_colorbar(
        fig,
        fig.add_subplot(gs[1, -1]),
        norm=std_norm,
        cmap=args.cmap,
        ticks=[0.3, 0.75, 2.0, 5.0, 10.0, 25.0, 50.0],
        label="Average predictive SD (kJ/mol)",
    )
    fig.supxlabel("Umbrella windows", y=0.025)
    fig.supylabel("Trajectory length (%)", x=0.025)
    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.11, top=0.92)
    _save_figure(fig, output_dir, stem, args.formats, args.dpi)
    plt.close(fig)


def _symmetric_percent_norm(grids: list[np.ndarray]) -> mcolors.TwoSlopeNorm:
    return mcolors.TwoSlopeNorm(vmin=-50.0, vcenter=0.0, vmax=50.0)


def _difference_ticks(norm: mcolors.Normalize) -> list[float]:
    return [-50.0, -25.0, 0.0, 25.0, 50.0]


def _plot_map_sampled_difference_preview(
    *,
    rmse_percent_diffs: dict[str, np.ndarray],
    std_percent_diffs: dict[str, np.ndarray],
    rmse_norm: mcolors.Normalize,
    std_norm: mcolors.Normalize,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    fig = plt.figure(figsize=(7.0, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        len(MAP_SAMPLED_PAIRS) + 1,
        width_ratios=[1.0] * len(MAP_SAMPLED_PAIRS) + [0.07],
        wspace=0.10,
        hspace=0.18,
    )
    row_specs = [
        (rmse_percent_diffs, rmse_norm, r"% change, HMC-NUTS vs MAP"),
        (std_percent_diffs, std_norm, r"% change, HMC-NUTS vs MAP"),
    ]
    for row, (diffs, norm, _label) in enumerate(row_specs):
        for col, (_sampled_key, _map_key, title) in enumerate(MAP_SAMPLED_PAIRS):
            ax = fig.add_subplot(gs[row, col])
            key = title.lower()
            _draw_preview_heatmap(
                ax,
                diffs[key],
                norm=norm,
                cmap="coolwarm",
                show_xlabels=row == 1,
                show_ylabels=col == 0,
            )
            avg_diff = float(np.nanmean(diffs[key]))
            ax.set_title(rf"$\bar{{\Delta}}$ = {_nice_tick_label(avg_diff)}%", pad=4)

    for row, (_diffs, norm, label) in enumerate(row_specs):
        ticks = _difference_ticks(norm)
        _add_vertical_colorbar(
            fig,
            fig.add_subplot(gs[row, -1]),
            norm=norm,
        cmap="coolwarm",
            ticks=ticks,
            label=label,
        )
    fig.supxlabel("Umbrella windows", y=0.025)
    fig.supylabel("Trajectory length (%)", x=0.025)
    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.11, top=0.90)
    _save_figure(fig, output_dir, "si_map_sampled_difference_preview", args.formats, args.dpi)
    plt.close(fig)


def plot_main_text_ablation_grids(args: argparse.Namespace) -> None:
    """Export individual main-text heatmap panels and shared colorbars."""
    results_dir = args.results_dir.resolve()
    output_dir = figure_output_dir(args, "main_text_ablation_grids")
    configure_main_text_matplotlib()

    methods = _main_text_methods(results_dir)
    rmse_grids = _load_main_text_metric_grids(methods, metric="rmse_wham")
    std_grids = _load_main_text_metric_grids(methods, metric="avg_total_std")

    rmse_norm = _main_text_norm(
        rmse_grids,
        center=args.main_text_center,
        vmin=args.main_text_vmin,
        vmax=args.main_text_vmax,
    )
    std_norm = _main_text_norm(
        std_grids,
        center=args.std_center,
        vmin=args.main_text_std_vmin,
        vmax=args.main_text_std_vmax,
    )

    _plot_main_text_metric_panels(
        methods=methods,
        grids=rmse_grids,
        norm=rmse_norm,
        output_dir=output_dir,
        file_prefix="rmse",
        cbar_label="RMSE vs WHAM (kJ/mol)",
        cbar_ticks=[0.25, 0.75, 2.0, 5.0, 15.0, 50.0, 175.0],
        args=args,
    )
    _plot_main_text_metric_panels(
        methods=methods,
        grids=std_grids,
        norm=std_norm,
        output_dir=output_dir,
        file_prefix="avg_std",
        cbar_label="Average predictive SD (kJ/mol)",
        cbar_ticks=[0.3, 0.75, 2.0, 5.0, 10.0, 25.0, 50.0],
        args=args,
    )
    _plot_main_text_combined_preview(
        methods=methods,
        rmse_grids=rmse_grids,
        std_grids=std_grids,
        rmse_norm=rmse_norm,
        std_norm=std_norm,
        output_dir=output_dir,
        args=args,
    )


def plot_si_lml_loo_ablation(args: argparse.Namespace) -> None:
    """Export SI comparison of sampled LML and LOO hierarchical GP grids."""
    results_dir = args.results_dir.resolve()
    output_dir = figure_output_dir(args, "si_lml_loo_ablation")
    configure_main_text_matplotlib()

    methods = _required_methods(results_dir, SI_LML_LOO_METHODS)
    rmse_grids = _load_main_text_metric_grids(methods, metric="rmse_wham")
    std_grids = _load_main_text_metric_grids(methods, metric="avg_total_std")
    rmse_norm = _main_text_norm(
        rmse_grids,
        center=args.main_text_center,
        vmin=args.main_text_vmin,
        vmax=args.main_text_vmax,
    )
    std_norm = _main_text_norm(
        std_grids,
        center=args.std_center,
        vmin=args.main_text_std_vmin,
        vmax=args.main_text_std_vmax,
    )

    _plot_main_text_metric_panels(
        methods=methods,
        grids=rmse_grids,
        norm=rmse_norm,
        output_dir=output_dir,
        file_prefix="rmse",
        cbar_label="RMSE vs WHAM (kJ/mol)",
        cbar_ticks=[0.25, 0.75, 2.0, 5.0, 15.0, 50.0, 175.0],
        args=args,
    )
    _plot_main_text_metric_panels(
        methods=methods,
        grids=std_grids,
        norm=std_norm,
        output_dir=output_dir,
        file_prefix="avg_std",
        cbar_label="Average predictive SD (kJ/mol)",
        cbar_ticks=[0.3, 0.75, 2.0, 5.0, 10.0, 25.0, 50.0],
        args=args,
    )
    _plot_two_metric_grid_preview(
        methods=methods,
        rmse_grids=rmse_grids,
        std_grids=std_grids,
        rmse_norm=rmse_norm,
        std_norm=std_norm,
        output_dir=output_dir,
        stem="si_lml_loo_ablation_preview",
        args=args,
    )


def plot_si_map_sampled_difference(args: argparse.Namespace) -> None:
    """Export percent change from MAP to HMC-NUTS hyperposterior heatmaps."""
    results_dir = args.results_dir.resolve()
    output_dir = figure_output_dir(args, "si_map_sampled_difference")
    configure_main_text_matplotlib()

    required_keys = sorted({key for pair in MAP_SAMPLED_PAIRS for key in pair[:2]})
    methods = _required_methods(results_dir, required_keys)
    rmse_grids = _load_main_text_metric_grids(methods, metric="rmse_wham")
    std_grids = _load_main_text_metric_grids(methods, metric="avg_total_std")

    rmse_diffs: dict[str, np.ndarray] = {}
    std_diffs: dict[str, np.ndarray] = {}
    rmse_percent_diffs: dict[str, np.ndarray] = {}
    std_percent_diffs: dict[str, np.ndarray] = {}
    summary_rows = [
        [
            "objective",
            "metric",
            "mean_map_minus_sampled",
            "mean_percent_map_minus_sampled_vs_sampled",
            "mean_percent_sampled_minus_map_vs_map",
        ]
    ]
    for sampled_key, map_key, label in MAP_SAMPLED_PAIRS:
        diff_key = label.lower()
        rmse_diffs[diff_key] = rmse_grids[map_key] - rmse_grids[sampled_key]
        std_diffs[diff_key] = std_grids[map_key] - std_grids[sampled_key]
        rmse_percent = 100.0 * rmse_diffs[diff_key] / rmse_grids[sampled_key]
        std_percent = 100.0 * std_diffs[diff_key] / std_grids[sampled_key]
        rmse_percent_diffs[diff_key] = 100.0 * (rmse_grids[sampled_key] - rmse_grids[map_key]) / rmse_grids[map_key]
        std_percent_diffs[diff_key] = 100.0 * (std_grids[sampled_key] - std_grids[map_key]) / std_grids[map_key]
        summary_rows.append([
            label,
            "rmse_wham",
            f"{float(np.nanmean(rmse_diffs[diff_key])):.8g}",
            f"{float(np.nanmean(rmse_percent)):.8g}",
            f"{float(np.nanmean(rmse_percent_diffs[diff_key])):.8g}",
        ])
        summary_rows.append([
            label,
            "avg_total_std",
            f"{float(np.nanmean(std_diffs[diff_key])):.8g}",
            f"{float(np.nanmean(std_percent)):.8g}",
            f"{float(np.nanmean(std_percent_diffs[diff_key])):.8g}",
        ])

    rmse_norm = _symmetric_percent_norm(list(rmse_percent_diffs.values()))
    std_norm = _symmetric_percent_norm(list(std_percent_diffs.values()))
    _plot_map_sampled_difference_preview(
        rmse_percent_diffs=rmse_percent_diffs,
        std_percent_diffs=std_percent_diffs,
        rmse_norm=rmse_norm,
        std_norm=std_norm,
        output_dir=output_dir,
        args=args,
    )

    summary_path = output_dir / "map_sampled_difference_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(summary_rows)
    print(f"Saved {summary_path}")


FIGURE_MODULES = {
    "ablation_heatmaps": plot_ablation_heatmaps,
    "main_text_ablation_grids": plot_main_text_ablation_grids,
    "si_lml_loo_ablation": plot_si_lml_loo_ablation,
    "si_map_sampled_difference": plot_si_map_sampled_difference,
}


def main() -> None:
    args = build_parser().parse_args()
    for figure_name in args.figures:
        FIGURE_MODULES[figure_name](args)


if __name__ == "__main__":
    main()
