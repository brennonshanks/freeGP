#!/usr/bin/env python3
"""Paper-ready visualization utilities for freeGP benchmark results.

The current entry point builds ablation-grid heatmaps from existing
``ablation_metrics.csv`` files. It intentionally reads saved summaries only;
it does not rerun any GP, UI, or HMC calculations.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator, FormatStrFormatter, NullFormatter
from matplotlib import rc
import numpy as np


DEFAULT_RESULTS_DIR = Path("results/membrane-ablation-10x10-5replicates")
DEFAULT_FIGURES = ["ablation_heatmaps"]
AVAILABLE_FIGURES = [
    "ablation_heatmaps",
    "main_text_ablation_grids",
    "ablation_parity",
    "metadynamics_convergence",
    "si_lml_loo_ablation",
    "si_map_sampled_difference",
    "lengthscale_prior_sensitivity",
    "hmc_calibration_traces",
    "noise_comparison",
]
DEFAULT_WINDOW_COUNTS = [3, 4, 6, 8, 10, 13, 16, 19, 22, 25]
DEFAULT_TRAJ_FRACTIONS = [1.0, 0.9, 0.8, 0.6, 0.4, 0.25, 0.16, 0.1, 0.063, 0.04]

# JCTC/ACS-style production sizes. The main text ablation grid is intended as
# a full-width figure that can be included in Overleaf without rescaling.
JCTC_DOUBLE_COLUMN_WIDTH_IN = 7.0
JCTC_SINGLE_COLUMN_WIDTH_IN = 3.33
MAIN_ABLATION_FIGSIZE = (JCTC_DOUBLE_COLUMN_WIDTH_IN, 3.35)
SI_TWO_METHOD_FIGSIZE = (JCTC_SINGLE_COLUMN_WIDTH_IN, 3.25)
MAIN_ABLATION_PANEL_FIGSIZE = (1.55, 1.45)
MAIN_ABLATION_COLORBAR_FIGSIZE = (0.28, 1.75)
PAPER_FONT_SIZE = 7.0
PAPER_LABEL_SIZE = 7.5
PAPER_TITLE_SIZE = 7.5
PAPER_TICK_SIZE = 5.4
PAPER_LINE_WIDTH = 0.5
PANEL_LETTERS = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    relative_csv: Path


METHOD_SPECS = {
    "ui": MethodSpec("ui", "Umbrella integration", Path("ui/ui/ablation_metrics.csv")),
    "fixed": MethodSpec("fixed", "Fixed GP", Path("fixed/ablation_metrics.csv")),
    "lml_map": MethodSpec("lml_map", "Optimized GP (LML)", Path("lml_map/ablation_metrics.csv")),
    "lml": MethodSpec("lml", "Hierarchical GP (LML)", Path("lml/ablation_metrics.csv")),
    "loo_map": MethodSpec("loo_map", "Optimized GP (LOO)", Path("loo_map/ablation_metrics.csv")),
    "loo": MethodSpec("loo", "Hierarchical GP (LOO)", Path("loo/ablation_metrics.csv")),
}
DEFAULT_METHODS = ["ui", "fixed", "lml_map", "lml", "loo_map", "loo"]
MAIN_TEXT_ABLATION_METHODS = ["ui", "fixed", "loo_map", "loo"]
PARITY_METHODS = ["ui", "fixed", "loo_map", "loo"]
SI_LML_LOO_METHODS = ["lml", "loo"]
MAP_SAMPLED_PAIRS = [
    ("lml", "lml_map", "LML"),
    ("loo", "loo_map", "LOO"),
]
LENGTHSCALE_PRIOR_COLORS = {
    "flat": "#999999",
    "current": "#0072B2",
    "ell_0p5_narrow": "#E69F00",
    "ell_0p5_very_narrow": "#009E73",
}
LENGTHSCALE_PRIOR_LINESTYLES = {
    "flat": (0, (1.0, 1.4)),
    "current": "-",
    "ell_0p5_narrow": "-.",
    "ell_0p5_very_narrow": "--",
}
OKABE_ITO_CHAIN_COLORS = ["#999999", "#0072B2", "#E69F00", "#009E73"]
PARITY_METHOD_COLORS = {
    "ui": "#999999",
    "fixed": "#0072B2",
    "loo_map": "#E69F00",
    "loo": "#009E73",
}
NOISE_MODEL_COLORS = {
    "fixed": "#0072B2",
    "inferred": "#E69F00",
}
NOISE_PRIOR_COLOR = "#7a7a7a"
NOISE_PRIOR_LOG_PARAMS = {
    "ell": (math.log(4.0), 1.0),
    "w": (1.0, 0.5),
    "sigma_f": (0.5, 2.0),
    "sigma_d": (0.5, 2.0),
}


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
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf"], help="Figure formats to write.")
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
    parser.add_argument(
        "--main-text-color-power",
        type=float,
        default=1.5,
        help="Power for main ablation centered-log color scale. Values < 1 reduce the white region around the center.",
    )
    parser.add_argument(
        "--hmc-trace-run",
        default="hard_w1000_s1000",
        help="Calibration sweep run folder used for HMC trace plots.",
    )
    parser.add_argument(
        "--hmc-trace-cell",
        default="w07_f0p25.pt",
        help="Cell artifact filename inside artifacts/cells for HMC trace plots.",
    )
    parser.add_argument(
        "--noise-reference-wham-path",
        type=Path,
        default=Path("reference_data/wham.dat"),
        help="WHAM reference used for the noise-comparison posterior predictive panel.",
    )
    parser.add_argument(
        "--metadynamics-results-dir",
        type=Path,
        default=Path("results/metadynamics"),
        help="Directory containing metadynamics convergence_data.csv.",
    )
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
    rc("font", **{"family": "sans-serif", "sans-serif": ["DejaVu Sans"], "size": PAPER_FONT_SIZE})
    rc("mathtext", **{"default": "regular"})
    plt.rcParams.update(
        {
            "axes.titlesize": PAPER_TITLE_SIZE,
            "axes.labelsize": PAPER_LABEL_SIZE,
            "xtick.labelsize": PAPER_TICK_SIZE,
            "ytick.labelsize": PAPER_TICK_SIZE,
            "axes.linewidth": PAPER_LINE_WIDTH,
            "xtick.major.width": PAPER_LINE_WIDTH,
            "ytick.major.width": PAPER_LINE_WIDTH,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
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
    ax.set_xlabel("Number of umbrella windows", labelpad=5)
    ax.set_ylabel("Trajectory length [%]", labelpad=6)
    ax.tick_params(axis="both", direction="in")
    ax.set_box_aspect(1)
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
        (rmse_grids, rmse_norm, "RMSE [kJ/mol]", [1.0, 2.0, 5.0, 10.0, 100.0]),
        (std_grids, std_norm, "Standard Deviation [kJ/mol]", [1.0, 2.0, 5.0, 10.0, 100.0]),
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
                ax.set_ylabel("Trajectory length [%]")
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


def _save_figure(
    fig,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
    *,
    tight: bool = True,
) -> None:
    for fmt in formats:
        out = output_dir / f"{stem}.{fmt}"
        if tight:
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(out, dpi=dpi)
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
    power: float = 1.0,
) -> TwoSlopeLogNorm:
    lower, upper = _main_text_color_limits(grids, center=center, vmin=vmin, vmax=vmax)
    if lower <= 0.0:
        lower = max(1e-3, 0.5 * float(np.nanmin(_positive_finite_values(list(grids.values())))))
    return TwoSlopeLogNorm(vmin=lower, vcenter=center, vmax=upper, power=power)


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
        fig, ax = plt.subplots(figsize=MAIN_ABLATION_PANEL_FIGSIZE)
        draw_main_text_heatmap(ax, grids[spec.key], norm=norm, cmap=args.cmap)
        fig.subplots_adjust(left=0.22, right=0.98, bottom=0.24, top=0.98)
        _save_figure(fig, output_dir, f"{file_prefix}_{spec.key}", args.formats, args.dpi)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=MAIN_ABLATION_COLORBAR_FIGSIZE)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=args.cmap)
    sm.set_array([])
    visual_ticks, visual_tick_labels = _even_visual_ticks(norm, n_ticks=7)
    cbar = fig.colorbar(
        sm,
        cax=ax,
        orientation="vertical",
        extend="neither",
        ticks=visual_ticks,
    )
    cbar.set_ticklabels(visual_tick_labels)
    cbar.set_label(cbar_label, labelpad=3)
    cbar.ax.tick_params(length=2.4, width=PAPER_LINE_WIDTH, pad=1)
    cbar.outline.set_linewidth(PAPER_LINE_WIDTH)
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
    ax.tick_params(axis="both", direction="in")
    ax.set_box_aspect(1)
    return im


def _add_vertical_colorbar(
    fig,
    cax,
    *,
    norm: mcolors.Normalize,
    cmap: str,
    ticks: list[float],
    label: str,
    label_x: float = 5.1,
) -> None:
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    visual_ticks, visual_tick_labels = _even_visual_ticks(norm, n_ticks=len(ticks))
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", extend="neither", ticks=visual_ticks)
    cbar.set_ticklabels(visual_tick_labels)
    cbar.set_label(label, labelpad=14)
    cbar.ax.yaxis.set_label_coords(label_x, 0.5)
    cbar.ax.tick_params(length=2.4, width=PAPER_LINE_WIDTH, pad=1)
    cbar.outline.set_linewidth(PAPER_LINE_WIDTH)


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
    fig = plt.figure(figsize=MAIN_ABLATION_FIGSIZE, constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        len(methods) + 1,
        width_ratios=[1.0] * len(methods) + [0.055],
        wspace=0.08,
        hspace=0.08,
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
                ax.set_title(f"({PANEL_LETTERS[col]}) {spec.label}", pad=6)

    _add_vertical_colorbar(
        fig,
        fig.add_subplot(gs[0, -1]),
        norm=rmse_norm,
        cmap=args.cmap,
        ticks=[0.25, 0.75, 2.0, 5.0, 15.0, 50.0, 175.0],
        label="RMSE [kJ/mol]",
        label_x=3.6,
    )
    _add_vertical_colorbar(
        fig,
        fig.add_subplot(gs[1, -1]),
        norm=std_norm,
        cmap=args.cmap,
        ticks=[0.3, 0.75, 2.0, 5.0, 10.0, 25.0, 50.0],
        label="Standard Deviation [kJ/mol]",
        label_x=3.6,
    )
    fig.supxlabel("Number of umbrella windows", y=0.022, fontsize=PAPER_LABEL_SIZE)
    fig.supylabel("Trajectory length [%]", x=0.016, fontsize=PAPER_LABEL_SIZE)
    fig.subplots_adjust(left=0.055, right=0.935, bottom=0.115, top=0.925)
    _save_figure(
        fig,
        output_dir,
        "main_text_ablation_grids_preview",
        args.formats,
        args.dpi,
        tight=False,
    )
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
    fig = plt.figure(figsize=SI_TWO_METHOD_FIGSIZE, constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        len(methods) + 1,
        width_ratios=[1.0] * len(methods) + [0.055],
        wspace=0.08,
        hspace=0.08,
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
                ax.set_title(f"({PANEL_LETTERS[col]}) {spec.label}", pad=6)

    _add_vertical_colorbar(
        fig,
        fig.add_subplot(gs[0, -1]),
        norm=rmse_norm,
        cmap=args.cmap,
        ticks=[0.25, 0.75, 2.0, 5.0, 15.0, 50.0, 175.0],
        label="RMSE [kJ/mol]",
        label_x=4.1,
    )
    _add_vertical_colorbar(
        fig,
        fig.add_subplot(gs[1, -1]),
        norm=std_norm,
        cmap=args.cmap,
        ticks=[0.3, 0.75, 2.0, 5.0, 10.0, 25.0, 50.0],
        label="Standard Deviation [kJ/mol]",
        label_x=4.1,
    )
    fig.supxlabel("Number of umbrella windows", y=0.022, fontsize=PAPER_LABEL_SIZE)
    fig.supylabel("Trajectory length [%]", x=0.000, fontsize=PAPER_LABEL_SIZE)
    fig.subplots_adjust(left=0.090, right=0.900, bottom=0.115, top=0.925)
    _save_figure(fig, output_dir, stem, args.formats, args.dpi, tight=False)
    plt.close(fig)


def _symmetric_percent_norm(grids: list[np.ndarray]) -> mcolors.TwoSlopeNorm:
    return mcolors.TwoSlopeNorm(vmin=-50.0, vcenter=0.0, vmax=50.0)


def _difference_ticks(norm: mcolors.Normalize) -> list[float]:
    return [-50.0, -25.0, 0.0, 25.0, 50.0]


def _positive_log_kde(samples: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 2:
        return np.zeros_like(x_grid)
    log_values = np.log(values)
    sample_std = float(np.std(log_values, ddof=1))
    bandwidth = max(sample_std * values.size ** (-1.0 / 5.0), 1e-3)
    z = (np.log(x_grid)[:, None] - log_values[None, :]) / bandwidth
    log_density = np.exp(-0.5 * z**2).mean(axis=1) / (
        bandwidth * math.sqrt(2.0 * math.pi)
    )
    return log_density / x_grid


def _ell_prior_density(
    ell_grid: np.ndarray,
    *,
    m_ell: float,
    s_ell: float | None,
    log_bounds: tuple[float, float],
) -> np.ndarray:
    if s_ell is None:
        lower, upper = log_bounds
        density = 1.0 / (ell_grid * (upper - lower))
        log_ell = np.log(ell_grid)
        return np.where((log_ell >= lower) & (log_ell <= upper), density, 0.0)
    z = (np.log(ell_grid) - m_ell) / s_ell
    return np.exp(-0.5 * z**2) / (
        ell_grid * s_ell * math.sqrt(2.0 * math.pi)
    )


def _normalized_density(values: np.ndarray) -> np.ndarray:
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(max_value) or max_value <= 0.0:
        return np.zeros_like(values)
    return values / max_value


def _chain_array(values, *, num_chains: int) -> np.ndarray:
    array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        if array.size % num_chains != 0:
            raise ValueError(f"Cannot reshape {array.size} samples into {num_chains} chains.")
        return array.reshape(num_chains, array.size // num_chains)
    if array.ndim >= 2:
        return array.reshape(array.shape[0], array.shape[1], -1)[:, :, 0]
    raise ValueError(f"Unsupported sample shape: {array.shape}")


def _load_hmc_trace_chains(cell_path: Path) -> tuple[dict[str, np.ndarray], int]:
    import torch

    payload = torch.load(cell_path, map_location="cpu", weights_only=False)
    samples = payload["canonical_nuts_samples"] or payload["nuts_samples"]
    model = payload.get("model", {})
    num_chains = int(model.get("num_chains", 1))
    transformed = {
        "log_ell": _chain_array(samples["theta_ell"], num_chains=num_chains),
        "log_w": _chain_array(samples["theta_w"], num_chains=num_chains),
        "log_sigma_f": _chain_array(samples["theta_sf"], num_chains=num_chains),
        "log_sigma_d": _chain_array(samples["theta_sd"], num_chains=num_chains),
    }
    return transformed, num_chains


def _load_noise_checkpoint(results_dir: Path, *, noise_model: str, objective: str = "LOO") -> dict:
    import torch

    stems = {
        "fixed": "hgp_fixed_noise_checkpoint.pt",
        "inferred": "hgp_full_checkpoint.pt",
    }
    if noise_model not in stems:
        raise ValueError(f"Unknown noise model: {noise_model}")
    path = results_dir / objective.upper() / stems[noise_model]
    if not path.exists():
        raise FileNotFoundError(f"Missing noise checkpoint: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _noise_physical_samples(checkpoint: dict) -> dict[str, np.ndarray]:
    samples = checkpoint["samples"]
    mapping = {
        "ell": "theta_ell",
        "w": "theta_w",
        "sigma_f": "theta_sf",
        "sigma_d": "theta_sd",
    }
    out: dict[str, np.ndarray] = {}
    for name, key in mapping.items():
        if key in samples:
            values = samples[key]
            values = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
            values = np.exp(np.asarray(values, dtype=float).reshape(-1))
            out[name] = values[np.isfinite(values) & (values > 0.0)]
    return out


def _noise_predictive(checkpoint: dict) -> dict[str, np.ndarray]:
    variance = checkpoint["total_variance"]
    mean = checkpoint["mean"]
    x_test = checkpoint["x_test"]
    variance = variance.detach().cpu().numpy() if hasattr(variance, "detach") else np.asarray(variance)
    mean = mean.detach().cpu().numpy() if hasattr(mean, "detach") else np.asarray(mean)
    x_test = x_test.detach().cpu().numpy() if hasattr(x_test, "detach") else np.asarray(x_test)
    return {
        "x": np.asarray(x_test, dtype=float),
        "mean": np.asarray(mean, dtype=float),
        "std": np.sqrt(np.clip(np.asarray(variance, dtype=float), 0.0, None)),
    }


def _load_wham_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing WHAM reference: {path}")
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"WHAM reference must have at least two columns: {path}")
    return np.asarray(data[:, 0], dtype=float), np.asarray(data[:, 1], dtype=float)


def _align_to_reference_minimum(mean: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return mean - float(np.nanmin(mean)) + float(np.nanmin(reference))


def _align_to_reference_tail(mean: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return mean - float(mean[-1]) + float(reference[-1])


def _log_kde_density(log_values: np.ndarray, log_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.zeros_like(log_grid)
    sample_std = float(np.std(values, ddof=1))
    bandwidth = max(sample_std * values.size ** (-1.0 / 5.0), 1e-3)
    z = (log_grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).mean(axis=1) / (bandwidth * math.sqrt(2.0 * math.pi))


def _log_kde_2d_density(
    log_x: np.ndarray,
    log_y: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    max_samples: int = 1200,
) -> np.ndarray:
    log_x = np.asarray(log_x, dtype=float)
    log_y = np.asarray(log_y, dtype=float)
    mask = np.isfinite(log_x) & np.isfinite(log_y)
    values = np.column_stack([log_x[mask], log_y[mask]])
    if values.shape[0] < 3:
        return np.zeros((y_grid.size, x_grid.size), dtype=float)
    if values.shape[0] > max_samples:
        indices = np.linspace(0, values.shape[0] - 1, max_samples, dtype=int)
        values = values[indices]
    std = np.std(values, axis=0, ddof=1)
    bandwidth = np.maximum(std * values.shape[0] ** (-1.0 / 6.0), 1e-3)
    dx = (x_grid[None, :] - values[:, 0, None]) / bandwidth[0]
    dy = (y_grid[None, :] - values[:, 1, None]) / bandwidth[1]
    density = np.exp(-0.5 * (dy[:, :, None] ** 2 + dx[:, None, :] ** 2)).mean(axis=0)
    density /= 2.0 * math.pi * bandwidth[0] * bandwidth[1]
    return density


def _linear_kde_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.zeros_like(grid)
    sample_std = float(np.std(values, ddof=1))
    bandwidth = max(sample_std * values.size ** (-1.0 / 5.0), 1e-12)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z**2).mean(axis=1) / (bandwidth * math.sqrt(2.0 * math.pi))


def _linear_kde_2d_density(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    max_samples: int = 1200,
) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    values = np.column_stack([x_values[mask], y_values[mask]])
    if values.shape[0] < 3:
        return np.zeros((y_grid.size, x_grid.size), dtype=float)
    if values.shape[0] > max_samples:
        indices = np.linspace(0, values.shape[0] - 1, max_samples, dtype=int)
        values = values[indices]
    std = np.std(values, axis=0, ddof=1)
    bandwidth = np.maximum(std * values.shape[0] ** (-1.0 / 6.0), 1e-12)
    dx = (x_grid[None, :] - values[:, 0, None]) / bandwidth[0]
    dy = (y_grid[None, :] - values[:, 1, None]) / bandwidth[1]
    density = np.exp(-0.5 * (dy[:, :, None] ** 2 + dx[:, None, :] ** 2)).mean(axis=0)
    density /= 2.0 * math.pi * bandwidth[0] * bandwidth[1]
    return density


def _noise_prior_density(parameter: str, grid: np.ndarray) -> np.ndarray:
    """Default physical-space hyperprior density for a positive parameter."""
    if parameter not in NOISE_PRIOR_LOG_PARAMS:
        return np.zeros_like(grid, dtype=float)
    mu, sigma = NOISE_PRIOR_LOG_PARAMS[parameter]
    grid = np.asarray(grid, dtype=float)
    density = np.zeros_like(grid, dtype=float)
    mask = grid > 0.0
    z = (np.log(grid[mask]) - mu) / sigma
    density[mask] = np.exp(-0.5 * z**2) / (grid[mask] * sigma * math.sqrt(2.0 * math.pi))
    return density


def _histogram2d_contour_density(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    bins: int = 26,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized 2D histogram density on bin centers for contours."""
    hist, x_edges, y_edges = np.histogram2d(
        x_values,
        y_values,
        bins=bins,
        range=[x_limits, y_limits],
        density=False,
    )
    hist = hist.T.astype(float)
    if np.nanmax(hist) > 0.0:
        hist /= np.nanmax(hist)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, hist


def _nice_linear_limits(values: np.ndarray, *, pad_fraction: float = 0.08) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(values, [0.005, 0.995])
    span = max(float(hi - lo), 1e-12)
    return float(lo - pad_fraction * span), float(hi + pad_fraction * span)


def _nice_log_limits(values: np.ndarray, *, pad_fraction: float = 0.08) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1e-3, 1.0
    log_min, log_max = np.quantile(np.log(values), [0.005, 0.995])
    span = max(float(log_max - log_min), 0.5)
    log_min -= pad_fraction * span
    log_max += pad_fraction * span
    return float(np.exp(log_min)), float(np.exp(log_max))


def _set_corner_ticks(
    ax,
    axis: str,
    values: np.ndarray,
    *,
    n_ticks: int = 3,
) -> None:
    lo, hi = _nice_linear_limits(values)
    span = hi - lo
    if span <= 0.0:
        ticks = np.full(n_ticks, lo)
    else:
        tick_pad = 0.08 * span
        ticks = np.linspace(lo + tick_pad, hi - tick_pad, n_ticks)
    labels = _corner_tick_labels(ticks)
    if axis == "x":
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
        ax.xaxis.set_minor_formatter(NullFormatter())
    elif axis == "y":
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter(labels))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        raise ValueError(axis)


def _corner_tick_labels(values: np.ndarray) -> list[str]:
    for sig_figs in (1, 2):
        labels = [_format_corner_tick(value, sig_figs=sig_figs) for value in values]
        if len(set(labels)) == len(labels):
            return labels
    return [_format_corner_tick(value, sig_figs=2) for value in values]


def _format_corner_tick(value: float, *, sig_figs: int) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e3:
        return f"{value:.{sig_figs - 1}e}"
    decimals = max(sig_figs - 1 - int(math.floor(math.log10(abs_value))), 0)
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def _draw_noise_corner(
    fig,
    gs_cell,
    samples: dict[str, np.ndarray],
    *,
    color: str,
    title: str,
    parameters: list[str] | None = None,
    origin_in: tuple[float, float] | None = None,
    cell_size_in: float | None = None,
    gap_in: float | None = None,
    y_label_x: float = -0.22,
) -> None:
    parameter_labels = {
        "ell": r"$\ell$ [nm]",
        "w": r"$w$ [kJ/mol]",
        "sigma_f": r"$\sigma_f$ [kJ/mol]",
        "sigma_d": r"$\sigma_d$ [kJ/mol/nm]",
    }
    parameters = parameters or ["ell", "w"]
    missing = [parameter for parameter in parameters if parameter not in samples]
    if missing:
        raise ValueError(f"Noise corner plot missing samples for: {', '.join(missing)}")

    n_params = len(parameters)
    if origin_in is None:
        sub = gs_cell.subgridspec(
            n_params,
            n_params,
            wspace=0.08,
            hspace=0.08,
        )
    else:
        if cell_size_in is None or gap_in is None:
            raise ValueError("Manual corner placement requires cell_size_in and gap_in.")
        fig_width, fig_height = fig.get_size_inches()
        left_in, bottom_in = origin_in
        sub = None
    limits = {parameter: _nice_linear_limits(samples[parameter]) for parameter in parameters}
    axes = np.empty((n_params, n_params), dtype=object)

    for row, y_parameter in enumerate(parameters):
        for col, x_parameter in enumerate(parameters):
            if sub is None:
                x_in = left_in + col * (cell_size_in + gap_in)
                y_in = bottom_in + (n_params - 1 - row) * (cell_size_in + gap_in)
                ax = fig.add_axes([
                    x_in / fig_width,
                    y_in / fig_height,
                    cell_size_in / fig_width,
                    cell_size_in / fig_height,
                ])
            else:
                ax = fig.add_subplot(sub[row, col])
            axes[row, col] = ax
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(PAPER_LINE_WIDTH)
            ax.set_box_aspect(1)

            if row == col:
                values = samples[x_parameter]
                lo, hi = limits[x_parameter]
                finite_values = values[np.isfinite(values)]
                hist_values, hist_edges = np.histogram(
                    finite_values,
                    bins=22,
                    range=(lo, hi),
                    density=True,
                )
                ax.stairs(
                    hist_values,
                    hist_edges,
                    fill=True,
                    color=color,
                    alpha=0.18,
                    linewidth=0.0,
                )
                ax.stairs(
                    hist_values,
                    hist_edges,
                    fill=False,
                    color=color,
                    linewidth=0.75,
                )
                grid = np.linspace(lo, hi, 300)
                prior_density = _noise_prior_density(x_parameter, grid)
                ax.plot(grid, prior_density, color=NOISE_PRIOR_COLOR, linewidth=0.85)
                ax.set_xlim(lo, hi)
                max_density = max(
                    float(np.nanmax(hist_values)) if hist_values.size else 0.0,
                    float(np.nanmax(prior_density)) if prior_density.size else 0.0,
                )
                ax.set_ylim(0.0, max_density * 1.12 if max_density > 0.0 else 1.0)
                ax.set_yticks([])
                if row == 0 and origin_in is None:
                    ax.set_title(title, pad=5, fontsize=PAPER_TITLE_SIZE)
            else:
                x_values = samples[x_parameter]
                y_values = samples[y_parameter]
                x_limits = limits[x_parameter]
                y_limits = limits[y_parameter]
                ax.set_xlim(*x_limits)
                ax.set_ylim(*y_limits)

                if row > col:
                    n_points = min(x_values.size, y_values.size)
                    if n_points > 650:
                        indices = np.linspace(0, n_points - 1, 650, dtype=int)
                    else:
                        indices = np.arange(n_points)
                    ax.scatter(
                        x_values[indices],
                        y_values[indices],
                        s=2.0,
                        color=color,
                        alpha=0.16,
                        linewidths=0.0,
                    )
                else:
                    contour_x, contour_y, density = _histogram2d_contour_density(
                        x_values,
                        y_values,
                        x_limits=x_limits,
                        y_limits=y_limits,
                        bins=26,
                    )
                    if np.nanmax(density) > 0.0:
                        ax.contour(
                            contour_x,
                            contour_y,
                            density,
                            levels=[0.12, 0.28, 0.45, 0.62],
                            colors=color,
                            linewidths=0.55,
                            alpha=0.85,
                        )

            if row < n_params - 1:
                ax.xaxis.set_major_formatter(NullFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(labelbottom=False)
            else:
                _set_corner_ticks(
                    ax,
                    "x",
                    samples[x_parameter],
                    n_ticks=3,
                )
                ax.set_xlabel(parameter_labels[x_parameter], labelpad=2, fontsize=PAPER_FONT_SIZE)

            if col > 0:
                ax.yaxis.set_major_formatter(NullFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(labelleft=False)
            elif row > 0:
                _set_corner_ticks(
                    ax,
                    "y",
                    samples[y_parameter],
                    n_ticks=3,
                )
                ax.set_ylabel(parameter_labels[y_parameter], labelpad=2, fontsize=PAPER_FONT_SIZE)
                ax.yaxis.set_label_coords(y_label_x, 0.5)
            else:
                ax.yaxis.set_major_formatter(NullFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(labelleft=False)

            ax.tick_params(axis="both", which="both", direction="in", labelsize=PAPER_TICK_SIZE)

    if origin_in is not None:
        block_width_in = n_params * cell_size_in + (n_params - 1) * gap_in
        block_top_in = bottom_in + block_width_in
        fig.text(
            (left_in + 0.5 * block_width_in) / fig_width,
            (block_top_in + 0.08) / fig_height,
            title,
            ha="center",
            va="bottom",
            fontsize=PAPER_TITLE_SIZE,
        )


def _draw_noise_predictive(
    ax,
    *,
    checkpoint: dict,
    color: str,
    label: str,
    wham_x: np.ndarray,
    wham_f: np.ndarray,
) -> None:
    ax.plot(wham_x, wham_f, color="black", linestyle=(0, (3, 2)), linewidth=0.85, label="WHAM")
    pred = _noise_predictive(checkpoint)
    reference = np.interp(pred["x"], wham_x, wham_f)
    mean = _align_to_reference_tail(pred["mean"], reference)
    std = pred["std"]
    ax.fill_between(
        pred["x"],
        mean - std,
        mean + std,
        color=color,
        alpha=0.12,
        linewidth=0.0,
    )
    ax.plot(pred["x"], mean, color=color, linewidth=1.05, label=label)

    y_pad = max(2.0, 0.08 * float(np.nanmax(wham_f) - np.nanmin(wham_f))) + 20.0
    ax.set_ylim(float(np.nanmin(wham_f) - y_pad), float(np.nanmax(wham_f) + y_pad))
    ax.set_xlabel("Position [nm]")
    ax.set_ylabel("Free Energy [kJ/mol]")
    ax.tick_params(axis="both", which="both", direction="in")
    ax.grid(False)
    ax.legend(fontsize=PAPER_TICK_SIZE, frameon=False, loc="best", handlelength=1.6)


def _plot_map_sampled_difference_preview(
    *,
    rmse_percent_diffs: dict[str, np.ndarray],
    std_percent_diffs: dict[str, np.ndarray],
    rmse_norm: mcolors.Normalize,
    std_norm: mcolors.Normalize,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    fig = plt.figure(figsize=SI_TWO_METHOD_FIGSIZE, constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        len(MAP_SAMPLED_PAIRS) + 1,
        width_ratios=[1.0] * len(MAP_SAMPLED_PAIRS) + [0.055],
        wspace=0.08,
        hspace=0.08,
    )
    row_specs = [
        (rmse_percent_diffs, rmse_norm, r"$\Delta$ [%]"),
        (std_percent_diffs, std_norm, r"$\Delta$ [%]"),
    ]
    for row, (diffs, norm, _label) in enumerate(row_specs):
        for col, (sampled_key, _map_key, title) in enumerate(MAP_SAMPLED_PAIRS):
            ax = fig.add_subplot(gs[row, col])
            key = title.lower()
            _draw_preview_heatmap(
                ax,
                diffs[key],
                norm=norm,
                cmap=args.cmap,
                show_xlabels=row == 1,
                show_ylabels=col == 0,
            )
            avg_diff = float(np.nanmean(diffs[key]))
            if row == 0:
                ax.set_title(f"({PANEL_LETTERS[col]}) {METHOD_SPECS[sampled_key].label}", pad=6)
            ax.text(
                0.96,
                0.06,
                rf"mean $\Delta$ = {_nice_tick_label(avg_diff)}%",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=PAPER_TICK_SIZE,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )

    for row, (_diffs, norm, label) in enumerate(row_specs):
        ticks = _difference_ticks(norm)
        _add_vertical_colorbar(
            fig,
            fig.add_subplot(gs[row, -1]),
            norm=norm,
            cmap=args.cmap,
            ticks=ticks,
            label=label,
            label_x=3.9,
        )
    fig.supxlabel("Number of umbrella windows", y=0.022, fontsize=PAPER_LABEL_SIZE)
    fig.supylabel("Trajectory length [%]", x=0.000, fontsize=PAPER_LABEL_SIZE)
    fig.subplots_adjust(left=0.090, right=0.900, bottom=0.115, top=0.925)
    _save_figure(
        fig,
        output_dir,
        "si_map_sampled_difference_preview",
        args.formats,
        args.dpi,
        tight=False,
    )
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
        power=args.main_text_color_power,
    )
    std_norm = _main_text_norm(
        std_grids,
        center=args.std_center,
        vmin=args.main_text_std_vmin,
        vmax=args.main_text_std_vmax,
        power=args.main_text_color_power,
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
        power=args.main_text_color_power,
    )
    std_norm = _main_text_norm(
        std_grids,
        center=args.std_center,
        vmin=args.main_text_std_vmin,
        vmax=args.main_text_std_vmax,
        power=args.main_text_color_power,
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


def plot_lengthscale_prior_sensitivity(args: argparse.Namespace) -> None:
    """Replot length-scale prior sensitivity from saved HMC artifacts."""
    results_dir = args.results_dir.resolve()
    data_path = results_dir / "prior_sensitivity_replot_data.npz"
    summary_path = results_dir / "run_summary.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Run scripts/compare_lengthscale_priors.py once to create it."
        )
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}.")

    output_dir = figure_output_dir(args, "lengthscale_prior_sensitivity")
    configure_main_text_matplotlib()

    data = np.load(data_path)
    summary = json.loads(summary_path.read_text())
    cases = summary["lengthscale_prior_cases"]
    for case in cases:
        case["label"] = (
            case["label"]
            .replace(r"\mathregular{N}", r"\mathcal{N}")
            .replace(r"\mathrm{N}", r"\mathcal{N}")
            .replace(r"\operatorname{N}", r"\mathcal{N}")
            .replace(r"\rm N", r"\mathcal{N}")
            .replace(r"\dot", r"\mathcal{N}")
            .replace(r"(\log 4, 1)", r"(\log 4, 1^2)")
        )
    log_bounds = tuple(float(value) for value in data["flat_log_ell_bounds"])
    ell_grid = np.asarray(data["ell_grid"], dtype=float)
    ell_kde_grid = np.asarray(data["ell_kde_grid"], dtype=float)
    density_xlim = (
        float(min(np.nanmin(ell_grid), np.nanmin(ell_kde_grid))),
        float(max(np.nanmax(ell_grid), np.nanmax(ell_kde_grid))),
    )

    fig = plt.figure(figsize=(JCTC_SINGLE_COLUMN_WIDTH_IN, 6.0), constrained_layout=False)
    outer_gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.85, 2.35],
        hspace=0.16,
    )
    ax_pmf = fig.add_subplot(outer_gs[0, 0])
    density_gs = outer_gs[1, 0].subgridspec(4, 1, hspace=0.11)
    density_axes = [fig.add_subplot(density_gs[index, 0]) for index in range(4)]
    for case in cases:
        slug = case["slug"]
        color = LENGTHSCALE_PRIOR_COLORS.get(slug, "black")
        linewidth = 1.5 if slug == "ell_0p5_very_narrow" else 1.1
        band_alpha = 0.08 if slug == "ell_0p5_very_narrow" else 0.045
        x = np.asarray(data[f"{slug}_x"], dtype=float)
        mean = np.asarray(data[f"{slug}_mean"], dtype=float)
        std = np.asarray(data[f"{slug}_std"], dtype=float)
        mean = mean - mean[-1]
        ax_pmf.plot(
            x,
            mean,
            color=color,
            linestyle="-",
            linewidth=linewidth,
            label=case["label"],
        )
        ax_pmf.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=band_alpha,
            linewidth=0.0,
        )

    for row, case in enumerate(cases):
        ax_density = density_axes[row]
        slug = case["slug"]
        color = LENGTHSCALE_PRIOR_COLORS.get(slug, "black")
        linewidth = 1.6 if slug == "ell_0p5_very_narrow" else 1.3

        prior_params = np.asarray(data[f"{slug}_prior_params"], dtype=float)
        s_ell = None if np.isnan(prior_params[1]) else float(prior_params[1])
        prior_density = _normalized_density(
            _ell_prior_density(
                ell_grid,
                m_ell=float(prior_params[0]),
                s_ell=s_ell,
                log_bounds=log_bounds,
            )
        )
        posterior_density = _normalized_density(
            _positive_log_kde(
                np.asarray(data[f"{slug}_ell_samples"], dtype=float), ell_kde_grid
            )
        )
        positive = prior_density > 0.0
        ax_density.plot(
            ell_grid[positive],
            prior_density[positive],
            color=color,
            linestyle=(0, (1.0, 1.4)),
            linewidth=1.2,
            alpha=0.75,
        )
        ax_density.plot(
            ell_kde_grid,
            posterior_density,
            color=color,
            linestyle="-",
            linewidth=linewidth,
        )
        ax_density.set_xscale("log")
        ax_density.set_xlim(density_xlim)
        ax_density.margins(x=0.0)
        ax_density.set_ylim(bottom=0.0, top=1.08)
        ax_density.set_yticks([])
        ax_density.grid(False)
        ax_density.text(
            0.98,
            0.78,
            case["label"],
            transform=ax_density.transAxes,
            ha="right",
            va="center",
            fontsize=PAPER_TICK_SIZE,
            color=color,
        )
        if row < len(density_axes) - 1:
            ax_density.set_xticklabels([])

    wham_x = np.asarray(data["wham_x"], dtype=float) if "wham_x" in data.files else np.asarray([])
    if wham_x.size:
        wham_f = np.asarray(data["wham_f"], dtype=float)
        wham_f = wham_f - wham_f[-1]
        ax_pmf.plot(
            wham_x,
            wham_f,
            color="black",
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            label="WHAM",
        )
    else:
        ui_x = np.asarray(data["ui_x"], dtype=float)
        if ui_x.size:
            ui_f = np.asarray(data["ui_f"], dtype=float)
            ui_f = ui_f - ui_f[-1]
            ui_stride = slice(None, None, 2)
            ax_pmf.errorbar(
                ui_x[ui_stride],
                ui_f[ui_stride],
                yerr=np.asarray(data["ui_e"], dtype=float)[ui_stride],
                color="black",
                linewidth=0.8,
                capsize=1.8,
                label="Block-averaged UI",
            )

    ax_pmf.set_xlabel("Position [nm]")
    ax_pmf.set_ylabel("Free Energy [kJ/mol]")
    ax_pmf.grid(False)
    ax_pmf.tick_params(axis="both", direction="in")
    ax_pmf.legend(fontsize=PAPER_TICK_SIZE, frameon=False)

    density_axes[-1].set_xlabel(r"Length scale $\ell$ [nm]")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.095, top=0.985)
    density_bottom = density_axes[-1].get_position().y0
    density_top = density_axes[0].get_position().y1
    fig.text(
        0.078,
        0.5 * (density_bottom + density_top),
        "Normalized Probability Density",
        rotation=90,
        va="center",
        ha="center",
        fontsize=PAPER_LABEL_SIZE,
    )
    _save_figure(fig, output_dir, "lengthscale_prior_sensitivity", args.formats, args.dpi)
    plt.close(fig)


def plot_hmc_calibration_traces(args: argparse.Namespace) -> None:
    """Export paper-ready NUTS trace plots from a calibration cell artifact."""
    results_dir = args.results_dir.resolve()
    cell_path = results_dir / args.hmc_trace_run / "artifacts" / "cells" / args.hmc_trace_cell
    if not cell_path.exists():
        raise FileNotFoundError(f"Missing HMC trace cell artifact: {cell_path}")

    output_dir = figure_output_dir(args, "hmc_calibration_traces")
    configure_main_text_matplotlib()
    chains, num_chains = _load_hmc_trace_chains(cell_path)
    parameter_specs = [
        ("log_ell", r"$\log \ell$"),
        ("log_w", r"$\log w$"),
        ("log_sigma_f", r"$\log \sigma_f$"),
        ("log_sigma_d", r"$\log \sigma_d$"),
    ]

    fig, axes = plt.subplots(
        len(parameter_specs),
        2,
        figsize=(JCTC_SINGLE_COLUMN_WIDTH_IN, 4.75),
        constrained_layout=False,
        gridspec_kw={"width_ratios": [2.25, 1.25]},
    )
    for row, (key, ylabel) in enumerate(parameter_specs):
        trace_ax = axes[row, 0]
        marginal_ax = axes[row, 1]
        values = chains[key]
        iterations = np.arange(1, values.shape[1] + 1)
        for chain_index in range(values.shape[0]):
            color = OKABE_ITO_CHAIN_COLORS[chain_index % len(OKABE_ITO_CHAIN_COLORS)]
            trace_ax.plot(
                iterations,
                values[chain_index],
                color=color,
                linewidth=0.45,
                alpha=0.9,
                label=f"Chain {chain_index + 1}" if row == 0 else None,
            )
            marginal_ax.hist(
                values[chain_index],
                bins=28,
                density=True,
                histtype="step",
                color=color,
                linewidth=0.8,
                alpha=0.95,
            )
        trace_ax.set_ylabel(ylabel)
        trace_ax.yaxis.set_label_coords(-0.18, 0.5)
        trace_ax.grid(False)
        trace_ax.tick_params(axis="both", direction="in")
        marginal_ax.grid(False)
        marginal_ax.set_yticks([])
        marginal_ax.tick_params(axis="both", direction="in")
        finite_values = values[np.isfinite(values)]
        x_min, x_max = np.quantile(finite_values, [0.005, 0.995])
        x_padding = 0.08 * (x_max - x_min) if x_max > x_min else 0.5
        x_min -= x_padding
        x_max += x_padding
        marginal_ax.set_xlim(x_min, x_max)
        marginal_ax.set_xticks(np.linspace(x_min, x_max, 4))
        marginal_ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if row < len(parameter_specs) - 1:
            trace_ax.tick_params(labelbottom=False)

    axes[-1, 0].set_xlabel("NUTS sample")
    y_min, y_max = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(y_min, y_max + 0.18 * (y_max - y_min))
    axes[0, 0].set_title("(a) NUTS trace plots", pad=6)
    axes[0, 1].set_title("(b) Posterior Marginals", pad=6)
    axes[0, 0].legend(
        fontsize=PAPER_TICK_SIZE,
        frameon=False,
        ncol=2,
        loc="upper right",
        handlelength=1.6,
        columnspacing=0.8,
    )
    fig.subplots_adjust(
        left=0.22,
        right=0.995,
        bottom=0.095,
        top=0.965,
        hspace=0.22,
        wspace=0.09,
    )
    stem = f"hmc_calibration_traces_{args.hmc_trace_run}_{Path(args.hmc_trace_cell).stem}"
    _save_figure(fig, output_dir, stem, args.formats, args.dpi)
    plt.close(fig)


def plot_noise_comparison(args: argparse.Namespace) -> None:
    """Export a paper-ready fixed-vs-inferred noise hyperposterior figure."""
    results_dir = args.results_dir.resolve()
    output_dir = figure_output_dir(args, "noise_comparison")
    configure_main_text_matplotlib()

    fixed_checkpoint = _load_noise_checkpoint(results_dir, noise_model="fixed", objective="LOO")
    inferred_checkpoint = _load_noise_checkpoint(results_dir, noise_model="inferred", objective="LOO")
    fixed_samples = _noise_physical_samples(fixed_checkpoint)
    inferred_samples = _noise_physical_samples(inferred_checkpoint)

    wham_path = args.noise_reference_wham_path
    if not wham_path.is_absolute():
        wham_path = Path.cwd() / wham_path
    wham_x, wham_f = _load_wham_reference(wham_path)

    fig_width = JCTC_DOUBLE_COLUMN_WIDTH_IN
    fig_height = 4.22
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    corner_cell_in = 0.86
    corner_gap_in = 0.05
    corner_top_in = 3.95
    fixed_block_width_in = 2 * corner_cell_in + corner_gap_in
    left_column_in = fixed_block_width_in
    fixed_left_in = 0.58
    inferred_left_in = 3.18
    fixed_bottom_in = corner_top_in - (2 * corner_cell_in + corner_gap_in)
    inferred_bottom_in = corner_top_in - (4 * corner_cell_in + 3 * corner_gap_in)
    left_bottom_in = inferred_bottom_in

    _draw_noise_corner(
        fig,
        None,
        fixed_samples,
        color=NOISE_MODEL_COLORS["fixed"],
        title="(a) Fixed noise",
        origin_in=(fixed_left_in, fixed_bottom_in),
        cell_size_in=corner_cell_in,
        gap_in=corner_gap_in,
    )
    predictive_left_in = fixed_left_in
    predictive_width_in = fixed_block_width_in
    predictive_height_in = 0.55
    predictive_gap_in = 0.04
    inferred_pred_ax = fig.add_axes([
        predictive_left_in / fig_width,
        left_bottom_in / fig_height,
        predictive_width_in / fig_width,
        predictive_height_in / fig_height,
    ])
    fixed_pred_ax = fig.add_axes([
        predictive_left_in / fig_width,
        (left_bottom_in + predictive_height_in + predictive_gap_in) / fig_height,
        predictive_width_in / fig_width,
        predictive_height_in / fig_height,
    ])
    predictive_top_in = left_bottom_in + 2 * predictive_height_in + predictive_gap_in
    fig.text(
        (predictive_left_in + 0.5 * predictive_width_in) / fig_width,
        (predictive_top_in + 0.08) / fig_height,
        "(c) Posterior predictive",
        ha="center",
        va="bottom",
        fontsize=PAPER_TITLE_SIZE,
    )
    _draw_noise_predictive(
        fixed_pred_ax,
        checkpoint=fixed_checkpoint,
        color=NOISE_MODEL_COLORS["fixed"],
        label="Fixed noise",
        wham_x=wham_x,
        wham_f=wham_f,
    )
    _draw_noise_predictive(
        inferred_pred_ax,
        checkpoint=inferred_checkpoint,
        color=NOISE_MODEL_COLORS["inferred"],
        label="Inferred noise",
        wham_x=wham_x,
        wham_f=wham_f,
    )
    fixed_pred_ax.tick_params(labelbottom=False)
    fixed_pred_ax.set_xlabel("")
    fixed_pred_ax.set_ylabel("")
    inferred_pred_ax.set_ylabel("")
    fig.text(
        (predictive_left_in - 0.25) / fig_width,
        (left_bottom_in + predictive_height_in + 0.5 * predictive_gap_in) / fig_height,
        "Free Energy [kJ/mol]",
        rotation=90,
        va="center",
        ha="center",
        fontsize=PAPER_LABEL_SIZE,
    )
    _draw_noise_corner(
        fig,
        None,
        inferred_samples,
        color=NOISE_MODEL_COLORS["inferred"],
        title="(b) Inferred noise",
        parameters=["ell", "w", "sigma_f", "sigma_d"],
        origin_in=(inferred_left_in, inferred_bottom_in),
        cell_size_in=corner_cell_in,
        gap_in=corner_gap_in,
        y_label_x=-0.22,
    )
    _save_figure(fig, output_dir, "noise_comparison", args.formats, args.dpi, tight=False)
    plt.close(fig)


def plot_ablation_parity(args: argparse.Namespace) -> None:
    """Plot RMS predictive SD against RMSE to WHAM for selected methods."""
    results_dir = args.results_dir.resolve()
    output_dir = figure_output_dir(args, "ablation_parity")
    configure_main_text_matplotlib()

    fig, ax = plt.subplots(figsize=(JCTC_SINGLE_COLUMN_WIDTH_IN, 2.75), constrained_layout=False)
    summary_rows: list[dict[str, object]] = []
    all_values: list[float] = []

    for method_key in PARITY_METHODS:
        spec = METHOD_SPECS[method_key]
        csv_path = results_dir / spec.relative_csv
        rows = _read_csv_rows(csv_path)
        pred_rms_std = np.array(
            [
                math.sqrt(max(_metric_value(row, "avg_total_variance"), 0.0))
                for row in rows
            ],
            dtype=float,
        )
        true_rmse = np.array([_metric_value(row, "rmse_wham") for row in rows], dtype=float)
        mask = (
            np.isfinite(pred_rms_std)
            & np.isfinite(true_rmse)
            & (pred_rms_std > 0.0)
            & (true_rmse > 0.0)
        )
        pred_rms_std = pred_rms_std[mask]
        true_rmse = true_rmse[mask]
        all_values.extend(pred_rms_std.tolist())
        all_values.extend(true_rmse.tolist())

        ax.scatter(
            true_rmse,
            pred_rms_std,
            s=13.0,
            color=PARITY_METHOD_COLORS[method_key],
            alpha=0.72,
            linewidths=0.25,
            edgecolors="white",
            label=spec.label,
        )
        summary_rows.append(
            {
                "method": spec.label,
                "n": int(mask.sum()),
                "mean_rmse_wham": float(np.mean(true_rmse)),
                "mean_rms_std": float(np.mean(pred_rms_std)),
                "median_rmse_wham": float(np.median(true_rmse)),
                "median_rms_std": float(np.median(pred_rms_std)),
            }
        )

    finite = np.array([value for value in all_values if np.isfinite(value) and value >= 0.0])
    upper = min(30.0, float(np.nanmax(finite) * 1.05))
    upper = max(upper, 12.0)
    lo = 0.0
    hi = upper
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.75, label="Parity")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("RMSE vs WHAM [kJ/mol]", labelpad=2)
    ax.set_ylabel("RMS Predictive SD [kJ/mol]", labelpad=2)
    ax.tick_params(axis="both", which="both", direction="in", labelsize=PAPER_TICK_SIZE)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(PAPER_LINE_WIDTH)
    ax.legend(
        fontsize=PAPER_TICK_SIZE,
        frameon=False,
        loc="lower right",
        handlelength=1.0,
        handletextpad=0.35,
        borderpad=0.2,
        labelspacing=0.25,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.98)

    summary_path = output_dir / "ablation_parity_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "n",
                "mean_rmse_wham",
                "mean_rms_std",
                "median_rmse_wham",
                "median_rms_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    _save_figure(fig, output_dir, "ablation_parity", args.formats, args.dpi, tight=False)
    plt.close(fig)


def _metadynamics_output_dir(args: argparse.Namespace, module_name: str) -> Path:
    root = (args.output_dir or args.metadynamics_results_dir / "paper_figures").resolve()
    out = root / module_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_metadynamics_convergence(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        metrics_path = path.with_name("trajectory_length_metrics.csv")
        if metrics_path.exists():
            rows = _read_csv_rows(metrics_path)
            out = []
            for row in rows:
                if row.get("method") not in {"hmc_gp", "map_gp"}:
                    continue
                try:
                    out.append(
                        {
                            "trajectory_fraction": float(row["fraction"]),
                            "rmse": float(row["rmse_kj_mol"]),
                            "std": float(row["avg_std_kj_mol"]),
                            "correlation": float("nan"),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            if out:
                return out
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    correlation_key = next((key for key in rows[0] if key.lower().startswith("correlation")), None)
    required = {"Time", "RMSE", "STD"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")
    out: list[dict[str, float]] = []
    for row in rows:
        out.append(
            {
                "trajectory_fraction": float(row["Time"]),
                "rmse": float(row["RMSE"]),
                "std": float(row["STD"]),
                "correlation": float(row[correlation_key]) if correlation_key else float("nan"),
            }
        )
    return out


def _load_metadynamics_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments=("#", "@", ";"))
    x = data[:, 0]
    free_energy = data[:, 1]
    mask = (x >= 0.0) & (x <= 4.5)
    x = x[mask]
    free_energy = free_energy[mask]
    free_energy = free_energy - free_energy[-1]
    return x, free_energy


def _load_metadynamics_pmf(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.asarray(data["x"]), np.asarray(data["mean"]), np.asarray(data["std"])


def _metadynamics_pmf_metrics(
    pmf_path: Path,
    ref_x: np.ndarray,
    ref_f: np.ndarray,
) -> tuple[float, float]:
    px, mean, pstdev = _load_metadynamics_pmf(pmf_path)
    mean = mean + (float(ref_f[-1]) - float(mean[-1]))
    ref_interp = np.interp(px, ref_x, ref_f)
    rmse = float(np.sqrt(np.nanmean((mean - ref_interp) ** 2)))
    avg_sd = float(np.nanmean(pstdev))
    return rmse, avg_sd


def _load_metadynamics_method_metrics(
    metad_dir: Path,
    *,
    fallback_rows: list[dict[str, float]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    metrics_path = metad_dir / "trajectory_length_metrics.csv"
    if not metrics_path.exists():
        rows = sorted(fallback_rows, key=lambda row: row["trajectory_fraction"])
        return {
            "hierarchical": (
                np.array([100.0 * row["trajectory_fraction"] for row in rows], dtype=float),
                np.array([row["rmse"] for row in rows], dtype=float),
                np.array([row["std"] for row in rows], dtype=float),
            )
        }

    rows = _read_csv_rows(metrics_path)
    raw: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        method = row.get("method", "")
        if method not in {"hmc_gp", "map_gp", "fixed_gp"}:
            continue
        try:
            x = 100.0 * float(row["fraction"])
            rmse = float(row["rmse_kj_mol"])
            std = float(row["avg_std_kj_mol"])
        except (KeyError, TypeError, ValueError):
            continue
        raw.setdefault(method, []).append((x, rmse, std))

    out: dict[str, list[tuple[float, float, float]]] = {}
    if raw.get("hmc_gp"):
        out["hierarchical"] = raw["hmc_gp"]
    elif raw.get("map_gp"):
        out["hierarchical"] = raw["map_gp"]
    if raw.get("fixed_gp"):
        out["fixed"] = raw["fixed_gp"]
    packed: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for method, values in out.items():
        values = sorted(values)
        packed[method] = (
            np.array([value[0] for value in values], dtype=float),
            np.array([value[1] for value in values], dtype=float),
            np.array([value[2] for value in values], dtype=float),
        )
    return packed


def _metadynamics_pmf_path(metad_dir: Path, fraction: float, method: str) -> Path | None:
    candidates = []
    if method == "hierarchical":
        candidates.extend([
            metad_dir / f"{fraction:.3f}_hmc_gp_pmf.csv",
            metad_dir / f"{fraction:.3f}_map_gp_pmf.csv",
            metad_dir / f"{fraction:.2f}metadynamics_gprhd_pmf.csv",
        ])
    elif method == "fixed":
        candidates.extend([
            metad_dir / f"{fraction:.3f}_fixed_gp_pmf.csv",
            metad_dir / f"{fraction:.2f}metadynamics_gprhd_pmf_fixed.csv",
        ])
    for path in candidates:
        if path.exists():
            return path
    return None


def plot_metadynamics_convergence(args: argparse.Namespace) -> None:
    """Plot metadynamics convergence metrics and representative posterior predictives."""
    metad_dir = args.metadynamics_results_dir.resolve()
    output_dir = _metadynamics_output_dir(args, "metadynamics_convergence")
    configure_main_text_matplotlib()

    rows = _load_metadynamics_convergence(metad_dir / "convergence_data.csv")
    selected = sorted(rows, key=lambda row: row["trajectory_fraction"])
    if not selected:
        raise ValueError("No metadynamics convergence rows found.")

    reference_path = metad_dir / "fes.dat.csv"
    if not reference_path.exists():
        fallback_reference = Path("results/metadynamics/fes.dat.csv").resolve()
        if fallback_reference.exists():
            reference_path = fallback_reference
    if not reference_path.exists():
        fallback_reference = Path("/home/bshanks/freeGP-datasets/membranes/zuzka_metadynamics/metad/fes.dat")
        if fallback_reference.exists():
            reference_path = fallback_reference
    ref_x, ref_f = _load_metadynamics_reference(reference_path)
    method_metrics = _load_metadynamics_method_metrics(metad_dir, fallback_rows=selected)
    hierarchical_metrics = method_metrics.get("hierarchical")
    if hierarchical_metrics is None:
        raise ValueError("No hierarchical/metadynamics GP metrics found.")
    x_percent, rmse, std = hierarchical_metrics
    fixed_metrics = method_metrics.get("fixed")
    predictive_cmap = plt.get_cmap("viridis")
    predictive_specs = [
        (0.01, "1%", predictive_cmap(0.12)),
        (0.25, "25%", predictive_cmap(0.55)),
        (1.00, "100%", predictive_cmap(0.88)),
    ]
    if fixed_metrics is None:
        fixed_x = fixed_rmse = fixed_std = np.array([], dtype=float)
    else:
        fixed_x, fixed_rmse, fixed_std = fixed_metrics

    fig = plt.figure(
        figsize=(JCTC_SINGLE_COLUMN_WIDTH_IN, 4.75),
        constrained_layout=False,
    )
    gs = fig.add_gridspec(
        5,
        1,
        height_ratios=[1.28, 1.28, 0.22, 0.82, 0.82],
        hspace=0.15,
    )
    axes = [fig.add_subplot(gs[index]) for index in (0, 1, 3, 4)]

    predictive_panels = [
        (axes[0], "fixed", "(a)", "Fixed GP"),
        (axes[1], "hierarchical", "(b)", "Hierarchical GP (LOO)"),
    ]
    for ax, method_key, panel_label, method_label in predictive_panels:
        ax.plot(
            ref_x,
            ref_f,
            color="0.15",
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            label="Reference",
            zorder=5,
        )
        for fraction, label, color in predictive_specs:
            path = _metadynamics_pmf_path(metad_dir, fraction, method_key)
            if path is None:
                continue
            px, mean, pstdev = _load_metadynamics_pmf(path)
            mean = mean + (float(ref_f[-1]) - float(mean[-1]))
            lower = mean - pstdev
            upper = mean + pstdev
            ax.plot(px, mean, color=color, linewidth=0.8, label=label)
            ax.fill_between(px, lower, upper, color=color, alpha=0.13, linewidth=0.0)
        ax.set_ylabel("Free Energy [kJ/mol]", labelpad=2)
        ax.set_ylim(-25.0, 35.0)
        ax.set_xlim(float(np.nanmin(ref_x)), float(np.nanmax(ref_x)))
        ax.text(
            0.03,
            0.93,
            f"{panel_label} {method_label}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=PAPER_TITLE_SIZE,
        )
        ax.tick_params(axis="both", which="both", direction="in", labelsize=PAPER_TICK_SIZE)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(PAPER_LINE_WIDTH)
    axes[1].set_xlabel("Position [nm]", labelpad=2)
    axes[1].xaxis.set_label_coords(0.5, -0.18)
    axes[0].tick_params(labelbottom=False)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend([
        plt.Line2D([0], [0], color="0.25", linewidth=0.8),
        Patch(facecolor="0.5", alpha=0.18, edgecolor="none"),
    ])
    labels.extend(["Mean", r"$\pm 1\sigma$"])
    axes[0].legend(
        handles,
        labels,
        loc="lower right",
        ncol=3,
        frameon=False,
        fontsize=PAPER_TICK_SIZE,
        handlelength=1.1,
        columnspacing=0.55,
        handletextpad=0.35,
        borderpad=0.1,
        labelspacing=0.2,
    )

    series = [
        (axes[2], rmse, fixed_rmse, "RMSE [kJ/mol]", "#0072B2", "(c)"),
        (axes[3], std, fixed_std, "Avg. SD [kJ/mol]", "#0072B2", ""),
    ]
    for ax, values, fixed_values, ylabel, color, panel_label in series:
        if len(fixed_x):
            ax.plot(
                fixed_x,
                fixed_values,
                color="#CC79A7",
                marker="s",
                markersize=2.25,
                linewidth=1.05,
                linestyle="--",
                label="Fixed GP",
            )
        ax.plot(
            x_percent,
            values,
            color=color,
            marker="o",
            markersize=2.4,
            linewidth=1.05,
            label="Hierarchical GP (LOO)",
        )
        if panel_label:
            ax.text(
                0.03,
                0.93,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PAPER_TITLE_SIZE,
            )
        ax.set_ylabel(ylabel, labelpad=2)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=PAPER_TICK_SIZE)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(PAPER_LINE_WIDTH)

    axes[3].set_xlabel("Trajectory Length [%]", labelpad=2)
    shown_xticks = np.array([0.0, 25.0, 50.0, 75.0, 100.0], dtype=float)
    shown_xticks = shown_xticks[(shown_xticks >= np.nanmin(x_percent)) & (shown_xticks <= np.nanmax(x_percent))]
    for ax in axes[2:4]:
        ax.set_xlim(float(np.nanmin(x_percent)) - 1.5, 101.0)
        ax.set_xticks(shown_xticks)
        ax.set_xticklabels([f"{value:g}" for value in shown_xticks])
    for label in axes[3].get_xticklabels():
        label.set_rotation(0)
    axes[3].xaxis.set_label_coords(0.5, -0.18)
    axes[2].tick_params(labelbottom=False)
    axes[2].legend(
        loc="upper right",
        ncol=1,
        frameon=False,
        fontsize=PAPER_TICK_SIZE,
        handlelength=1.2,
        handletextpad=0.35,
        borderpad=0.1,
        labelspacing=0.2,
    )

    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.145, right=0.99, bottom=0.065, top=0.99)

    summary_path = output_dir / "metadynamics_convergence_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trajectory_fraction", "trajectory_percent", "rmse", "std", "correlation"],
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "trajectory_fraction": row["trajectory_fraction"],
                    "trajectory_percent": 100.0 * row["trajectory_fraction"],
                    "rmse": row["rmse"],
                    "std": row["std"],
                    "correlation": row["correlation"],
                }
            )

    _save_figure(fig, output_dir, "metadynamics_convergence", args.formats, args.dpi, tight=False)
    plt.close(fig)


FIGURE_MODULES = {
    "ablation_heatmaps": plot_ablation_heatmaps,
    "main_text_ablation_grids": plot_main_text_ablation_grids,
    "ablation_parity": plot_ablation_parity,
    "metadynamics_convergence": plot_metadynamics_convergence,
    "si_lml_loo_ablation": plot_si_lml_loo_ablation,
    "si_map_sampled_difference": plot_si_map_sampled_difference,
    "lengthscale_prior_sensitivity": plot_lengthscale_prior_sensitivity,
    "hmc_calibration_traces": plot_hmc_calibration_traces,
    "noise_comparison": plot_noise_comparison,
}


def main() -> None:
    args = build_parser().parse_args()
    for figure_name in args.figures:
        FIGURE_MODULES[figure_name](args)


if __name__ == "__main__":
    main()
