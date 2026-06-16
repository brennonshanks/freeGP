#!/usr/bin/env python3
"""Run an ablation-grid study using Umbrella Integration + block averaging.

Replicates the window and trajectory selection grid from the freeGP ablation
experiment (membrane.toml: 10×10 grid, 5 replicates, random_subset windows,
contiguous trajectory truncation, seed=42) and applies the Umbrella Integration
(UI) method with block-averaging error estimation as a reference/comparison.

Outputs (all under ``--results-dir``):
    ui/ablation_metrics.csv
        One row per grid cell; columns match freeGP's ablation_metrics.csv so
        that replot_grids.py can consume this file directly as a fourth method.

    ui/curves/w{WW}_f{FF}.csv
        Per-cell PMF curve: x_nm, pmf_mean, pmf_std, pmf_within_std,
        pmf_between_std, pmf_variance, pmf_within_variance, pmf_between_variance.
        All energy columns in kJ/mol (std) or kJ/mol² (variance).

    ui/references.csv
        Reference WHAM and UI PMF curves from the saved artifacts.

    ui/run_summary.txt
        Human-readable method summary.

Usage:
    python run_ui_ablation_grid.py \\
        --dataset-root /path/to/katka \\
        --reference-npz /path/to/membrane-ablation-*/lml/artifacts/references.npz \\
        --results-dir   /path/to/results/membrane-ablation-ui

    # or using the membrane.toml defaults:
    python run_ui_ablation_grid.py \\
        --dataset-root ~/projects/datasets/free-energy/membranes/katka \\
        --reference-npz ~/projects/freeGP/results/membrane-ablation-10x10-5replicates/lml/artifacts/references.npz \\
        --results-dir   ~/projects/freeGP/results/membrane-ablation-ui-block-avg
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# ── freeGP data-loading imports ───────────────────────────────────────────────
# We rely only on freeGP's I/O layer (no GP or MCMC machinery).
_pkg_root = Path(__file__).resolve().parent / "src"
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from freegp.data import load_umbrella_windows  # noqa: E402

# ── UI implementation ─────────────────────────────────────────────────────────
from umbrella_integration import (  # noqa: E402
    UIResult,
    UIAblationResult,
    aggregate_ui_replicates,
    rmse_vs_reference,
    shift_pmf,
    umbrella_integration,
)

# ── Ablation-grid parameters (from membrane.toml) ─────────────────────────────
WINDOW_COUNTS       = [25, 22, 19, 16, 13, 10, 8, 6, 4, 3]
TRAJ_FRACTIONS      = [1.0, 0.9, 0.8, 0.6, 0.4, 0.25, 0.16, 0.1, 0.063, 0.04]
RANDOM_SEED         = 42
N_EQUILIBRATION     = 40_000   # frames to discard at the start of each window
NUM_TEST_POINTS     = 100      # number of PMF evaluation points
N_REPLICATES        = 5        # 1 canonical (evenly-spaced) + 4 random-subset
N_BLOCKS            = 10       # blocks for block averaging per replicate


# ── Window / trajectory selection (mirrors freegp.studies.ablation) ──────────

def _select_evenly_spaced(windows, keep_count: int):
    if keep_count >= len(windows):
        return list(windows)
    indices = np.linspace(0, len(windows) - 1, keep_count, dtype=int)
    indices = np.unique(indices)
    return [windows[int(idx)] for idx in indices]


def _select_random(windows, keep_count: int, *, rng: np.random.Generator):
    if keep_count >= len(windows):
        return list(windows)
    indices = np.sort(rng.choice(len(windows), size=keep_count, replace=False))
    return [windows[int(idx)] for idx in indices]


def _truncate_positions(
    position,          # torch.Tensor or array-like
    *,
    n_equilibration: int,
    retain_fraction: float,
) -> np.ndarray:
    """Remove equilibration frames, keep a contiguous front fraction."""
    pos = np.asarray(position, dtype=float)
    pos_eq = pos[n_equilibration:] if len(pos) > n_equilibration else pos
    if pos_eq.size == 0:
        return pos_eq
    n_keep = max(2, int(np.floor(pos_eq.size * retain_fraction)))
    n_keep = min(pos_eq.size, n_keep)
    return pos_eq[:n_keep]


def _replicate_plan(n_replicates: int) -> list[dict]:
    """Build the replicate plan matching ablation.py's _replicate_plan logic.

    Replicate 0 is the *canonical* run (evenly-spaced windows, contiguous
    trajectory) so that the random-selection noise is visible relative to a
    deterministic baseline.  Replicates 1-N use random_subset selection.
    """
    plan = [
        {
            "replicate_index": 0,
            "is_canonical": True,
            "window_mode": "evenly_spaced",
        }
    ]
    for rep in range(1, n_replicates):
        plan.append(
            {
                "replicate_index": rep,
                "is_canonical": False,
                "window_mode": "random_subset",
            }
        )
    return plan


def _cell_seed(
    random_seed: int,
    i: int,
    j: int,
    window_count: int,
    rep: int,
) -> int:
    """Reproduce the per-cell seed from ablation.py exactly."""
    return int(random_seed + 1009 * i + 9173 * j + 7919 * window_count + 101 * rep)


# ── Metric helpers ────────────────────────────────────────────────────────────

def _cell_slug(window_count: int, traj_frac: float) -> str:
    frac_str = f"{traj_frac:.2f}".replace(".", "p")
    return f"w{window_count:02d}_f{frac_str}"


def _avg_std(variance: np.ndarray) -> float:
    return float(np.sqrt(np.clip(variance, 0.0, None)).mean())


# ── Main ablation loop ────────────────────────────────────────────────────────

def run_ui_ablation(
    dataset_root: str,
    reference_npz: str,
    results_dir: str,
    *,
    window_counts: list[int] = WINDOW_COUNTS,
    traj_fractions: list[float] = TRAJ_FRACTIONS,
    random_seed: int = RANDOM_SEED,
    n_equilibration: int = N_EQUILIBRATION,
    num_test_points: int = NUM_TEST_POINTS,
    n_replicates: int = N_REPLICATES,
    n_blocks: int = N_BLOCKS,
    pmf_alignment: str = "max",
) -> None:

    out_root = Path(results_dir).expanduser().resolve() / "ui"
    out_root.mkdir(parents=True, exist_ok=True)
    curves_dir = out_root / "curves"
    curves_dir.mkdir(exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading umbrella windows …")
    windows = load_umbrella_windows(dataset_root)
    print(f"  {len(windows)} windows loaded.")

    ref = np.load(Path(reference_npz).expanduser().resolve())
    wham_x   = ref["wham_x"]   if "wham_x"   in ref else None
    wham_f   = ref["wham_f"]   if "wham_f"   in ref else None
    wham_e   = ref["wham_e"]   if "wham_e"   in ref else None
    ui_x     = ref["umbrella_x"] if "umbrella_x" in ref else None
    ui_f     = ref["umbrella_f"] if "umbrella_f" in ref else None
    ui_e     = ref["umbrella_e"] if "umbrella_e" in ref else None

    has_wham = wham_x is not None
    has_ui   = ui_x is not None

    # Build test grid spanning both reference curves (same as freeGP logic).
    x_mins, x_maxs = [], []
    if has_wham:
        x_mins.append(float(wham_x.min()))
        x_maxs.append(float(wham_x.max()))
    if has_ui:
        x_mins.append(float(ui_x.min()))
        x_maxs.append(float(ui_x.max()))
    x_min = min(x_mins) if x_mins else None
    x_max = max(x_maxs) if x_maxs else None

    if x_min is None or x_max is None:
        # Fallback: span the umbrella centres.
        centres = np.array([float(w.folder_number) for w in windows])
        x_min, x_max = float(centres.min()), float(centres.max())

    x_test = np.linspace(x_min, x_max, num_test_points)
    print(f"  Test grid: {num_test_points} points from {x_min:.4f} to {x_max:.4f} nm")

    # Save reference curves.
    _save_references_csv(out_root, wham_x, wham_f, wham_e, ui_x, ui_f, ui_e)

    # ── Ablation loop ─────────────────────────────────────────────────────
    plan = _replicate_plan(n_replicates)
    rows: list[dict] = []

    total_cells = len(window_counts) * len(traj_fractions)
    cell_idx = 0

    for i, wc in enumerate(window_counts):
        for j, tf in enumerate(traj_fractions):
            cell_idx += 1
            slug = _cell_slug(wc, tf)
            print(f"[{cell_idx}/{total_cells}] cell {slug} …", flush=True)

            rep_results: list[UIResult] = []

            for rep_spec in plan:
                rep      = rep_spec["replicate_index"]
                seed     = _cell_seed(random_seed, i, j, wc, rep)
                rng      = np.random.default_rng(seed)
                win_mode = rep_spec["window_mode"]

                # Select windows.
                if win_mode == "evenly_spaced":
                    selected = _select_evenly_spaced(windows, wc)
                else:
                    selected = _select_random(windows, wc, rng=rng)

                # Build windows_data for UI (positions, x_eq, k).
                windows_data: list[tuple[np.ndarray, float, float]] = []
                for w in selected:
                    pos = _truncate_positions(
                        w.position,
                        n_equilibration=n_equilibration,
                        retain_fraction=tf,
                    )
                    if pos.size < 2:
                        continue
                    x_eq = float(w.folder_number)
                    k    = float(w.mdp_last_value)
                    windows_data.append((pos, x_eq, k))

                if not windows_data:
                    print(f"  WARNING: no usable windows for replicate {rep}, skipping.")
                    continue

                result = umbrella_integration(windows_data, x_test, n_blocks=n_blocks)
                rep_results.append(result)

            if not rep_results:
                print(f"  WARNING: all replicates failed for cell {slug}, skipping.")
                continue

            agg = aggregate_ui_replicates(rep_results)

            # Metrics.
            rmse_wham_val = (
                rmse_vs_reference(x_test, agg.pmf_mean, wham_x, wham_f, pmf_alignment)
                if has_wham else float("nan")
            )
            rmse_ui_val = (
                rmse_vs_reference(x_test, agg.pmf_mean, ui_x, ui_f, pmf_alignment)
                if has_ui else float("nan")
            )

            avg_total_var    = float(agg.total_variance.mean())
            avg_within_var   = float(agg.within_variance.mean())
            avg_between_var  = float(agg.between_variance.mean())
            avg_total_std    = _avg_std(agg.total_variance)
            avg_within_std   = _avg_std(agg.within_variance)
            avg_between_std  = _avg_std(agg.between_variance)

            rows.append({
                "window_count":         wc,
                "trajectory_fraction":  tf,
                "replicate_count":      len(rep_results),
                "rmse_wham":            rmse_wham_val,
                "rmse_ui":              rmse_ui_val,
                "avg_total_std":        avg_total_std,
                "avg_within_std":       avg_within_std,
                "avg_between_std":      avg_between_std,
                "avg_total_variance":   avg_total_var,
                "avg_within_variance":  avg_within_var,
                "avg_between_variance": avg_between_var,
            })

            _save_cell_curves_csv(curves_dir, slug, agg)

    # ── Write summary CSV ─────────────────────────────────────────────────
    _save_ablation_metrics_csv(out_root, rows)

    # ── Write run summary ─────────────────────────────────────────────────
    _save_run_summary(
        out_root,
        dataset_root=dataset_root,
        reference_npz=reference_npz,
        window_counts=window_counts,
        traj_fractions=traj_fractions,
        random_seed=random_seed,
        n_equilibration=n_equilibration,
        num_test_points=num_test_points,
        n_replicates=n_replicates,
        n_blocks=n_blocks,
        pmf_alignment=pmf_alignment,
        n_cells=len(rows),
    )

    print(f"\nDone. Results written to: {out_root}")
    print(f"  ablation_metrics.csv : {out_root / 'ablation_metrics.csv'}")
    print(f"  per-cell curves      : {curves_dir}")


# ── CSV writers ───────────────────────────────────────────────────────────────

def _save_ablation_metrics_csv(out_root: Path, rows: list[dict]) -> None:
    """Write ablation_metrics.csv with the same column set as freeGP's output.

    HMC-specific columns are left empty so the file is loadable by
    replot_grids.py without modification.
    """
    fieldnames = [
        "window_count",
        "trajectory_fraction",
        "replicate_count",
        "rmse_wham",
        "rmse_ui",
        "avg_total_std",
        "avg_within_std",
        "avg_between_std",
        "avg_total_variance",
        "avg_within_variance",
        "avg_between_variance",
        # HMC diagnostics — not applicable for UI; kept for schema compatibility.
        "barrier_height_mean",
        "barrier_height_std",
        "hmc_step_size",
        "hmc_mean_accept_prob",
        "hmc_accept_count",
        "hmc_divergence_count",
        "hmc_mean_sample_std",
        "hmc_max_sample_std",
        "hmc_min_sample_std",
        "hmc_poor_acceptance",
        "hmc_looks_stuck",
        "mcmc_max_r_hat",
        "mcmc_min_n_eff",
        "mcmc_divergence_total",
    ]
    path = out_root / "ablation_metrics.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Fill HMC columns with empty string.
            full_row = {k: "" for k in fieldnames}
            full_row.update(row)
            writer.writerow(full_row)
    print(f"  Wrote {path}")


def _save_cell_curves_csv(curves_dir: Path, slug: str, agg: UIAblationResult) -> None:
    """Save the per-cell PMF curve and associated uncertainties."""
    path = curves_dir / f"{slug}.csv"
    total_std   = np.sqrt(np.clip(agg.total_variance,   0.0, None))
    within_std  = np.sqrt(np.clip(agg.within_variance,  0.0, None))
    between_std = np.sqrt(np.clip(agg.between_variance, 0.0, None))

    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "x_nm",
            "pmf_mean_kJmol",
            "pmf_std_kJmol",
            "pmf_within_std_kJmol",
            "pmf_between_std_kJmol",
            "pmf_variance_kJmol2",
            "pmf_within_variance_kJmol2",
            "pmf_between_variance_kJmol2",
        ])
        for k in range(len(agg.x_test)):
            writer.writerow([
                agg.x_test[k],
                agg.pmf_mean[k],
                total_std[k],
                within_std[k],
                between_std[k],
                agg.total_variance[k],
                agg.within_variance[k],
                agg.between_variance[k],
            ])


def _save_references_csv(
    out_root: Path,
    wham_x, wham_f, wham_e,
    ui_x, ui_f, ui_e,
) -> None:
    """Save reference WHAM and UI curves to a single CSV file."""
    path = out_root / "references.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "x_nm", "pmf_kJmol", "error_kJmol"])
        if wham_x is not None:
            for k in range(len(wham_x)):
                e = float(wham_e[k]) if wham_e is not None else ""
                writer.writerow(["wham", wham_x[k], wham_f[k], e])
        if ui_x is not None:
            for k in range(len(ui_x)):
                e = float(ui_e[k]) if ui_e is not None else ""
                writer.writerow(["ui", ui_x[k], ui_f[k], e])
    print(f"  Wrote {path}")


def _save_run_summary(out_root: Path, **kwargs) -> None:
    lines = [
        "method: Umbrella Integration (Kästner–Thiel 2006) + block averaging",
        f"dataset_root: {kwargs['dataset_root']}",
        f"reference_npz: {kwargs['reference_npz']}",
        f"window_counts: {kwargs['window_counts']}",
        f"trajectory_fractions: {kwargs['traj_fractions']}",
        f"window_selection_mode: random_subset (replicate 0: evenly_spaced canonical)",
        f"trajectory_selection_mode: contiguous",
        f"random_seed: {kwargs['random_seed']}",
        f"n_equilibration: {kwargs['n_equilibration']}",
        f"num_test_points: {kwargs['num_test_points']}",
        f"n_replicates: {kwargs['n_replicates']}",
        f"n_blocks_block_averaging: {kwargs['n_blocks']}",
        f"pmf_alignment: {kwargs['pmf_alignment']}",
        f"cells_completed: {kwargs['n_cells']}",
        "",
        "uncertainty_decomposition:",
        "  within_variance : mean block-averaging variance across replicates",
        "  between_variance: variance of mean PMFs across replicates",
        "  total_variance  : within + between  (analogous to GP total predictive variance)",
        "",
        "output_files:",
        "  ablation_metrics.csv  — grid summary, compatible with replot_grids.py",
        "  curves/<slug>.csv     — per-cell PMF curve and uncertainties",
        "  references.csv        — reference WHAM and UI curves",
    ]
    path = out_root / "run_summary.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Wrote {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run UI + block-averaging ablation grid on the membrane dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the Katka-format umbrella dataset directory.",
    )
    p.add_argument(
        "--reference-npz",
        required=True,
        help=(
            "Path to a references.npz file containing wham_x, wham_f, "
            "umbrella_x, umbrella_f arrays (as saved by freeGP artifacts)."
        ),
    )
    p.add_argument(
        "--results-dir",
        default="./results/membrane-ablation-ui-block-avg",
        help="Root directory for output files. A 'ui/' subdirectory is created inside.",
    )
    p.add_argument(
        "--window-counts",
        default=",".join(str(w) for w in WINDOW_COUNTS),
        help="Comma-separated list of window counts.",
    )
    p.add_argument(
        "--traj-fractions",
        default=",".join(str(f) for f in TRAJ_FRACTIONS),
        help="Comma-separated list of trajectory fractions.",
    )
    p.add_argument("--random-seed",      type=int,   default=RANDOM_SEED)
    p.add_argument("--n-equilibration",  type=int,   default=N_EQUILIBRATION)
    p.add_argument("--num-test-points",  type=int,   default=NUM_TEST_POINTS)
    p.add_argument("--n-replicates",     type=int,   default=N_REPLICATES)
    p.add_argument("--n-blocks",         type=int,   default=N_BLOCKS,
                   help="Number of blocks for block averaging within each replicate.")
    p.add_argument(
        "--pmf-alignment",
        choices=("max", "min"),
        default="max",
        help="Align shifted PMF at its maximum ('max') or minimum ('min').",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    window_counts   = [int(x)   for x in args.window_counts.split(",")   if x.strip()]
    traj_fractions  = [float(x) for x in args.traj_fractions.split(",")  if x.strip()]

    run_ui_ablation(
        dataset_root  = args.dataset_root,
        reference_npz = args.reference_npz,
        results_dir   = args.results_dir,
        window_counts = window_counts,
        traj_fractions= traj_fractions,
        random_seed   = args.random_seed,
        n_equilibration = args.n_equilibration,
        num_test_points = args.num_test_points,
        n_replicates  = args.n_replicates,
        n_blocks      = args.n_blocks,
        pmf_alignment = args.pmf_alignment,
    )


if __name__ == "__main__":
    main()
