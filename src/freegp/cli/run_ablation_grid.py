#!/usr/bin/env python3
"""Run a publication-oriented ablation-grid study and save summary heatmaps."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from freegp.studies import StudyModelConfig, run_ablation_study, save_ablation_summary
else:
    from ..studies import StudyModelConfig, run_ablation_study, save_ablation_summary


def _parse_float_list(value: str) -> list[float]:
    return [float(piece) for piece in value.split(",") if piece.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(piece) for piece in value.split(",") if piece.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the freeGP ablation-grid study scaffold.")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--project-root", default=None, type=str)
    parser.add_argument("--window-counts", default="25,13", type=str)
    parser.add_argument("--trajectory-fractions", default="1.0,0.5", type=str)
    parser.add_argument("--method", choices=("fixed_gp", "nuts"), default="fixed_gp")
    parser.add_argument("--kernel", choices=("stationary", "gibbs"), default="stationary")
    parser.add_argument("--length-model", choices=("exp_linear_bump", "constant"), default="exp_linear_bump")
    parser.add_argument("--width-model", choices=("tanh_decay", "constant"), default="tanh_decay")
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--num-test-points", type=int, default=100)
    parser.add_argument("--n-equilibration", type=int, default=40_000)
    parser.add_argument("--test-grid-source", choices=("umbrella_centers", "histogram_support"), default="umbrella_centers")
    parser.add_argument("--test-grid-mode", choices=("full_dataset", "per_cell"), default="full_dataset")
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--ell", type=float, default=4.0)
    parser.add_argument("--w", type=float, default=3.3)
    parser.add_argument("--a0", type=float, default=1.38629436112)
    parser.add_argument("--a1", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--length-w", type=float, default=0.5)
    parser.add_argument("--s", type=float, default=1.65)
    parser.add_argument("--u", type=float, default=None)
    parser.add_argument("--w2", type=float, default=0.5)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--objective", choices=("lml", "loo"), default="lml")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument("--predictive-samples", type=int, default=10)
    parser.add_argument("--figure-dir", type=str, default=None)
    return parser


def prepare_figure_dir(path: str | None, *, project_root: str | None = None) -> Path:
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if project_root is None:
            repo_root = Path(__file__).resolve().parents[3]
        else:
            repo_root = Path(project_root).expanduser().resolve()
        root = repo_root / "figures" / f"ablation-{stamp}"
    else:
        root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model = StudyModelConfig(
        method=args.method,
        kernel=args.kernel,
        length_model=args.length_model,
        width_model=args.width_model,
        ell=args.ell,
        w=args.w,
        a0=args.a0,
        a1=args.a1,
        b=args.b,
        c=args.c,
        length_w=args.length_w,
        s=args.s,
        u=args.u,
        w2=args.w2,
        jitter=args.jitter,
        objective=args.objective,
        warmup_steps=args.warmup_steps,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        target_accept_prob=args.target_accept_prob,
        predictive_samples=args.predictive_samples,
    )

    result = run_ablation_study(
        dataset_root=args.dataset_root,
        project_root=args.project_root,
        window_counts=_parse_int_list(args.window_counts),
        trajectory_fractions=_parse_float_list(args.trajectory_fractions),
        model=model,
        n_equilibration=args.n_equilibration,
        num_bins=args.num_bins,
        num_test_points=args.num_test_points,
        test_grid_source=args.test_grid_source,
        x_min=args.x_min,
        x_max=args.x_max,
        test_grid_mode=args.test_grid_mode,
    )

    figure_dir = prepare_figure_dir(args.figure_dir, project_root=args.project_root)
    save_ablation_summary(result, figure_dir)
    print(f"figure_dir: {figure_dir}")
    print(f"cells completed: {len(result.cells)}")


if __name__ == "__main__":
    main()
