#!/usr/bin/env python3
"""Run a publication-oriented ablation-grid study and save data artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
import tomllib

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from freegp.studies import (
        CSANYI_FIXED_ELL,
        CSANYI_FIXED_W,
        StudyModelConfig,
        run_ablation_study,
        save_ablation_summary,
    )
else:
    from ..studies import (
        CSANYI_FIXED_ELL,
        CSANYI_FIXED_W,
        StudyModelConfig,
        run_ablation_study,
        save_ablation_summary,
    )


def _parse_float_list(value: str) -> list[float]:
    return [float(piece) for piece in value.split(",") if piece.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(piece) for piece in value.split(",") if piece.strip()]


def _normalize_config_value(key: str, value):
    if key in {"window_counts", "trajectory_fractions"}:
        if isinstance(value, list):
            return ",".join(str(piece) for piece in value)
    return value


def _load_config_defaults(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        loaded = tomllib.load(handle)

    if "ablation" in loaded and isinstance(loaded["ablation"], dict):
        loaded = loaded["ablation"]

    key_aliases = {
        "results_dir": "figure_dir",
        "figure_dir": "figure_dir",
    }
    defaults = {
        key_aliases.get(key.replace("-", "_"), key.replace("-", "_")):
        _normalize_config_value(key_aliases.get(key.replace("-", "_"), key.replace("-", "_")), value)
        for key, value in loaded.items()
    }
    defaults["config"] = str(config_path)
    return defaults


def build_parser(*, defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the freeGP ablation-grid study scaffold.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a TOML config file. CLI flags override config values.",
    )
    parser.add_argument("--dataset-root", default=None, type=str)
    parser.add_argument("--project-root", default=None, type=str)
    parser.add_argument("--reference-wham-path", default=None, type=str)
    parser.add_argument("--reference-wham-x-units", default="nm", type=str)
    parser.add_argument("--reference-ui-path", default=None, type=str)
    parser.add_argument("--reference-ui-x-units", default="nm", type=str)
    parser.add_argument(
        "--pmf-alignment",
        choices=("max", "min"),
        default="max",
        help="Align shifted PMF curves at their maximum ('max', default) or minimum ('min').",
    )
    parser.add_argument("--window-counts", default="25,13", type=str)
    parser.add_argument("--trajectory-fractions", default="1.0,0.5", type=str)
    parser.add_argument("--method", choices=("fixed_gp", "optimized_gp", "nuts"), default="fixed_gp")
    parser.add_argument("--kernel", choices=("stationary", "gibbs"), default="stationary")
    parser.add_argument("--length-model", choices=("exp_linear_bump", "constant"), default="exp_linear_bump")
    parser.add_argument("--width-model", choices=("tanh_decay", "constant"), default="tanh_decay")
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--num-test-points", type=int, default=100)
    parser.add_argument("--n-equilibration", type=int, default=40_000)
    parser.add_argument("--test-grid-source", choices=("umbrella_centers", "histogram_support"), default="umbrella_centers")
    parser.add_argument("--test-grid-mode", choices=("full_dataset", "per_cell"), default="full_dataset")
    parser.add_argument(
        "--window-selection-mode",
        choices=("evenly_spaced", "random_subset"),
        default="evenly_spaced",
    )
    parser.add_argument(
        "--trajectory-selection-mode",
        choices=("contiguous", "random_subsample"),
        default="contiguous",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--ell", type=float, default=CSANYI_FIXED_ELL)
    parser.add_argument("--w", type=float, default=CSANYI_FIXED_W)
    parser.add_argument("--a0", type=float, default=1.38629436112)
    parser.add_argument("--a1", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--length-w", type=float, default=0.5)
    parser.add_argument("--s", type=float, default=1.65)
    parser.add_argument("--u", type=float, default=None)
    parser.add_argument("--w2", type=float, default=0.5)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--objective", choices=("lml", "loo", "both"), default="lml")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument(
        "--max-tree-depth",
        type=int,
        default=10,
        help="Maximum NUTS tree depth. Lower values cap pathological long iterations but may increase sampler bias if too small.",
    )
    parser.add_argument("--opt-steps", type=int, default=250)
    parser.add_argument("--opt-restarts", type=int, default=3)
    parser.add_argument("--opt-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used for GP and NUTS tensor computations.",
    )
    parser.add_argument("--target-accept-prob", type=float, default=0.8)
    parser.add_argument("--predictive-samples", type=int, default=10)
    parser.add_argument("--barrier-bins", type=int, default=30)
    parser.add_argument(
        "--selection-replicates",
        type=int,
        default=None,
        help="Number of random-selection replicates per ablation cell. Defaults to 5 when a random selection mode is active, otherwise 1.",
    )
    parser.add_argument(
        "--include-fixed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When using --method nuts, also write a fixed-hyperparameter stationary baseline to fixed/.",
    )
    parser.add_argument(
        "--include-optimized",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When using --method nuts, also write optimized plug-in GP baselines. "
            "With --objective both, writes lml_map/ and loo_map/."
        ),
    )
    parser.add_argument(
        "--checkpoint-cells",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write each completed ablation cell to FIGURE_DIR/_checkpoints and reuse it when --resume is set.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse per-cell checkpoints from FIGURE_DIR/_checkpoints when available.",
    )
    parser.add_argument(
        "--results-dir",
        "--figure-dir",
        dest="figure_dir",
        type=str,
        default=None,
        help="Directory for saved result artifacts and figures. Defaults to ./results/...",
    )
    if defaults:
        known_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - known_dests)
        if unknown:
            raise ValueError(f"Unsupported config keys: {unknown}")
        parser.set_defaults(**defaults)
    return parser


def prepare_figure_dir(
    path: str | None,
    *,
    project_root: str | None = None,
    compare_objectives: bool = False,
) -> Path:
    if path is None:
        if project_root is None:
            repo_root = Path(__file__).resolve().parents[3]
        else:
            repo_root = Path(project_root).expanduser().resolve()
        if compare_objectives:
            root = repo_root / "results" / "ablation-grid"
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            root = repo_root / "results" / f"ablation-{stamp}"
    else:
        root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = _load_config_defaults(pre_args.config)

    parser = build_parser(defaults=config_defaults)
    args = parser.parse_args()
    if args.dataset_root is None:
        parser.error("--dataset-root is required unless provided in --config.")
    if args.include_optimized and args.method != "nuts":
        parser.error("--include-optimized is intended for --method nuts. Use --method optimized_gp to run only MAP baselines.")

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
        max_tree_depth=args.max_tree_depth,
        predictive_samples=args.predictive_samples,
        barrier_bins=args.barrier_bins,
        selection_replicates=args.selection_replicates,
        opt_steps=args.opt_steps,
        opt_restarts=args.opt_restarts,
        opt_learning_rate=args.opt_learning_rate,
    )
    window_counts = _parse_int_list(args.window_counts)
    trajectory_fractions = _parse_float_list(args.trajectory_fractions)

    objectives = [args.objective]
    if args.objective == "both":
        if args.method not in {"nuts", "optimized_gp"}:
            raise ValueError("--objective both is only supported for --method nuts or optimized_gp.")
        objectives = ["lml", "loo"]

    compare_objectives = len(objectives) > 1
    root_dir = prepare_figure_dir(
        args.figure_dir,
        project_root=args.project_root,
        compare_objectives=compare_objectives,
    )

    def _run(model_cfg, figure_dir: Path) -> object:
        return run_ablation_study(
            dataset_root=args.dataset_root,
            project_root=args.project_root,
            reference_wham_path=args.reference_wham_path,
            reference_wham_x_units=args.reference_wham_x_units,
            reference_ui_path=args.reference_ui_path,
            reference_ui_x_units=args.reference_ui_x_units,
            pmf_alignment=args.pmf_alignment,
            window_counts=window_counts,
            trajectory_fractions=trajectory_fractions,
            model=model_cfg,
            n_equilibration=args.n_equilibration,
            num_bins=args.num_bins,
            num_test_points=args.num_test_points,
            test_grid_source=args.test_grid_source,
            x_min=args.x_min,
            x_max=args.x_max,
            test_grid_mode=args.test_grid_mode,
            window_selection_mode=args.window_selection_mode,
            trajectory_selection_mode=args.trajectory_selection_mode,
            random_seed=args.random_seed,
            device=args.device,
            checkpoint_dir=figure_dir / "_checkpoints" if args.checkpoint_cells else None,
            resume=args.resume,
        )

    run_specs = []
    for objective in objectives:
        if compare_objectives:
            dirname = f"{objective}_map" if model.method == "optimized_gp" else objective
            figure_dir = root_dir / dirname
        else:
            figure_dir = root_dir
        run_specs.append((replace(model, objective=objective), figure_dir, objective))

    if args.include_optimized:
        for objective in objectives:
            opt_dir = root_dir / f"{objective}_map" if compare_objectives or args.include_optimized else root_dir
            opt_model = replace(
                model,
                method="optimized_gp",
                kernel="stationary",
                objective=objective,
            )
            run_specs.append((opt_model, opt_dir, f"{objective}_map"))

    if args.include_fixed:
        fixed_dir = root_dir / "fixed" if compare_objectives or args.include_fixed else root_dir
        run_specs.append((replace(model, method="fixed_gp", kernel="stationary", objective="fixed"), fixed_dir, "fixed"))

    for model_cfg, figure_dir, label in run_specs:
        result = _run(model_cfg, figure_dir)
        save_ablation_summary(result, figure_dir)
        print(f"figure_dir ({label}): {figure_dir}")
        print(f"cells completed ({label}): {len(result.cells)}")


if __name__ == "__main__":
    main()
