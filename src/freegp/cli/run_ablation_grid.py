#!/usr/bin/env python3
"""Run a publication-oriented ablation-grid study and save summary heatmaps."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
import tomllib

import matplotlib
matplotlib.use("Agg")

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parents[2]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from freegp.studies import (
        CSANYI_FIXED_ELL,
        CSANYI_FIXED_W,
        StudyModelConfig,
        compute_metric_clims,
        compute_param_clims,
        compute_predictive_y_lim,
        run_ablation_study,
        save_ablation_summary,
    )
else:
    from ..studies import (
        CSANYI_FIXED_ELL,
        CSANYI_FIXED_W,
        StudyModelConfig,
        compute_metric_clims,
        compute_param_clims,
        compute_predictive_y_lim,
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
    parser.add_argument("--method", choices=("fixed_gp", "nuts"), default="fixed_gp")
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


def _compute_global_scales(results: list) -> dict:
    """Compute unified axis/color scales across all AblationStudyResult objects."""
    y_lims = [compute_predictive_y_lim(r) for r in results]
    y_lim = (min(lo for lo, _ in y_lims), max(hi for _, hi in y_lims))

    all_mc = [compute_metric_clims(r) for r in results]
    metric_clims: dict = {}
    for name in {k for d in all_mc for k in d}:
        vals = [d[name] for d in all_mc if name in d]
        metric_clims[name] = (min(lo for lo, _ in vals), max(hi for _, hi in vals))

    all_pc = [compute_param_clims(r) for r in results]
    param_clims: dict = {}
    for name in {k for d in all_pc for k in d}:
        vals = [d[name] for d in all_pc if name in d]
        param_clims[name] = (min(lo for lo, _ in vals), max(hi for _, hi in vals))

    return {"predictive_y_lim": y_lim, "metric_clims": metric_clims, "param_clims": param_clims}


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    config_defaults = _load_config_defaults(pre_args.config)

    parser = build_parser(defaults=config_defaults)
    args = parser.parse_args()
    if args.dataset_root is None:
        parser.error("--dataset-root is required unless provided in --config.")

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
        barrier_bins=args.barrier_bins,
        selection_replicates=args.selection_replicates,
    )
    window_counts = _parse_int_list(args.window_counts)
    trajectory_fractions = _parse_float_list(args.trajectory_fractions)

    objectives = [args.objective]
    if args.objective == "both":
        if args.method != "nuts":
            raise ValueError("--objective both is only supported for --method nuts.")
        objectives = ["lml", "loo"]

    compare_objectives = len(objectives) > 1
    root_dir = prepare_figure_dir(
        args.figure_dir,
        project_root=args.project_root,
        compare_objectives=compare_objectives,
    )

    def _run(model_cfg) -> object:
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
        )

    result_plan = []
    for objective in objectives:
        figure_dir = root_dir / objective if compare_objectives else root_dir
        result_plan.append((_run(replace(model, objective=objective)), figure_dir, objective))

    if args.include_fixed:
        fixed_dir = root_dir / "fixed" if compare_objectives or args.include_fixed else root_dir
        result_plan.append((_run(replace(model, method="fixed_gp", kernel="stationary", objective="fixed")), fixed_dir, "fixed"))

    scale_kwargs = _compute_global_scales([r for r, _, _ in result_plan])
    for result, figure_dir, label in result_plan:
        save_ablation_summary(result, figure_dir, **scale_kwargs)
        print(f"figure_dir ({label}): {figure_dir}")
        print(f"cells completed ({label}): {len(result.cells)}")


if __name__ == "__main__":
    main()
