#!/usr/bin/env python3
"""Benchmark the paper's four reconstruction methods on representative R9 cells.

Each timed run is executed in a fresh Python subprocess and includes loading the
trajectory data, preprocessing, inference, prediction, and writing the normal
scientific outputs. Molecular-dynamics generation is not included.

The default ``paper`` profile reproduces the inference settings used in the
manuscript. Nothing is executed unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    windows: int
    trajectory_fraction: float


# Match the super_hard, hard, and easy cases in the archived
# results/hmc-calibration-sweep analysis, respectively.
CONDITIONS = {
    "sparse": Condition("sparse", "Sparse (archived super_hard)", 3, 0.10),
    "intermediate": Condition("intermediate", "Intermediate (archived hard)", 7, 0.25),
    "full": Condition("full", "Full (archived easy)", 25, 1.00),
}
METHODS = ("ui", "fixed", "map", "hierarchical")


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run commands; otherwise print a dry run.")
    parser.add_argument(
        "--profile",
        choices=("paper", "smoke"),
        default="paper",
        help="Use exact manuscript settings or a reduced validation profile.",
    )
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260830)
    parser.add_argument("--dataset-root", type=Path, default=Path.home() / "freeGP-datasets/membranes/katka")
    parser.add_argument("--paper-repo", type=Path, default=Path.home() / "freeGP-v0.1.0")
    parser.add_argument("--active-repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-root", type=Path, default=here / "runs")
    return parser


def inference_settings(profile: str) -> dict[str, int]:
    if profile == "paper":
        return {
            "warmup_steps": 500,
            "num_samples": 1000,
            "num_chains": 4,
            "predictive_samples": 100,
            "opt_steps": 250,
            "opt_restarts": 3,
        }
    return {
        "warmup_steps": 2,
        # Pyro's split-R-hat diagnostic requires at least four retained draws.
        "num_samples": 4,
        "num_chains": 1,
        "predictive_samples": 2,
        "opt_steps": 2,
        "opt_restarts": 1,
    }


def gp_command(
    args: argparse.Namespace,
    condition: Condition,
    method: str,
    output_dir: Path,
) -> list[str]:
    settings = inference_settings(args.profile)
    method_name = {"fixed": "fixed_gp", "map": "optimized_gp", "hierarchical": "nuts"}[method]
    return [
        str(args.python.absolute()),
        "-m",
        "freegp.cli.run_ablation_grid",
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--project-root",
        str(args.paper_repo.resolve()),
        "--results-dir",
        str(output_dir),
        "--window-counts",
        str(condition.windows),
        "--trajectory-fractions",
        str(condition.trajectory_fraction),
        "--method",
        method_name,
        "--kernel",
        "stationary",
        "--objective",
        "loo",
        "--num-bins",
        "20",
        "--num-test-points",
        "100",
        "--n-equilibration",
        "40000",
        "--window-selection-mode",
        "evenly_spaced",
        "--trajectory-selection-mode",
        "contiguous",
        "--random-seed",
        "42",
        "--selection-replicates",
        "1",
        "--warmup-steps",
        str(settings["warmup_steps"]),
        "--num-samples",
        str(settings["num_samples"]),
        "--num-chains",
        str(settings["num_chains"]),
        "--predictive-samples",
        str(settings["predictive_samples"]),
        "--opt-steps",
        str(settings["opt_steps"]),
        "--opt-restarts",
        str(settings["opt_restarts"]),
        "--opt-learning-rate",
        "0.05",
        "--max-tree-depth",
        "10",
        "--device",
        "cpu",
        "--no-checkpoint-cells",
        "--no-resume",
    ]


def ui_command(
    args: argparse.Namespace,
    condition: Condition,
    output_dir: Path,
) -> list[str]:
    reference = (
        args.paper_repo
        / "results/membrane-ablation-10x10-5replicates/loo/artifacts/references.npz"
    )
    return [
        str(args.python.absolute()),
        str((args.active_repo / "scripts/run_ui_ablation_grid.py").resolve()),
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--reference-npz",
        str(reference.resolve()),
        "--results-dir",
        str(output_dir),
        "--window-counts",
        str(condition.windows),
        "--traj-fractions",
        str(condition.trajectory_fraction),
        "--random-seed",
        "42",
        "--n-equilibration",
        "40000",
        "--num-test-points",
        "100",
        "--n-replicates",
        "1",
    ]


def command_for(
    args: argparse.Namespace,
    condition: Condition,
    method: str,
    output_dir: Path,
) -> list[str]:
    if method == "ui":
        return ui_command(args, condition, output_dir)
    return gp_command(args, condition, method, output_dir)


def benchmark_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    thread_value = str(args.threads)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[key] = thread_value
    env["PYTHONHASHSEED"] = "0"
    env["MPLCONFIGDIR"] = str((Path(__file__).resolve().parent / ".matplotlib").resolve())
    return env


def software_versions(python: Path) -> dict[str, str]:
    code = (
        "import json,platform; import numpy,torch,pyro; "
        "print(json.dumps({'python':platform.python_version(),'numpy':numpy.__version__,"
        "'torch':torch.__version__,'pyro':pyro.__version__}))"
    )
    result = subprocess.run(
        [str(python.absolute()), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "benchmark_profile": args.profile,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware_description": "Apple MacBook Pro; Apple M3 Pro; 11 CPU cores (5 performance, 6 efficiency); 18 GB unified memory",
        "threads_per_process": args.threads,
        "warmup_runs": args.warmup_runs,
        "measured_repeats": args.repeats,
        "software": software_versions(args.python),
        "inference_settings": inference_settings(args.profile),
        "timed_scope": "Python interpreter startup, trajectory loading, preprocessing, inference, prediction, and normal result serialization; excludes MD generation",
    }


def write_raw(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]], path: Path) -> list[dict[str, object]]:
    measured = [row for row in rows if not row["warmup"] and row["return_code"] == 0]
    summaries: list[dict[str, object]] = []
    for condition_key in dict.fromkeys(row["condition"] for row in measured):
        condition_rows = [row for row in measured if row["condition"] == condition_key]
        medians: dict[str, float] = {}
        for method in dict.fromkeys(row["method"] for row in condition_rows):
            values = [float(row["wall_seconds"]) for row in condition_rows if row["method"] == method]
            medians[method] = statistics.median(values)
        fixed_time = medians.get("fixed", float("nan"))
        for method, median in medians.items():
            values = [float(row["wall_seconds"]) for row in condition_rows if row["method"] == method]
            summaries.append(
                {
                    "condition": condition_key,
                    "method": method,
                    "n": len(values),
                    "median_seconds": median,
                    "min_seconds": min(values),
                    "max_seconds": max(values),
                    "relative_to_fixed": median / fixed_time,
                }
            )
    if summaries:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return summaries


def validate_inputs(args: argparse.Namespace) -> None:
    required = [
        args.dataset_root,
        args.paper_repo,
        args.active_repo / "scripts/run_ui_ablation_grid.py",
        args.paper_repo / "results/membrane-ablation-10x10-5replicates/loo/artifacts/references.npz",
        args.python,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark inputs:\n" + "\n".join(missing))


def main() -> None:
    args = build_parser().parse_args()
    validate_inputs(args)
    session = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{args.profile}"
    session_dir = args.output_root.expanduser().resolve() / session
    env = benchmark_environment(args)
    rng = random.Random(args.random_seed)

    planned: list[tuple[Condition, str, bool, int]] = []
    for condition_key in args.conditions:
        condition = CONDITIONS[condition_key]
        for warmup_index in range(args.warmup_runs):
            methods = list(args.methods)
            rng.shuffle(methods)
            planned.extend((condition, method, True, warmup_index + 1) for method in methods)
        for repeat in range(1, args.repeats + 1):
            methods = list(args.methods)
            rng.shuffle(methods)
            planned.extend((condition, method, False, repeat) for method in methods)

    if not args.execute:
        print(f"Dry run: {len(planned)} subprocesses would be executed.")
        for condition, method, warmup, repeat in planned:
            output = session_dir / "outputs" / condition.key / method / (
                f"warmup_{repeat}" if warmup else f"repeat_{repeat}"
            )
            print(" ".join(command_for(args, condition, method, output)))
        print("Re-run with --execute to start the benchmark.")
        return

    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "metadata.json").write_text(json.dumps(metadata(args), indent=2) + "\n")
    raw_rows: list[dict[str, object]] = []
    raw_path = session_dir / "raw_timings.csv"
    commands_path = session_dir / "commands.txt"

    with commands_path.open("w") as command_log:
        for run_number, (condition, method, warmup, repeat) in enumerate(planned, start=1):
            run_name = f"warmup_{repeat}" if warmup else f"repeat_{repeat}"
            output = session_dir / "outputs" / condition.key / method / run_name
            output.mkdir(parents=True, exist_ok=False)
            command = command_for(args, condition, method, output)
            command_log.write(" ".join(command) + "\n")
            command_log.flush()
            print(
                f"[{run_number}/{len(planned)}] {condition.label} / {method} / {run_name}",
                flush=True,
            )
            log_path = output / "benchmark_subprocess.log"
            start = time.perf_counter()
            with log_path.open("w") as log:
                completed = subprocess.run(
                    command,
                    cwd=args.active_repo.resolve(),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            elapsed = time.perf_counter() - start
            raw_rows.append(
                {
                    "condition": condition.key,
                    "window_count": condition.windows,
                    "trajectory_fraction": condition.trajectory_fraction,
                    "method": method,
                    "warmup": warmup,
                    "repeat": repeat,
                    "wall_seconds": elapsed,
                    "return_code": completed.returncode,
                    "output_dir": str(output),
                }
            )
            write_raw(raw_rows, raw_path)
            if completed.returncode != 0:
                raise RuntimeError(f"Benchmark failed; inspect {log_path}")

    summaries = summarize(raw_rows, session_dir / "summary.csv")
    print(f"Completed benchmark: {session_dir}")
    for row in summaries:
        print(
            f"{row['condition']:>12} {row['method']:>12}: "
            f"{row['median_seconds']:.3f} s ({row['relative_to_fixed']:.2f}x fixed)"
        )


if __name__ == "__main__":
    main()
