#!/usr/bin/env python3
"""Rerun high-R-hat BayesWHAM replicates with the validated fast protocol."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import time


def main() -> None:
    root = Path(__file__).resolve().parent
    grid = root / "r9_ablation_grid"
    analysis = grid / "analysis"
    with (analysis / "replicate_metrics.csv").open(newline="") as handle:
        metrics = list(csv.DictReader(handle))
    targets = [
        row for row in metrics
        if float(row["max_r_hat"]) > 1.05 and row.get("result_source", "nuts") == "nuts"
    ]
    targets.sort(
        key=lambda row: (
            int(row["window_count"]),
            float(row["trajectory_fraction"]),
            int(row["replicate"]),
        )
    )
    status_path = grid / "refined_run_status.json"
    statuses = json.loads(status_path.read_text()) if status_path.exists() else {}
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/freegp-matplotlib"})
    runner = root / "run_r9_bayeswham_nuts.py"

    for ordinal, row in enumerate(targets, start=1):
        window_count = int(row["window_count"])
        fraction = float(row["trajectory_fraction"])
        replicate = int(row["replicate"])
        candidates = []
        for cell_dir in grid.glob(f"w{window_count:02d}_f*"):
            manifest = cell_dir / f"replicate_{replicate}/input/manifest.json"
            if manifest.is_file():
                archived = json.loads(manifest.read_text())["archived_cell"]
                archive_fraction = {
                    "f0p04": 0.04, "f0p06": 0.063, "f0p10": 0.1,
                    "f0p16": 0.16, "f0p25": 0.25, "f0p40": 0.4,
                    "f0p60": 0.6, "f0p80": 0.8, "f0p90": 0.9,
                    "f1p00": 1.0,
                }[Path(archived).stem.split("_", 1)[1]]
                if abs(archive_fraction - fraction) < 1e-9:
                    candidates.append(cell_dir)
        if len(candidates) != 1:
            raise RuntimeError(f"Could not uniquely resolve target {row}")
        replicate_dir = candidates[0] / f"replicate_{replicate}"
        output = replicate_dir / "uq/nuts_refined_fast"
        key = str(replicate_dir.relative_to(grid))
        diagnostic = output / "diagnostics.json"
        if diagnostic.is_file():
            statuses[key] = {"status": "complete", "skipped_existing": True}
            continue
        output.mkdir(parents=True, exist_ok=True)
        command = [
            os.sys.executable,
            str(runner),
            "--input", str(replicate_dir / "input"),
            "--output", str(output),
            "--cell-map", "--init-jitter", "0.05",
            "--warmup", "300", "--samples", "500", "--chains", "4",
            "--target-accept", "0.8", "--seed", str(20263000 + ordinal),
        ]
        start = time.perf_counter()
        with (output / "run.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        status = "complete" if result.returncode == 0 else "failed"
        statuses[key] = {
            "status": status,
            "return_code": result.returncode,
            "wall_time_seconds": elapsed,
            "ordinal": ordinal,
            "total": len(targets),
        }
        status_path.write_text(json.dumps(statuses, indent=2) + "\n")
        print(f"[{ordinal}/{len(targets)}] {key}: {status} ({elapsed:.1f} s)", flush=True)

    failed = [key for key, value in statuses.items() if value["status"] == "failed"]
    if failed:
        raise RuntimeError(f"Refined runs failed: {failed}")
    for script in (root / "analyze_r9_ablation_grid.py", root / "plot_r9_ablation_grid.py"):
        subprocess.run([os.sys.executable, str(script)], check=True, env=env)
    print("All refined runs and final BayesWHAM figure completed.")


if __name__ == "__main__":
    main()
