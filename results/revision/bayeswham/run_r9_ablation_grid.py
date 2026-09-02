#!/usr/bin/env python3
"""Run the prepared BayesWHAM grid sequentially with restartable cells."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--chains", type=int, default=2)
    return p


def main() -> None:
    args = parser().parse_args()
    root = Path(__file__).resolve().parent
    grid = root / "r9_ablation_grid"
    runner = root / "run_r9_bayeswham_nuts.py"
    map_dir = root / "r9_full/map_validation/effective_counts"
    cells = sorted(grid.glob("w*_f*/replicate_*/input"))
    status_path = grid / "run_status.json"
    statuses = json.loads(status_path.read_text()) if status_path.exists() else {}
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/freegp-matplotlib"})

    for ordinal, input_dir in enumerate(cells, start=1):
        run_dir = input_dir.parent / "uq" / "nuts"
        diagnostic = run_dir / "diagnostics.json"
        key = str(input_dir.parent.relative_to(grid))
        if diagnostic.exists():
            statuses[key] = {"status": "complete", "skipped_existing": True}
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(Path(os.sys.executable)), str(runner),
            "--input", str(input_dir), "--map-output", str(map_dir),
            "--output", str(run_dir), "--warmup", str(args.warmup),
            "--samples", str(args.samples), "--chains", str(args.chains),
            "--seed", str(20261000 + ordinal),
        ]
        start = time.perf_counter()
        with (run_dir / "run.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start
        statuses[key] = {
            "status": "complete" if result.returncode == 0 else "failed",
            "return_code": result.returncode,
            "wall_time_seconds": elapsed,
            "ordinal": ordinal,
            "total": len(cells),
        }
        status_path.write_text(json.dumps(statuses, indent=2) + "\n")
        print(f"[{ordinal}/{len(cells)}] {key}: {statuses[key]['status']} ({elapsed:.1f} s)", flush=True)
    print(f"Grid driver finished; status is in {status_path}")


if __name__ == "__main__":
    main()
