#!/usr/bin/env python3
"""Generate a tiny synthetic umbrella-sampling dataset for the tutorial."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


KBT = 8.3144621e-3 * 303.15
BETA = 1.0 / KBT


def true_free_energy(x: np.ndarray) -> np.ndarray:
    return 8.0 * (x**2 - 1.0) ** 2 + 1.5 * np.sin(4.0 * x)


def sample_biased_window(
    *,
    center: float,
    force_constant: float,
    grid: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    biased = true_free_energy(grid) + 0.5 * force_constant * (grid - center) ** 2
    weights = np.exp(-BETA * (biased - biased.min()))
    probs = weights / weights.sum()
    return rng.choice(grid, size=n_samples, replace=True, p=probs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="example_data")
    parser.add_argument("--n-windows", type=int, default=11)
    parser.add_argument("--n-samples", type=int, default=2500)
    parser.add_argument("--force-constant", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    centers = np.linspace(-1.8, 1.8, args.n_windows)
    grid = np.linspace(-2.4, 2.4, 5000)

    with (out / "README").open("w") as handle:
        handle.write("# window_id center_nm force_constant_kJ_mol_nm2\n")
        for i, center in enumerate(centers):
            window_id = f"w{i:02d}"
            handle.write(f"{window_id} {center:.8f} {args.force_constant:.8f}\n")
            samples = sample_biased_window(
                center=center,
                force_constant=args.force_constant,
                grid=grid,
                n_samples=args.n_samples,
                rng=rng,
            )
            time = np.arange(samples.size, dtype=float)
            data = np.column_stack([time, samples])
            np.savetxt(out / f"{window_id}-pullx.xvg", data, fmt="%.8f")

    x_ref = np.linspace(-1.65, 1.65, 500)
    f_ref = true_free_energy(x_ref)
    f_ref = f_ref - f_ref.min()
    np.savetxt(
        out / "ground_truth.csv",
        np.column_stack([x_ref, f_ref]),
        delimiter=",",
        header="x_nm,F_kJ_per_mol",
    )
    print(f"Wrote tutorial dataset to {out.resolve()}")


if __name__ == "__main__":
    main()
