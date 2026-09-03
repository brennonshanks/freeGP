from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Potential energy function U(x, y) -- edit this to whatever you need.
# The sampled (canonical / Boltzmann) distribution is p(x, y) ~ exp(-U(x, y) / kT)
# ---------------------------------------------------------------------------
def potential_with_umbrella(x, y, x0 = 0.0, y0 = 0.0, kappa = 100.0):
    '''
    Toy potential: a double-well in u=x+y and a harmonic well in v=x-y. You can replace this with your own potential function.
    '''
    #"""Double-well potential as an example. Replace with your own U(x, y)."""
    #return (x**2 - 1) ** 2 + y**2
    a = 4.0
    b = 3.0
    c = 50.0
    
    return a*(1/2*(x+y)**2 - b**2)**2 + c/2*(x-y)**2 + 1/2*kappa*((x-x0)**2 + (y-y0)**2)

# ---------------------------------------------------------------------------
# Metropolis-Hastings sampler
# ---------------------------------------------------------------------------
def metropolis_hastings_umbrella(
    potential_fn,
    n_samples=20000,
    n_burn=2000,
    step_size=0.3,
    kT=1.0,
    start=(0.0, 0.0),
    seed=None,
    x0=0.0,
    y0=0.0,
    kappa=100.0
):
    """
    Sample (x, y) from p(x, y) ~ exp(-U(x, y) / kT) using a random-walk
    Metropolis-Hastings algorithm.

    Returns an array of shape (n_samples, 2).
    """
    rng = np.random.default_rng(seed)

    current = np.array(start, dtype=float)
    current_U = potential_fn(*current, x0=x0, y0=y0, kappa=kappa)

    samples = np.empty((n_samples, 2))
    n_accept = 0
    total_steps = n_burn + n_samples

    for i in range(total_steps):
        proposal = current + rng.normal(scale=step_size, size=2)
        proposal_U = potential_fn(*proposal, x0=x0, y0=y0, kappa=kappa)

        dU = proposal_U - current_U
        if dU <= 0 or rng.random() < np.exp(-dU / kT):
            current = proposal
            current_U = proposal_U
            n_accept += 1

        if i >= n_burn:
            samples[i - n_burn] = current

    acceptance_rate = n_accept / total_steps
    #print(f"Acceptance rate: {acceptance_rate:.2%}")

    return samples


# ---------------------------------------------------------------------------
# Plotting: 2D heatmap of the potential, optionally with samples overlaid
# ---------------------------------------------------------------------------
def plot_potential_heatmap(
    potential_fn,
    samples=None,
    xlim=(-4.0, 4.0),
    ylim=(-4.0, 4.0),
    grid_res=200,
    out_path="potential_heatmap.png",
    vmax=None,
):
    """Evaluate U(x, y) on a grid and plot it as a heatmap, optionally
    overlaying the MCMC samples as scatter points on top."""
    x = np.linspace(*xlim, grid_res)
    y = np.linspace(*ylim, grid_res)
    X, Y = np.meshgrid(x, y)
    U = potential_fn(X, Y, kappa=0.0)
    if vmax is None:
        vmax = np.max(U)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(X, Y, U, shading="auto", cmap="viridis", vmax=vmax)
    fig.colorbar(im, ax=ax, label="U(x, y)")

    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], s=2, c="white", alpha=0.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Free energy heatmap")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="example_data")
    parser.add_argument("--n-windows", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--n-burn", type=int, default=2000)
    parser.add_argument("--step-size", type=float, default=0.3)
    parser.add_argument("--kt", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-res", type=int, default=500)
    parser.add_argument("--grid-xlim", type=float, nargs=2, default=(-4.0, 4.0))
    parser.add_argument("--grid-ylim", type=float, nargs=2, default=(-4.0, 4.0))
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x0s = np.linspace(-3.75, 3.75, args.n_windows)
    y0s = np.linspace(-3.75, 3.75, args.n_windows)
    window_centers = list(itertools.product(x0s, y0s))

    with (out / "README").open("w") as handle:
        handle.write("# window_id x0 y0 force_constant\n")
        for i, (x0, y0) in enumerate(window_centers):
            window_id = f"w{i:02d}"
            handle.write(f"{window_id} {x0:.8f} {y0:.8f} {args.kappa:.8f}\n")

            samples = metropolis_hastings_umbrella(
                potential_with_umbrella,
                n_samples=args.n_samples,
                n_burn=args.n_burn,
                step_size=args.step_size,
                kT=args.kt,
                start=(x0, y0),
                seed=args.seed,
                x0=x0,
                y0=y0,
                kappa=args.kappa,
            )

            time = np.arange(samples.shape[0], dtype=float)
            data = np.column_stack([time, samples])
            np.savetxt(out / f"{window_id}-xyplane.xvg", data, fmt="%.8f")

    x_grid = np.linspace(*args.grid_xlim, args.grid_res)
    y_grid = np.linspace(*args.grid_ylim, args.grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    U = potential_with_umbrella(X, Y, kappa=0.0)
    np.savetxt(
        out / "ground_truth.csv",
        np.column_stack([X.ravel(), Y.ravel(), U.ravel()]),
        delimiter=",",
        header="x,y,U",
    )

    print(f"Wrote tutorial dataset to {out.resolve()}")
    plot_potential_heatmap(potential_with_umbrella, samples=np.array(window_centers), out_path=out / "free_energy.png", vmax=350.0)


if __name__ == "__main__":
    main()
