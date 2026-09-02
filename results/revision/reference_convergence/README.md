# R9 WHAM reference convergence

This analysis checks whether the full-data WHAM reference used for the R9
membrane ablation is supported by converged trajectories and overlapping
umbrella histograms.

`run_reference_convergence.py` removes the same 40,000 equilibration frames
used in the manuscript analysis and performs two complementary calculations:

1. Each post-equilibration trajectory is divided into five contiguous blocks,
   and a separate WHAM surface is reconstructed from each block.
2. WHAM surfaces are reconstructed from the first 20, 40, 60, 80, and 100%
   of every post-equilibration trajectory.

For each analyzed segment, the repository's Bayesian AR(1) estimator is used
to estimate the autocorrelation time. Samples are deterministically thinned at
a stride of `ceil(tau)` before histogramming. WHAM is then solved using the
uniform-prior BayesWHAM MAP fixed-point equations, which are equivalent to the
standard WHAM point estimate used in the BayesWHAM validation.

The full-data thinned histograms are plotted as ridgelines on an expanded grid
that contains the complete end-window distributions, exposing gaps or weak
overlap between adjacent umbrella windows. The WHAM calculations themselves
remain on the published reference grid for exact comparison. Numerical adjacent-window
overlap coefficients, reconstruction RMSEs, thinning metadata, and WHAM
iteration diagnostics are written to CSV/JSON files under `outputs/`.

Run from the repository root:

```bash
source .venv/bin/activate
MPLBACKEND=Agg python results/revision/reference_convergence/run_reference_convergence.py
```
