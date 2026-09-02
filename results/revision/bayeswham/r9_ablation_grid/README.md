# BayesWHAM R9 ablation grid

This directory contains BayesWHAM posterior calculations for the same 10 by 10
window-count/trajectory-fraction grid and saved window selections used in the
R9 GP ablation study. There are 460 calculations: five selections per cell,
except for the ten 25-window cells where every selection is identical.

Aggregate the posterior draws into cell and replicate metrics with:

```bash
source .venv/bin/activate
python results/revision/bayeswham/analyze_r9_ablation_grid.py
```

Each posterior free-energy draw is maximum-aligned before computing the
posterior mean, pointwise standard deviation, central 68 and 95% intervals,
and empirical ensemble CRPS. RMSE and interval inclusion are evaluated against
the full-data WHAM reference. Metrics are first calculated per window-selection
replicate and then averaged within each grid cell.

Generate the manuscript-style RMSE heatmap, standard-deviation heatmap, and
parity panel with:

```bash
MPLBACKEND=Agg python results/revision/bayeswham/plot_r9_ablation_grid.py
```

The initial grid used two NUTS chains with 300 warmup and 500 retained samples.
Although every calculation completed without divergences, the aggregate
diagnostics identify poorly mixed replicates (including a maximum R-hat of
2.66). The current figure should therefore be treated as preliminary until
those replicates are rerun with a larger sampling budget or otherwise resolved.
