# freeGP

Gaussian-process free-energy reconstruction tools for umbrella-sampling analysis, with support for:

- stationary and Gibbs-style nonstationary kernels
- deterministic GP reconstruction from histogram and derivative information
- HMC-NUTS hyperparameter inference with `lml` and `loo` objectives
- ablation-grid studies for window/trajectory retention experiments
- saved result artifacts for later plotting, resampling, and analysis

This repository has been cleaned down to the streamlined implementation in [`src/freegp`](/home/bshanks/freeGP-dev/src/freegp). A full historical backup of the older notebook-heavy code lives separately in:

- [`/home/bshanks/freeGP-dev-legacy`](/home/bshanks/freeGP-dev-legacy)

## What This Package Does

The current package is built around a joint GP model for free-energy reconstruction from umbrella sampling data. The code:

- loads umbrella trajectories and reference curves
- preprocesses histograms, derivative observations, and noise estimates
- builds stationary or Gibbs-kernel GPs
- runs Pyro HMC-NUTS over GP hyperparameters
- propagates hyperposterior uncertainty into the predictive posterior
- runs data-ablation studies and writes publication-oriented plots and reusable artifacts

The main scientific use case is to compare how predictive uncertainty behaves under:

- different kernel families
- different hyperparameter objectives (`lml` vs `loo`)
- fixed-hyperparameter baselines
- progressive removal of windows and/or trajectory data

## Repository Layout

The cleaned repo is organized as:

```text
freeGP-dev/
  configs/
    ablation/                  Example TOML configs for ablation studies
  reference_data/
    UI-Semen/pmf.dat           UI reference curve
    wham.dat                   WHAM reference curve
    umbrella.dat               Reference umbrella metadata
  results/                     Saved outputs from runs
  src/freegp/
    data.py                    Dataset and reference-data loading
    preprocess.py              Histogram/trajectory preprocessing
    workflow.py                Shared pipeline assembly
    gp.py                      Stationary + Gibbs GP implementation
    hmc.py                     Pyro HMC-NUTS interface
    posterior.py               Hyperposterior predictive summaries
    metrics.py                 RMSE and uncertainty summaries
    studies/ablation.py        Ablation-grid experiment logic
    cli/run_gprhd_hmc.py       Single-run CLI
    cli/run_ablation_grid.py   Ablation-study CLI
```

There are also compatibility wrappers at:

- [`run_gprhd_hmc.py`](/home/bshanks/freeGP-dev/src/freegp/run_gprhd_hmc.py)
- [`run_ablation_grid.py`](/home/bshanks/freeGP-dev/src/freegp/run_ablation_grid.py)

so you can still run the files directly with `python`.

## Installation

Create or activate your environment, then install the package in editable mode:

```bash
cd /home/bshanks/freeGP-dev
python -m pip install -e ".[all]"
```

The optional dependencies in `.[all]` include:

- `pyro-ppl` for HMC-NUTS
- `corner` for corner plots
- notebook tools for exploratory work

## Data Requirements

The umbrella-sampling dataset is expected to be available externally, for example under:

```text
~/freeGP-datasets/membranes/katka/
  d_1.45/
  d_1.60/
  ...
```

Each umbrella window directory should contain:

- `step7_production_pullx.xvg`
- `step7_production.mdp`

You can point the code at the dataset in either of these ways:

```bash
export FREEGP_DATASETS=~/freeGP-datasets/membranes/katka
```

or:

```bash
--dataset-root ~/freeGP-datasets/membranes/katka
```

Reference curves used for comparison live inside this repo under:

- [`reference_data/UI-Semen/pmf.dat`](/home/bshanks/freeGP-dev/reference_data/UI-Semen/pmf.dat)
- [`reference_data/wham.dat`](/home/bshanks/freeGP-dev/reference_data/wham.dat)

## Main Workflows

### 1. Single GP or HMC-NUTS Run

Use [`run_gprhd_hmc.py`](/home/bshanks/freeGP-dev/src/freegp/run_gprhd_hmc.py) for a single reconstruction run.

Deterministic stationary GP:

```bash
python /home/bshanks/freeGP-dev/src/freegp/run_gprhd_hmc.py \
  --mode gp \
  --kernel stationary \
  --dataset-root ~/freeGP-datasets/membranes/katka
```

Single NUTS run:

```bash
python /home/bshanks/freeGP-dev/src/freegp/run_gprhd_hmc.py \
  --mode nuts \
  --kernel stationary \
  --objective lml \
  --dataset-root ~/freeGP-datasets/membranes/katka \
  --warmup-steps 100 \
  --num-samples 100
```

Results are written by default to a timestamped directory under:

- [`results/`](/home/bshanks/freeGP-dev/results)

### 2. Ablation-Grid Study

Use [`run_ablation_grid.py`](/home/bshanks/freeGP-dev/src/freegp/run_ablation_grid.py) for publication-oriented ablation tests.

Example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
python /home/bshanks/freeGP-dev/src/freegp/run_ablation_grid.py \
  --dataset-root ~/freeGP-datasets/membranes/katka \
  --method nuts \
  --kernel stationary \
  --objective both \
  --include-fixed \
  --window-counts 25,13,7 \
  --trajectory-fractions 1.0,0.67,0.33 \
  --results-dir /home/bshanks/freeGP-dev/results/ablation-grid
```

When `--objective both` is used with `--method nuts`, the run produces:

- `results/ablation-grid/lml`
- `results/ablation-grid/loo`

and if `--include-fixed` is enabled:

- `results/ablation-grid/fixed`

## Config-Driven Runs

The ablation CLI supports TOML config files through `--config`.

Example smoke test:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
/home/bshanks/miniforge3/envs/freegp311/bin/python \
/home/bshanks/freeGP-dev/src/freegp/run_ablation_grid.py \
  --config /home/bshanks/freeGP-dev/configs/ablation/smoke.toml
```

Available starter configs:

- [`configs/ablation/smoke.toml`](/home/bshanks/freeGP-dev/configs/ablation/smoke.toml)
- [`configs/ablation/random_both_smoke.toml`](/home/bshanks/freeGP-dev/configs/ablation/random_both_smoke.toml)
- [`configs/ablation/overnight_stationary.toml`](/home/bshanks/freeGP-dev/configs/ablation/overnight_stationary.toml)

CLI flags override config values, so this is valid:

```bash
python /home/bshanks/freeGP-dev/src/freegp/run_ablation_grid.py \
  --config /home/bshanks/freeGP-dev/configs/ablation/smoke.toml \
  --num-samples 5 \
  --warmup-steps 5
```

## Randomized Ablation Modes

The ablation pipeline supports independent choices for:

- window selection
- trajectory retention

Window modes:

- `evenly_spaced`
- `random_subset`

Trajectory modes:

- `contiguous`
- `random_subsample`

Examples:

Random windows, contiguous trajectories:

```bash
python /home/bshanks/freeGP-dev/src/freegp/run_ablation_grid.py \
  --config /home/bshanks/freeGP-dev/configs/ablation/smoke.toml \
  --window-selection-mode random_subset \
  --trajectory-selection-mode contiguous
```

Random windows and random effective-sample trajectories:

```bash
python /home/bshanks/freeGP-dev/src/freegp/run_ablation_grid.py \
  --config /home/bshanks/freeGP-dev/configs/ablation/smoke.toml \
  --window-selection-mode random_subset \
  --trajectory-selection-mode random_subsample
```

If either random mode is active, the study defaults to:

- `5` selection replicates per ablation cell

You can override that with:

```bash
--selection-replicates 10
```

## Fixed-Hyperparameter Baseline

The fixed baseline uses the stationary kernel with defaults:

- `ell = pi / 2`
- `w = 4.184 * sqrt(10)`

This is intended to provide a Csanyi-style fixed-hyperparameter comparison in kJ/mol units.

## Result Outputs

A typical ablation result folder contains:

- `ablation_summary.png`
- `ablation_predictive_grid.png`
- `ablation_metrics.csv`
- `run_summary.txt`
- `hyperparameter_heatmaps.png` for NUTS runs
- `barrier_histograms_by_windows.png`
- `barrier_histograms_by_trajectory.png`
- `predictive_cells/`
- `nuts_diagnostics/` for HMC cells
- `artifacts/`

The saved artifact layer is designed so you can later reconstruct or resample saved GPs.

Important saved files include:

- `artifacts/study_manifest.json`
- `artifacts/references.npz`
- `artifacts/cells/<cell>.pt`

For NUTS runs, each cell artifact stores enough information to recover:

- the processed observation bundle
- the retained hyperposterior samples
- predictive summaries on the saved grid

## Current Objective Implementations

The repository currently supports two HMC objectives:

- `lml`
- `loo`

Both were recently checked carefully in the cleaned implementation:

- `lml` is the marginalized Gaussian likelihood for the profiled linear-offset model
- `loo` uses the projected precision required by the same profiled model

These are implemented in:

- [`gp.py`](/home/bshanks/freeGP-dev/src/freegp/gp.py)

## Notes On Performance

For CPU-based runs, it is often better to limit BLAS threading:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Single-chain NUTS is currently the most reliable mode for this workload. Multi-chain Pyro multiprocessing may work on some machines, but for this project it has been less stable than single-chain runs.

## Suggested Starting Points

Quick smoke test:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
/home/bshanks/miniforge3/envs/freegp311/bin/python \
/home/bshanks/freeGP-dev/src/freegp/run_ablation_grid.py \
  --config /home/bshanks/freeGP-dev/configs/ablation/smoke.toml
```

Randomized smoke test:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
/home/bshanks/miniforge3/envs/freegp311/bin/python \
/home/bshanks/freeGP-dev/src/freegp/run_ablation_grid.py \
  --config /home/bshanks/freeGP-dev/configs/ablation/random_both_smoke.toml
```

Longer overnight stationary ablation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
/home/bshanks/miniforge3/envs/freegp311/bin/python \
/home/bshanks/freeGP-dev/src/freegp/run_ablation_grid.py \
  --config /home/bshanks/freeGP-dev/configs/ablation/overnight_stationary.toml
```

## Status

This repository is now the streamlined implementation intended for ongoing development, debugging, and figure generation. The original notebook-heavy history has been preserved separately in the legacy backup.
