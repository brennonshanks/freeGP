# freeGP

Gaussian-process tools for reconstructing one-dimensional free-energy surfaces
from enhanced-sampling data. The code supports umbrella-sampling workflows,
metadynamics trajectory-length comparisons, fixed-hyperparameter GP baselines,
MAP hyperparameter estimates, and HMC-NUTS hyperposterior uncertainty
propagation.

## Install

```bash
cd /path/to/freeGP-dev
python -m pip install -e ".[all]"
```

The package requires Python 3.11 or newer. Core dependencies are listed in
`pyproject.toml`; `requirements.txt` mirrors the same runtime stack plus
notebook tools.

For long CPU runs, use single-threaded BLAS unless you have tested otherwise:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl
```

## Repository Layout

```text
configs/                 TOML configs for ablation and single-run studies
docs/                    SI method notes and draft text
reference_data/          WHAM/UI reference data bundled with the repo
results/                 Generated analysis outputs and paper figures
scripts/                 Paper-analysis scripts and visualization entry point
src/freegp/              Installable Python package
tutorials/               Small synthetic examples for new users
```

Important package modules:

```text
src/freegp/data.py             Dataset and reference-curve loading
src/freegp/preprocess.py       Umbrella histogram/derivative preprocessing
src/freegp/metadynamics.py     Metadynamics discretization/preprocessing
src/freegp/gp.py               Stationary and Gibbs GP models
src/freegp/hyperopt.py         MAP hyperparameter optimization
src/freegp/hmc.py              Pyro HMC-NUTS sampling
src/freegp/posterior.py        Predictive uncertainty summaries
src/freegp/metrics.py          RMSE and uncertainty metrics
src/freegp/studies/ablation.py Ablation-grid study logic
```

## Data

Simulation data are kept outside the repo. Point scripts to a dataset directly
with `--dataset-root` / `--data-root`, or set:

```bash
export FREEGP_DATASETS=/path/to/freeGP-datasets
```

The current paper analyses use membrane umbrella-sampling data, sugar
umbrella-sampling data, and a membrane metadynamics dataset.

### Umbrella-Sampling Data Layout

freeGP expects one-dimensional umbrella-sampling trajectories. Internally, the
examples assume coordinates in nm, energies in kJ/mol, and umbrella force
constants in kJ/mol/nm^2.

The GPR(H+D) workflow builds histogram observations of the local biased
distribution and derivative observations from the umbrella restoring force.
Each window therefore needs a trajectory, an umbrella center, and a harmonic
force constant.

The simplest supported layout is:

```text
my_dataset/
  README
  window0-pullx.xvg
  window1-pullx.xvg
  ...
```

The `README` file contains whitespace-separated rows:

```text
window_id  umbrella_center_nm  force_constant_kJ_mol_nm2
```

Example:

```text
# window_id center_nm force_constant_kJ_mol_nm2
w00 -1.50 1000.0
w01 -1.25 1000.0
w02 -1.00 1000.0
```

Each `*-pullx.xvg` file contains at least two columns:

```text
time  coordinate
```

Comment lines beginning with `#` or `@` are ignored. The `window_id` in the
README must match the filename prefix before `-pullx.xvg`.

Optional reference PMFs should contain at least two columns:

```text
x_nm  F_kJ_per_mol
```

Reference curves are only used for plotting/comparison; they are not required
to fit the GP.

If your simulation uses different units, convert the input files before running
freeGP or add a small preprocessing wrapper. Common conversions:

- Angstrom to nm: multiply coordinates and umbrella centers by `0.1`.
- kJ/mol/A^2 to kJ/mol/nm^2: multiply force constants by `100`.
- kcal/mol to kJ/mol: multiply energies and force constants by `4.184`.

freeGP also supports a Gromacs-style directory layout:

```text
my_gromacs_dataset/
  d_1.45/
    step7_production_pullx.xvg
    step7_production.mdp
  d_1.60/
    step7_production_pullx.xvg
    step7_production.mdp
  ...
```

The folder name must contain the umbrella center as `d_<center>`, and the MDP
file must contain the force constant. If your data come from another simulation
package, convert them to the flat layout above.

### Metadynamics Data Layout

The metadynamics scripts expect a PLUMED-style trajectory file, normally named
`COLVAR`, with columns for time, metadynamics auxiliary coordinate, physical
collective variable, and bias. The default column names used by the scripts are:

```text
time  MetaCV  CV  bias  lower  upper
```

An optional reference FES file, normally `fes.dat`, can be supplied for
plotting/comparison. It is not required for fitting.

Metadynamics trajectories are not naturally split into umbrella windows, so
the code discretizes them after the run. The main controls are:

```bash
--interval 0 4.5
--n-histogram-windows 120
--n-derivative-bins 60
--histogram-binning quantile
--derivative-binning quantile
```

`--n-histogram-windows` controls the pseudo-window histogram observations, and
`--n-derivative-bins` controls the binned mean-force observations. `quantile`
binning gives approximately equal numbers of samples per bin; `uniform`
binning gives equal spatial widths. Increase bin counts for more spatial
resolution, but reduce them if each bin becomes too noisy or HMC becomes too
slow.

## Main Workflows

### Tutorials

For a small self-contained example that does not require the paper datasets:

```bash
cd tutorials/synthetic_umbrella
python make_dataset.py
python run_reconstruction.py
```

The tutorial creates pseudo-umbrella trajectories from a known synthetic
surface and reconstructs them with fixed, MAP, and short HMC-NUTS GP models.

### Single GP/HMC Run

```bash
python -m freegp.run_gprhd_hmc \
  --mode nuts \
  --kernel stationary \
  --objective loo \
  --dataset-root /path/to/umbrella/data \
  --warmup-steps 500 \
  --num-samples 1000
```

Installed CLI equivalent:

```bash
freegp-gprhd-hmc --mode nuts --objective loo --dataset-root /path/to/data
```

### Ablation Grid

Use configs for reproducible benchmark runs:

```bash
python -m freegp.run_ablation_grid \
  --config configs/ablation/membrane.toml
```

Installed CLI equivalent:

```bash
freegp-ablation-grid --config configs/ablation/membrane.toml
```

The main ablation runner can compare fixed GP, MAP GP, and HMC-NUTS
hyperposterior methods over trajectory fractions and umbrella-window counts.
Outputs are written under `results/`.

### Metadynamics Trajectory-Length Comparison

```bash
python scripts/compare_metadynamics_trajectory_lengths.py \
  --data-root /path/to/metadynamics/data \
  --interval 0 4.5 \
  --n-histogram-windows 120 \
  --n-derivative-bins 60 \
  --histogram-binning quantile \
  --derivative-binning quantile \
  --objective loo \
  --warmup-steps 500 \
  --num-samples 1000 \
  --predictive-samples 100 \
  --results-dir results/metadynamics
```

The script saves per-fraction PMFs and a `trajectory_length_metrics.csv` file
used by the paper visualization code.

### Prior Sensitivity

```bash
python scripts/compare_lengthscale_priors.py \
  --window-count 7 \
  --trajectory-fraction 0.25 \
  --warmup-steps 500 \
  --num-samples 1000 \
  --num-chains 4 \
  --results-dir results/lengthscale-prior-sensitivity
```

This compares length-scale priors while keeping the other hyperpriors fixed.

## Paper Figures

Paper-ready figures are generated from saved result summaries with:

```bash
python scripts/visualization.py --figures all
```

To regenerate only one group:

```bash
python scripts/visualization.py --figures main_ablation
python scripts/visualization.py --figures metadynamics_convergence
python scripts/visualization.py --figures lengthscale_prior_sensitivity
python scripts/visualization.py --figures noise_comparison
```

Figures are saved as vector outputs under `results/**/paper_figures/`.

## Development Notes

- Prefer running package entry points through `python -m freegp...` or the
  installed console scripts.
- Keep large generated artifacts in `results/`; do not add new analysis logic
  inside result folders unless it is temporary.
- Use `scripts/visualization.py` for manuscript figure formatting so fonts,
  sizes, labels, and output paths stay consistent.
