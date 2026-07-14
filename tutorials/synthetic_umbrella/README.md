# Synthetic Umbrella-Sampling Tutorial

This tutorial creates a small one-dimensional umbrella-sampling dataset from a
known free-energy surface and reconstructs it with freeGP.

The synthetic target surface is

\[
F(x)=8(x^2-1)^2 + 1.5\sin(4x).
\]

For each umbrella center \(c_i\), the script samples positions from

\[
p_i(x) \propto
\exp[-\beta(F(x) + \tfrac{1}{2}k(x-c_i)^2)].
\]

This mimics umbrella sampling without requiring molecular dynamics software.
The generated files use the simple flat dataset format supported by freeGP:

```text
example_data/
  README              # window_id, center [nm], force constant [kJ/mol/nm^2]
  w00-pullx.xvg
  w01-pullx.xvg
  ...
  ground_truth.csv    # exact synthetic reference surface
```

## Run

From this directory:

```bash
python make_dataset.py
python run_reconstruction.py
```

Outputs are written to:

```text
tutorial_results/
  synthetic_reconstruction.png
  synthetic_reconstruction.csv
```

The default run compares:

- the known reference surface
- a fixed-hyperparameter GP
- a MAP hierarchical GP using the LOO objective and default hyperpriors
- a HMC-NUTS hyperposterior-propagated GP

The default HMC settings are those used in the manuscript:

```bash
python run_reconstruction.py \
  --warmup-steps 500 \
  --num-samples 1000 \
  --predictive-samples 100 \
  --num-chains 4
```

To skip HMC and run only the fixed and MAP reconstructions:

```bash
python run_reconstruction.py --skip-hmc
```

## Trying Your Own Data

See the top-level repository `README.md` for the supported umbrella-sampling
data layouts and unit conventions. This tutorial script can be pointed at a
compatible dataset with:

```bash
python run_reconstruction.py \
  --dataset-root /path/to/my_dataset \
  --reference-path /path/to/reference.csv
```

The reference path is optional and is used only for plotting/comparison.

## Notes

- Use enough umbrella windows to cover the region where you want predictions.
- Make sure the coordinate units and force constants are consistent.
- Start with MAP optimization before running HMC-NUTS.
- For real analyses, increase HMC warmup/sampling and check diagnostics.
