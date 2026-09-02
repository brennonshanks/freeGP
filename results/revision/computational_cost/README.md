# Computational-cost benchmark

This directory contains the reproducible timing analysis for the R9 membrane
data-ablation study. The benchmark compares UI with block averaging, the fixed
GP, the LOO-MAP GP, and the LOO hierarchical GP under the three conditions used
in the archived HMC calibration tests:

- sparse (`super_hard`): 3 evenly spaced windows and 10% of each trajectory;
- intermediate (`hard`): 7 evenly spaced windows and 25% of each trajectory;
- full (`easy`): all 25 windows and 100% of each trajectory.

Each benchmark invocation reconstructs one canonical, evenly spaced window
selection, matching those archived tests.

The paper profile uses the manuscript settings: 20 bins, 100 prediction points,
250 MAP optimization steps with three restarts, and HMC-NUTS with 500 warmup
steps, 1,000 retained samples, four chains, and 100 predictive samples.

First inspect the planned commands without running them:

```bash
source .venv/bin/activate
python results/revision/benchmarks/run_computational_benchmarks.py
```

Validate the harness using reduced inference settings:

```bash
python results/revision/benchmarks/run_computational_benchmarks.py \
  --profile smoke --conditions sparse --warmup-runs 0 --repeats 1 --execute
```

Run the publication benchmark while the laptop is connected to power, Low
Power Mode is disabled, and computationally intensive applications are closed:

```bash
python results/revision/benchmarks/run_computational_benchmarks.py \
  --profile paper --warmup-runs 1 --repeats 3 --threads 1 --execute
```

Every run is executed in a fresh subprocess. Method order is shuffled within
each condition and repeat to reduce systematic thermal/order bias. The session
directory contains `metadata.json`, `commands.txt`, `raw_timings.csv`,
`summary.csv`, subprocess logs, and normal scientific result artifacts.

Wall time includes Python interpreter startup, trajectory loading,
preprocessing, inference, prediction, and output serialization. It excludes
molecular dynamics trajectory generation. `summary.csv` reports median,
minimum, maximum, and runtime relative to the fixed GP for each condition.
