#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/bshanks/freeGP-dev"
PYTHON_BIN="${PYTHON_BIN:-/home/bshanks/miniforge3/envs/freegp311/bin/python}"
RUNNER="${RUNNER:-$REPO_ROOT/src/freegp/run_ablation_grid.py}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/hmc-calibration-sweep}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
DATASET_ROOT="${DATASET_ROOT:-~/freeGP-datasets/membranes/katka}"

CASES=(easy medium hard super_hard)
WARMUPS=(250 500 1000)
SAMPLES=(250 500 1000)

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--cases easy,medium,hard,super_hard] [--warmups 250,500,...] [--samples 250,500,...]

Optional env vars:
  PYTHON_BIN     Python executable
  RESULTS_ROOT   Root output directory
  DATASET_ROOT    Umbrella-sampling dataset root
  MPLCONFIGDIR   Matplotlib cache dir
  OMP_NUM_THREADS / MKL_NUM_THREADS
USAGE
}

split_csv() {
  local input="$1"
  local -n out_ref=$2
  IFS=',' read -r -a out_ref <<< "$input"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cases)
      split_csv "$2" CASES
      shift 2
      ;;
    --warmups)
      split_csv "$2" WARMUPS
      shift 2
      ;;
    --samples)
      split_csv "$2" SAMPLES
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$RESULTS_ROOT"

case_settings() {
  case "$1" in
    easy) echo "25 1.0" ;;
    medium) echo "13 0.5" ;;
    hard) echo "7 0.25" ;;
    super_hard) echo "3 0.1" ;;
    *)
      echo "Unknown case: $1" >&2
      exit 1
      ;;
  esac
}

write_config() {
  local path="$1"
  local windows="$2"
  local fraction="$3"
  local output_dir="$4"
  cat > "$path" <<EOF
[ablation]
dataset_root = "$DATASET_ROOT"
method = "nuts"
kernel = "stationary"
objective = "loo"
include_fixed = false
device = "cpu"

window_counts = [$windows]
trajectory_fractions = [$fraction]

window_selection_mode = "evenly_spaced"
trajectory_selection_mode = "contiguous"
random_seed = 42

num_bins = 20
num_test_points = 100
warmup_steps = 1000
num_samples = 1000
num_chains = 4
predictive_samples = 100
barrier_bins = 30

results_dir = "$output_dir"
EOF
}

for case_name in "${CASES[@]}"; do
  read -r windows fraction <<< "$(case_settings "$case_name")"

  for warmup in "${WARMUPS[@]}"; do
    for samples in "${SAMPLES[@]}"; do
      run_dir="$RESULTS_ROOT/${case_name}_w${warmup}_s${samples}"
      config_path="$run_dir/config.toml"
      mkdir -p "$run_dir"
      write_config "$config_path" "$windows" "$fraction" "$run_dir"
      echo "=== ${case_name} | warmup=${warmup} | samples=${samples} ==="
      OMP_NUM_THREADS="$OMP_NUM_THREADS" \
      MKL_NUM_THREADS="$MKL_NUM_THREADS" \
      MPLCONFIGDIR="$MPLCONFIGDIR" \
      "$PYTHON_BIN" "$RUNNER" \
        --config "$config_path" \
        --warmup-steps "$warmup" \
        --num-samples "$samples" \
        --results-dir "$run_dir"
    done
  done
done
