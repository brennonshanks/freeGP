#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/bshanks/freeGP-dev"
PYTHON_BIN="${PYTHON_BIN:-/home/bshanks/miniforge3/envs/freegp311/bin/python}"
RUNNER="${RUNNER:-$REPO_ROOT/src/freegp/run_ablation_grid.py}"
CONFIG_DIR="${CONFIG_DIR:-$REPO_ROOT/configs/ablation}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results/hmc-calibration-sweep}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

CASES=(easy medium hard)
WARMUPS=(250 500 1000)
SAMPLES=(250 500 1000)

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--cases easy,medium,hard] [--warmups 250,500,...] [--samples 250,500,...]

Optional env vars:
  PYTHON_BIN     Python executable
  RESULTS_ROOT   Root output directory
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

for case_name in "${CASES[@]}"; do
  config_path="$CONFIG_DIR/hmc_calibration_${case_name}.toml"
  if [[ ! -f "$config_path" ]]; then
    echo "Missing config: $config_path" >&2
    exit 1
  fi

  for warmup in "${WARMUPS[@]}"; do
    for samples in "${SAMPLES[@]}"; do
      run_dir="$RESULTS_ROOT/${case_name}_w${warmup}_s${samples}"
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
