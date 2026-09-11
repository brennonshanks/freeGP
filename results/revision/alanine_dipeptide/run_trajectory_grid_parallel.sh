#!/usr/bin/env bash
# Launch the alanine trajectory-grid ablation with the four window counts
# running in parallel, each grid in its own tmux pane showing live HMC
# progress bars. Cell outputs land in the standard trajectory_grid directory;
# per-window-count console logs are tee'd to parallel_run_logs/.
#
# Usage:
#   ./run_trajectory_grid_parallel.sh                    # 500 warmup / 1000 samples / 4 chains
#   WARMUP=100 SAMPLES=100 ./run_trajectory_grid_parallel.sh   # quick override
#
# Safe to re-run: completed cells (completed.json + reconstruction CSV) are
# skipped by the grid runner.
#
# tmux controls: the session attaches automatically. Detach with Ctrl-b d
# (grids keep running); reattach with:  tmux attach -t alanine-grid
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-$HERE/../../../.venv/bin/python}"
SESSION="${SESSION:-alanine-grid}"
WINDOW_COUNTS="${WINDOW_COUNTS:-18 36 72 144}"
FRACTIONS="${FRACTIONS:-0.25 0.5 0.75 1.0}"
WARMUP="${WARMUP:-500}"
SAMPLES="${SAMPLES:-250}"
CHAINS="${CHAINS:-4}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/data_ablation_results/trajectory_grid}"
LOG_DIR="$OUTPUT_DIR/parallel_run_logs"
GRID_SCRIPT="$HERE/run_ablation_grid.py"

mkdir -p "$LOG_DIR"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required for the pane view; install with: brew install tmux" >&2
    exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2
    exit 1
fi

grid_command() {
    local w="$1"
    echo "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
        '$PYTHON' '$GRID_SCRIPT' \
        --window-counts $w \
        --fractions $FRACTIONS \
        --output-dir '$OUTPUT_DIR' \
        --warmup-steps $WARMUP \
        --num-samples $SAMPLES \
        --num-chains $CHAINS \
        --max-tree-depth $MAX_TREE_DEPTH \
        --run 2>&1 | tee '$LOG_DIR/${w}w.log'; \
        echo; echo '=== $w-window grid finished; pane will close in 10s ==='; sleep 10"
}

first=1
for w in $WINDOW_COUNTS; do
    if [ "$first" -eq 1 ]; then
        tmux new-session -d -s "$SESSION" -n grids "$(grid_command $w)"
        tmux select-pane -t "$SESSION" -T "${w}w"
        first=0
    else
        tmux split-window -t "$SESSION" "$(grid_command $w)"
        tmux select-pane -t "$SESSION" -T "${w}w"
        # Re-tile into a 2x2 grid after each split.
        tmux select-layout -t "$SESSION" tiled
    fi
done

tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format "#{pane_index}: #{pane_title}"
tmux set-option -t "$SESSION" mouse on

echo "Launched ${#WINDOW_COUNTS} window grids in tmux session '$SESSION'."
echo "Attaching (detach with Ctrl-b d; reattach with: tmux attach -t $SESSION)..."
tmux attach-session -t "$SESSION"
