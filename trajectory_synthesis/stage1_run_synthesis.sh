#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." &> /dev/null && pwd )"

cd "$REPO_ROOT"
echo "Working directory: $REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

CONFIG="${SCRIPT_DIR}/configs/WTCCC1_mcts_config.json"
LOG_DIR="${SCRIPT_DIR}/logs/WTCCC1"
RESULTS_DIR="${SCRIPT_DIR}/results/WTCCC1"
SEEDS="${SCRIPT_DIR}/seeds/seeds.jsonl"
# Root path of the scientific dataset to explore.
DATASET_PATH="${DATASET_PATH:-}"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

DATASET_PATH_ARGS=()
if [[ -n "$DATASET_PATH" ]]; then
  DATASET_PATH_ARGS=(--dataset_path "$DATASET_PATH")
fi

# Log file
python -u "${SCRIPT_DIR}/sci_pipeline.py" \
  --config "$CONFIG" \
  --seeds "$SEEDS" \
  --output-dir "$RESULTS_DIR" \
  --log-file "$LOG_DIR/sci_synthesis.log" \
  "${DATASET_PATH_ARGS[@]}" \
  "$@"
