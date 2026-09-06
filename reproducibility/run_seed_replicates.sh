#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PAPER_PYTHON:-python}"

source "${SCRIPT_DIR}/config.example.env"
cd "${REPO_ROOT}"
"${SCRIPT_DIR}/prepare_seed_plan_aliases.sh"

experiments=(
  seed23456_bottleneck
  seed23456_middle
  seed23456_broad
  seed34567_bottleneck
  seed34567_middle
  seed34567_broad
)

echo "[$(date --iso-8601=seconds)] Starting ${#experiments[@]} sequential seed-replicate runs."
for experiment in "${experiments[@]}"; do
  echo "[$(date --iso-8601=seconds)] TRAIN ${experiment}"
  "${PYTHON_BIN}" reproducibility/run_experiment.py "${experiment}" --stage train --execute
  echo "[$(date --iso-8601=seconds)] EVALUATE ${experiment}"
  "${PYTHON_BIN}" reproducibility/run_experiment.py "${experiment}" --stage evaluate --execute
  echo "[$(date --iso-8601=seconds)] COMPLETE ${experiment}"
done
echo "[$(date --iso-8601=seconds)] All seed-replicate runs and evaluations completed."
