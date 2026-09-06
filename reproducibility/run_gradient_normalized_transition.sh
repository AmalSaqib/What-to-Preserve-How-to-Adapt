#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PAPER_PYTHON:-python}"

source "${SCRIPT_DIR}/config.example.env"
cd "${REPO_ROOT}"

plan_names=(
  nnUNetPlansv2.1_unfreeze_E56D5_gradnorm
  nnUNetPlansv2.1_unfreeze_E3456D345_gradnorm
)

for task in Task666_UMD Task555_ECPC; do
  plan_dir="${nnUNet_preprocessed}/${task}"
  source_plan="${plan_dir}/nnUNetPlansv2.1_plans_3D.pkl"
  if [[ ! -f "${source_plan}" ]]; then
    echo "Missing base plans file: ${source_plan}" >&2
    exit 1
  fi

  for plan in "${plan_names[@]}"; do
    target_plan="${plan_dir}/${plan}_plans_3D.pkl"
    if [[ -e "${target_plan}" ]] && ! cmp -s "${source_plan}" "${target_plan}"; then
      echo "Refusing to replace non-identical plans file: ${target_plan}" >&2
      exit 1
    fi
    if [[ ! -e "${target_plan}" ]]; then
      cp "${source_plan}" "${target_plan}"
    fi
  done
done

experiments=(
  ecpc_e5_gradient_normalized
  ecpc_e3_gradient_normalized
)

echo "[$(date --iso-8601=seconds)] Starting gradient-normalized transition controls."
for experiment in "${experiments[@]}"; do
  echo "[$(date --iso-8601=seconds)] TRAIN ${experiment}"
  "${PYTHON_BIN}" reproducibility/run_experiment.py "${experiment}" --stage train --execute
  echo "[$(date --iso-8601=seconds)] EVALUATE ${experiment}"
  "${PYTHON_BIN}" reproducibility/run_experiment.py "${experiment}" --stage evaluate --execute
  echo "[$(date --iso-8601=seconds)] COMPLETE ${experiment}"
done
echo "[$(date --iso-8601=seconds)] All gradient-normalized transition controls completed."
