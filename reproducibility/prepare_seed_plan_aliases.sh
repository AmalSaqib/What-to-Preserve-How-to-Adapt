#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.example.env"

datasets=(Task666_UMD Task555_ECPC)
base_plans=(
  nnUNetPlansv2.1_unfreeze_E6
  nnUNetPlansv2.1_unfreeze_E456D45
  nnUNetPlansv2.1_unfreeze_E123456D12345
)
seeds=(23456 34567)

for dataset in "${datasets[@]}"; do
  dataset_dir="${nnUNet_preprocessed}/${dataset}"
  for base_plan in "${base_plans[@]}"; do
    source_plan="${dataset_dir}/${base_plan}_plans_3D.pkl"
    if [[ ! -f "${source_plan}" ]]; then
      echo "Missing source plans file: ${source_plan}" >&2
      exit 1
    fi
    for seed in "${seeds[@]}"; do
      target_plan="${dataset_dir}/${base_plan}_seed${seed}_plans_3D.pkl"
      if [[ -e "${target_plan}" ]]; then
        if ! cmp -s "${source_plan}" "${target_plan}"; then
          echo "Existing alias differs from source: ${target_plan}" >&2
          exit 1
        fi
      else
        cp "${source_plan}" "${target_plan}"
        echo "Created ${target_plan}"
      fi
    done
  done
done
