#!/usr/bin/env python3
"""Validate nnU-Net raw datasets and optionally run nnU-Net preprocessing."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_TASKS = (555, 666, 778)
MANIFEST = Path(__file__).resolve().parent / "experiments.json"


def raw_data_root() -> Path:
    base = os.environ.get("nnUNet_raw_data_base")
    if not base:
        raise SystemExit("nnUNet_raw_data_base is unset; source reproducibility/config.example.env")
    return Path(base) / "nnUNet_raw_data"


def case_name(path_string: str) -> str:
    name = Path(path_string).name
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def validate_task(task_dir: Path, require_test_labels: bool = False):
    errors = []
    dataset_path = task_dir / "dataset.json"
    if not dataset_path.is_file():
        return [f"missing {dataset_path}"]
    data = json.loads(dataset_path.read_text())
    modalities = sorted(int(index) for index in data.get("modality", {}))
    training = data.get("training", [])
    testing = data.get("test", [])
    if data.get("numTraining") != len(training):
        errors.append(f"numTraining={data.get('numTraining')} but training has {len(training)} entries")
    if data.get("numTest") != len(testing):
        errors.append(f"numTest={data.get('numTest')} but test has {len(testing)} entries")
    if not modalities:
        errors.append("no modalities declared")
    if "0" not in data.get("labels", {}):
        errors.append("background label 0 is missing")

    def check_images(folder: str, cases):
        for entry in cases:
            base = case_name(entry)
            for modality in modalities:
                path = task_dir / folder / f"{base}_{modality:04d}.nii.gz"
                if not path.is_file():
                    errors.append(f"missing modality file {path.relative_to(task_dir)}")

    for entry in training:
        base = case_name(entry["image"])
        label = task_dir / "labelsTr" / f"{base}.nii.gz"
        if not label.is_file():
            errors.append(f"missing label {label.relative_to(task_dir)}")
    check_images("imagesTr", [entry["image"] for entry in training])
    check_images("imagesTs", testing)
    if require_test_labels:
        for entry in testing:
            base = case_name(entry)
            label = task_dir / "labelsTs" / f"{base}.nii.gz"
            if not label.is_file():
                errors.append(f"missing held-out label {label.relative_to(task_dir)}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", type=int, default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--require-test-labels",
        action="store_true",
        help="also require labelsTs files used by the paper evaluation",
    )
    parser.add_argument(
        "--skip-registry-counts",
        action="store_true",
        help="validate dataset.json internally without enforcing paper cohort counts",
    )
    parser.add_argument("--execute", action="store_true", help="run nnUNet_plan_and_preprocess after validation")
    args = parser.parse_args()

    root = raw_data_root()
    registered = json.loads(MANIFEST.read_text())["datasets"]
    all_errors = []
    names = []
    for task_id in args.tasks:
        matches = sorted(root.glob(f"Task{task_id:03d}_*"))
        if len(matches) != 1:
            all_errors.append(f"task {task_id}: expected one directory, found {len(matches)}")
            continue
        task_dir = matches[0]
        names.append(task_dir.name)
        errors = validate_task(task_dir, require_test_labels=args.require_test_labels)
        if not args.skip_registry_counts:
            specification = registered.get(str(task_id))
            if specification is None:
                errors.append("task is not registered in experiments.json")
            elif (task_dir / "dataset.json").is_file():
                dataset = json.loads((task_dir / "dataset.json").read_text())
                if dataset.get("numTraining") != specification["train"]:
                    errors.append(
                        f"registered train count is {specification['train']}, "
                        f"dataset.json reports {dataset.get('numTraining')}"
                    )
                if dataset.get("numTest") != specification["test"]:
                    errors.append(
                        f"registered test count is {specification['test']}, "
                        f"dataset.json reports {dataset.get('numTest')}"
                    )
        all_errors.extend(f"{task_dir.name}: {error}" for error in errors)
        print(f"{task_dir.name}: {'OK' if not errors else f'{len(errors)} error(s)'}")
    if all_errors:
        print("\n".join(f"ERROR: {error}" for error in all_errors))
        raise SystemExit(1)

    print(f"Validated {len(names)} dataset(s): {', '.join(names)}")
    if args.execute:
        command_name = "nnUNet_plan_and_preprocess"
        if shutil.which(command_name) is None:
            raise SystemExit(f"{command_name} is not available in PATH")
        command = [command_name, "-t", *map(str, args.tasks), "--verify_dataset_integrity"]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
