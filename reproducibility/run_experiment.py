#!/usr/bin/env python3
"""Render or execute a registered paper experiment.

Execution is opt-in. Without --execute, the script prints shell-escaped commands
so configurations can be reviewed before expensive training begins.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "experiments.json"
TRAIN_COMMANDS = {
    "single_task": "nnUNet_train",
    "channel_preserve": "nnUNet_train_sequential_channel_preserve",
    "sequential": "nnUNet_train_sequential",
    "rehearsal": "nnUNet_train_rehearsal",
    "lwf": "nnUNet_train_lwf",
    "ewc": "nnUNet_train_ewc",
}
EVAL_TRAINERS = {
    "channel_preserve": "nnUNetTrainerSequentialChannelPreserve",
    "sequential": "nnUNetTrainerSequential",
    "rehearsal": "nnUNetTrainerRehearsal",
    "lwf": "nnUNetTrainerLWF",
    "ewc": "nnUNetTrainerEWC",
}
REQUIRED_ENV = ("nnUNet_raw_data_base", "nnUNet_preprocessed", "RESULTS_FOLDER", "EVALUATION_FOLDER")


def load_manifest():
    manifest = json.loads(MANIFEST.read_text())
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    ids = [exp["id"] for exp in manifest["experiments"]]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise SystemExit("Duplicate experiment ids: " + ", ".join(duplicates))
    known_tasks = set(manifest["datasets"])
    for exp in manifest["experiments"]:
        if exp["method"] not in TRAIN_COMMANDS:
            raise SystemExit(f"{exp['id']}: unsupported method {exp['method']}")
        unknown = [str(task) for task in exp["tasks"] if str(task) not in known_tasks]
        if unknown:
            raise SystemExit(f"{exp['id']}: unregistered task(s): {', '.join(unknown)}")
        if exp["method"] == "channel_preserve" and not exp.get("full_body_freeze"):
            declared = set(exp.get("blocks", []))
            derived = {"B" if i == 6 else f"E{i}" for i in exp.get("encoder_cli", [])}
            derived |= {f"D{i}" for i in exp.get("decoder_cli", [])}
            if declared != derived:
                raise SystemExit(
                    f"{exp['id']}: blocks {sorted(declared)} do not match CLI selection {sorted(derived)}"
                )


def experiment_by_id(manifest, experiment_id):
    matches = [x for x in manifest["experiments"] if x["id"] == experiment_id]
    if not matches:
        choices = ", ".join(x["id"] for x in manifest["experiments"])
        raise SystemExit(f"Unknown experiment '{experiment_id}'. Available: {choices}")
    return matches[0]


def add_many(command, flag, values):
    if values:
        command.extend([flag, *map(str, values)])


def training_command(exp, defaults):
    if exp["method"] == "single_task":
        return [
            "nnUNet_train", defaults["network"], "nnUNetTrainerV2",
            str(exp["tasks"][0]), str(defaults["fold"]), "--npz",
        ]
    command = [TRAIN_COMMANDS[exp["method"]], defaults["network"]]
    add_many(command, "-t", exp["tasks"])
    command.extend(["-f", str(defaults["fold"]), "-s", defaults["split"], "-p", exp["plan"]])
    command.extend(["-num_epochs", str(defaults["epochs"]), "-save_interval", str(defaults["save_interval"])])

    if exp["method"] == "channel_preserve":
        init_path = os.environ.get("PAPER_INIT_CHECKPOINT")
        if not init_path:
            raise SystemExit("PAPER_INIT_CHECKPOINT is required for controlled channel-preserve runs")
        command.extend([
            "--init_seq", "-initialize_with_network_trainer", "nnUNetTrainerSequential",
            "--init_from_trainer_path", init_path,
            "-used_identifier_in_init_network_trainer", "nnUNetPlansv2.1",
        ])
        if exp.get("full_body_freeze"):
            command.append("--full_body_freeze")
        add_many(command, "--encoder_unfreeze", exp.get("encoder_cli"))
        add_many(command, "--decoder_unfreeze", exp.get("decoder_cli"))
        if exp.get("base_lr") is not None:
            command.extend(["--base_learning_rate", str(exp["base_lr"])])
        if exp.get("run_seed") is not None:
            command.extend(["--run_seed", str(exp["run_seed"])])
        if exp.get("gradient_lr_control"):
            command.extend([
                "--gradient_lr_control",
                "--gradient_calibration_batches", str(exp["gradient_calibration_batches"]),
                "--gradient_lr_reference", exp["gradient_lr_reference"],
                "--gradient_displacement_log_interval", str(exp["gradient_displacement_log_interval"]),
            ])
    elif exp["method"] == "rehearsal":
        command.extend(["-samples_in_perc", str(exp["samples_in_perc"]), "-seed", str(exp["rehearsal_seed"])])
    elif exp["method"] == "lwf":
        command.extend(["-lwf_temperature", str(exp["lwf_temperature"])])
    elif exp["method"] == "ewc":
        command.extend(["-ewc_lambda", str(exp["ewc_lambda"])])
    return command


def evaluation_commands(exp, defaults, datasets):
    if exp["method"] == "single_task":
        task = str(exp["tasks"][0])
        task_name = datasets[task]["name"]
        raw_root = Path(os.environ["nnUNet_raw_data_base"]) / "nnUNet_raw_data" / task_name
        output = (Path(os.environ["EVALUATION_FOLDER"]) / "predictions" /
                  "baselines_single_dataset" / task_name / f"fold{defaults['fold']}")
        labels = list(range(1, len(datasets[task]["targets"]) + 1))
        return [
            ["nnUNet_predict", "-i", str(raw_root / "imagesTs"), "-o", str(output),
             "-t", task, "-m", defaults["network"], "-f", str(defaults["fold"]), "--save_npz"],
            ["nnUNet_evaluate_folder", "-ref", str(raw_root / "labelsTs"),
             "-pred", str(output), "-l", *map(str, labels)],
        ]
    trainer = EVAL_TRAINERS[exp["method"]]
    commands = []
    for task in exp["tasks"]:
        command = [
            "nnUNet_evaluate2", defaults["network"], trainer,
            "-trained_on", *map(str, exp["tasks"]), "-p", exp["plan"],
            "-use_model", *map(str, exp["tasks"]),
            "-evaluate_on", str(task), "-use_head", str(task),
            "-f", str(defaults["fold"]), "--store_csv",
            "-chk", defaults["checkpoint"], "-no_delete",
        ]
        commands.append(command)
    return commands


def validate_environment(commands, execute):
    missing_env = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing_env:
        raise SystemExit("Missing environment variables: " + ", ".join(missing_env) +
                         ". Source reproducibility/config.example.env first.")
    if execute:
        missing_commands = sorted({cmd[0] for cmd in commands if shutil.which(cmd[0]) is None})
        if missing_commands:
            raise SystemExit("Commands not found in PATH: " + ", ".join(missing_commands))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", help="experiment id from experiments.json")
    parser.add_argument("--group", help="render or execute every experiment in a registered group")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--execute", action="store_true", help="execute commands; default is dry-run")
    parser.add_argument("--list", action="store_true", help="list registered experiments")
    args = parser.parse_args()

    manifest = load_manifest()
    if args.list:
        for exp in manifest["experiments"]:
            print(f"{exp['id']:<38} {exp['group']:<20} tasks={exp['tasks']}")
        return
    if bool(args.experiment) == bool(args.group):
        parser.error("provide exactly one experiment id or --group (or use --list)")

    if args.group:
        experiments = [exp for exp in manifest["experiments"] if exp["group"] == args.group]
        if not experiments:
            groups = sorted({exp["group"] for exp in manifest["experiments"]})
            raise SystemExit(f"Unknown group '{args.group}'. Available: {', '.join(groups)}")
    else:
        experiments = [experiment_by_id(manifest, args.experiment)]

    for exp in experiments:
        commands = []
        if args.stage in ("train", "all"):
            commands.append(training_command(exp, manifest["defaults"]))
        if args.stage in ("evaluate", "all"):
            commands.extend(evaluation_commands(exp, manifest["defaults"], manifest["datasets"]))
        validate_environment(commands, args.execute)
        print(f"\n# experiment: {exp['id']}", flush=True)
        for command in commands:
            print("$ " + shlex.join(command), flush=True)
            if args.execute:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
