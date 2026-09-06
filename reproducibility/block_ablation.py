#!/usr/bin/env python3
"""Run full-block activation ablations with ``nnUNet_evaluate2``.

The command is a dry run unless ``--execute`` is supplied. Each encoder and
decoder block is zeroed independently at inference; no parameters are updated.
The resulting Dice decrease measures functional reliance, not the causal effect
of training or freezing that block.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch


def load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def block_counts(path: Path) -> tuple[int, int]:
    checkpoint = load_checkpoint(path)
    state = checkpoint.get("state_dict", checkpoint.get("network_weights", checkpoint))
    encoder, decoder = set(), set()
    for raw_key in state:
        key = str(raw_key)
        for prefix in ("module.", "network.", "model.", "body."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        parts = key.split(".")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        if parts[0] == "conv_blocks_context":
            encoder.add(int(parts[1]))
        elif parts[0] == "conv_blocks_localization":
            decoder.add(int(parts[1]))
    if not encoder or not decoder:
        raise ValueError("could not infer encoder and decoder blocks from checkpoint")
    return max(encoder) + 1, max(decoder) + 1


def load_macro_dice(path: Path) -> float:
    raw = json.loads(path.read_text())
    per_label: dict[str, list[float]] = {}
    for epoch in raw.values():
        for task in epoch.values():
            for case in task.values():
                for label, metrics in case.items():
                    value = metrics.get("Dice")
                    if value is not None:
                        per_label.setdefault(label, []).append(float(value))
    means = [float(np.mean(values)) for values in per_label.values() if values]
    if not means:
        raise ValueError(f"no Dice values found in {path}")
    return float(np.mean(means))


def newest_metrics(root: Path, required_tag: str) -> Path:
    candidates = [
        path
        for path in root.rglob("val_metrics_eval.json")
        if required_tag in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no val_metrics_eval.json beneath {root} contains directory {required_tag!r}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def task_name(raw_data_base: Path, task: str) -> str:
    if task.startswith("Task"):
        return task
    matches = sorted(raw_data_base.glob(f"Task{int(task):03d}_*"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one Task{int(task):03d}_* under {raw_data_base}; "
            f"found {len(matches)}"
        )
    return matches[0].name


def prediction_tag(task: str, kind: str, index: int, seed: int) -> str:
    return f"Preds_{task}_{kind}block{index}_drop100_seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--raw-task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trained-on", nargs="+", required=True)
    parser.add_argument("--use-model", nargs="+", required=True)
    parser.add_argument("--use-head", required=True)
    parser.add_argument("--evaluate-on", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--network", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default="model_final_checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-command", default="nnUNet_evaluate2")
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    encoder_count, decoder_count = block_counts(args.checkpoint_path)
    evaluated_task = task_name(args.raw_task_root, args.evaluate_on)
    base = [
        args.evaluate_command,
        args.network,
        args.trainer,
        "-trained_on",
        *args.trained_on,
        "-p",
        args.plans,
        "-use_model",
        *args.use_model,
        "-evaluate_on",
        args.evaluate_on,
        "-use_head",
        args.use_head,
        "-f",
        str(args.fold),
        "--store_csv",
        "-chk",
        args.checkpoint,
        "-no_delete",
    ]
    commands: list[tuple[str, int, list[str]]] = []
    for index in range(encoder_count):
        commands.append(
            (
                "encoder",
                index,
                base
                + [
                    "--drop_encoder_block",
                    str(index),
                    "--drop_feature_ratio",
                    "1.0",
                    "--drop_seed",
                    str(args.seed),
                ],
            )
        )
    for index in range(decoder_count):
        commands.append(
            (
                "decoder",
                index,
                base
                + [
                    "--drop_decoder_block",
                    str(index),
                    "--drop_feature_ratio",
                    "1.0",
                    "--drop_seed",
                    str(args.seed),
                ],
            )
        )

    print("# baseline")
    print("$ " + shlex.join(base))
    for kind, index, command in commands:
        print(f"# {kind} block {index}")
        print("$ " + shlex.join(command))
    if not args.execute:
        print("\nDry run only; add --execute to run all evaluations.")
        return
    if shutil.which(args.evaluate_command) is None:
        parser.error(f"command not found: {args.evaluate_command}")

    if args.baseline_metrics:
        baseline_path = args.baseline_metrics
    else:
        subprocess.run(base, check=True)
        baseline_path = newest_metrics(args.evaluation_root, f"Preds_{evaluated_task}")
    baseline = load_macro_dice(baseline_path)

    rows = []
    for kind, index, command in commands:
        subprocess.run(command, check=True)
        tag = prediction_tag(evaluated_task, kind[:3], index, args.seed)
        metrics_path = newest_metrics(args.evaluation_root, tag)
        ablated = load_macro_dice(metrics_path)
        rows.append(
            {
                "kind": kind,
                "block_index": index,
                "baseline_dice": baseline,
                "ablated_dice": ablated,
                "dice_reduction": baseline - ablated,
                "metrics_path": str(metrics_path),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"block_ablation_{evaluated_task}.csv"
    json_path = args.output_dir / f"block_ablation_{evaluated_task}.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
