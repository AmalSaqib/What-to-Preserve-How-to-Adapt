#!/usr/bin/env python3
"""Compute case-bootstrap intervals for Dice retention and acquisition.

The input files are ``val_metrics_eval.json`` outputs produced by
``nnUNet_evaluate2``. These intervals quantify held-out-case uncertainty, not
variation across independently trained models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_cases(path: Path) -> tuple[list[str], np.ndarray]:
    raw = json.loads(path.read_text())
    epoch = raw[next(iter(raw))]
    task = epoch[next(iter(epoch))]
    case_ids = sorted(task)
    labels = sorted({label for case in task.values() for label in case})
    values = np.full((len(case_ids), len(labels)), np.nan, dtype=float)
    for row, case_id in enumerate(case_ids):
        for column, label in enumerate(labels):
            dice = task[case_id].get(label, {}).get("Dice")
            if dice is not None:
                values[row, column] = float(dice)
    return case_ids, values


def macro_dice(values: np.ndarray) -> float:
    means = [
        float(np.mean(values[np.isfinite(values[:, column]), column]))
        for column in range(values.shape[1])
        if np.any(np.isfinite(values[:, column]))
    ]
    if not means:
        raise ValueError("no finite Dice values found")
    return float(np.mean(means))


def sampled_macro_and_validity(
    values: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sampled = values[indices]
    finite = np.isfinite(sampled)
    counts = finite.sum(axis=1)
    valid = np.all(counts > 0, axis=1)
    safe_counts = np.maximum(counts, 1)
    class_means = np.nansum(sampled, axis=1) / safe_counts
    return np.mean(class_means, axis=1), valid


def resampled_macro(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    macros, valid = sampled_macro_and_validity(values, indices)
    return macros[valid]


def interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="old-task metrics before adaptation")
    parser.add_argument("--retained", type=Path, required=True, help="old-task metrics after adaptation")
    parser.add_argument("--new-task", type=Path, required=True, help="new-task metrics after adaptation")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output", type=Path, help="optional JSON output")
    args = parser.parse_args()
    if args.resamples < 100:
        parser.error("--resamples must be at least 100")

    baseline_ids, baseline = load_cases(args.baseline)
    retained_ids, retained = load_cases(args.retained)
    new_ids, new_task = load_cases(args.new_task)
    if baseline_ids != retained_ids:
        parser.error("baseline and retained metrics do not contain the same ordered cases")

    rng = np.random.default_rng(args.seed)
    old_indices = rng.integers(0, len(baseline_ids), size=(args.resamples, len(baseline_ids)))
    baseline_boot, baseline_valid = sampled_macro_and_validity(baseline, old_indices)
    retained_boot, retained_valid = sampled_macro_and_validity(retained, old_indices)
    paired_valid = baseline_valid & retained_valid
    forgetting_boot = baseline_boot[paired_valid] - retained_boot[paired_valid]

    new_indices = rng.integers(0, len(new_ids), size=(args.resamples, len(new_ids)))
    new_boot = resampled_macro(new_task, new_indices)
    result = {
        "bootstrap_unit": "held-out case",
        "resamples_requested": args.resamples,
        "seed": args.seed,
        "retained_task_cases": len(baseline_ids),
        "new_task_cases": len(new_ids),
        "baseline_dice": macro_dice(baseline),
        "retained_dice": macro_dice(retained),
        "forgetting": macro_dice(baseline) - macro_dice(retained),
        "forgetting_95_ci": interval(forgetting_boot),
        "new_task_dice": macro_dice(new_task),
        "new_task_dice_95_ci": interval(new_boot),
        "limitation": "case bootstrap does not measure training-seed variability",
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
