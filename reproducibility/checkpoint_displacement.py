#!/usr/bin/env python3
"""Measure aggregate and per-block displacement between nnU-Net checkpoints.

Only shared-body tensors are compared; task-specific heads are excluded. Legacy
nnU-Net checkpoints use Python pickle, so only pass checkpoints that you trust.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch


TensorMap = Dict[str, torch.Tensor]


def load_body(path: Path) -> TensorMap:
    """Load floating-point body tensors from a trusted nnU-Net checkpoint."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")

    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError(f"{path} does not contain a state dictionary")

    body = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        for prefix in ("module.", "network.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if not key.startswith("body.") or not torch.is_tensor(value):
            continue
        body[key.removeprefix("body.")] = value.detach().cpu().float()
    if not body:
        raise ValueError(f"{path} contains no tensors with the 'body.' prefix")
    return body


def architecture_depths(keys: Iterable[str]) -> tuple[int, int]:
    """Infer encoder and decoder block counts from state-dictionary keys."""
    encoder = []
    decoder = []
    for key in keys:
        parts = key.split(".")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        if parts[0] == "conv_blocks_context":
            encoder.append(int(parts[1]))
        elif parts[0] == "conv_blocks_localization":
            decoder.append(int(parts[1]))
    if not encoder or not decoder:
        raise ValueError("could not infer encoder/decoder depth from checkpoint keys")
    return max(encoder) + 1, max(decoder) + 1


def block_for_key(key: str, encoder_count: int, decoder_count: int) -> str:
    """Map an nnU-Net body tensor to the paper-facing E/B/D convention."""
    parts = key.split(".")
    if len(parts) < 2 or not parts[1].isdigit():
        return "other"
    index = int(parts[1])
    if parts[0] == "conv_blocks_context":
        return "B" if index == encoder_count - 1 else f"E{index}"
    if parts[0] == "td":
        return f"E{index}"
    if parts[0] in {"conv_blocks_localization", "tu"}:
        return f"D{decoder_count - 1 - index}"
    return "other"


def displacement(
    initial: Mapping[str, torch.Tensor],
    final: Mapping[str, torch.Tensor],
    selected_blocks: Optional[set[str]] = None,
) -> dict[str, float]:
    """Return count, RMS, relative L2, L2, and maximum absolute displacement."""
    encoder_count, decoder_count = architecture_depths(initial)
    count = 0
    initial_sq = 0.0
    delta_sq = 0.0
    max_abs = 0.0
    for key, before in initial.items():
        if key not in final or final[key].shape != before.shape:
            raise ValueError(f"missing or incompatible tensor in final checkpoint: {key}")
        block = block_for_key(key, encoder_count, decoder_count)
        if selected_blocks is not None and block not in selected_blocks:
            continue
        delta = final[key] - before
        count += before.numel()
        initial_sq += float(torch.sum(before * before).item())
        delta_sq += float(torch.sum(delta * delta).item())
        max_abs = max(max_abs, float(torch.max(torch.abs(delta)).item()))

    l2_delta = math.sqrt(delta_sq)
    return {
        "parameters": count,
        "rms_delta": math.sqrt(delta_sq / count) if count else 0.0,
        "relative_l2_percent": (
            100.0 * l2_delta / (math.sqrt(initial_sq) + 1e-12) if count else 0.0
        ),
        "l2_delta": l2_delta,
        "max_abs_delta": max_abs,
    }


def available_blocks(state: Mapping[str, torch.Tensor]) -> list[str]:
    encoder_count, decoder_count = architecture_depths(state)
    blocks = {
        block_for_key(key, encoder_count, decoder_count)
        for key in state
    }
    order = [*(f"E{i}" for i in range(encoder_count - 1)), "B"]
    order.extend(f"D{i}" for i in reversed(range(decoder_count)))
    return [block for block in order if block in blocks]


def rows_for_comparison(
    initial: Mapping[str, torch.Tensor],
    final: Mapping[str, torch.Tensor],
    selected: list[str],
) -> list[dict[str, object]]:
    selected_set = set(selected)
    aggregate = displacement(initial, final, selected_set)
    rows: list[dict[str, object]] = [
        {
            "scope": "aggregate",
            "block": "all",
            **aggregate,
            "squared_displacement_share_percent": 100.0,
        }
    ]
    denominator = max(aggregate["l2_delta"] ** 2, 1e-30)
    for block in selected:
        stats = displacement(initial, final, {block})
        rows.append(
            {
                "scope": "block",
                "block": block,
                **stats,
                "squared_displacement_share_percent": (
                    100.0 * stats["l2_delta"] ** 2 / denominator
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True, help="pre-adaptation checkpoint")
    parser.add_argument("--final", type=Path, required=True, help="post-adaptation checkpoint")
    parser.add_argument(
        "--blocks",
        nargs="+",
        help="trainable paper-facing blocks, for example E4 E5 B D5 D4; default: all",
    )
    parser.add_argument("--output", type=Path, help="optional .csv or .json output path")
    args = parser.parse_args()

    initial = load_body(args.initial)
    final = load_body(args.final)
    known = available_blocks(initial)
    selected = args.blocks or known
    unknown = sorted(set(selected) - set(known))
    if unknown:
        parser.error(f"unknown blocks {unknown}; available blocks: {known}")

    rows = rows_for_comparison(initial, final, selected)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".json":
            args.output.write_text(json.dumps(rows, indent=2) + "\n")
        elif args.output.suffix.lower() == ".csv":
            with args.output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            parser.error("--output must end in .csv or .json")

    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
