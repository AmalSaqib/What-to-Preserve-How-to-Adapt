# Paper reproduction interface

This directory is the stable entry point for the experiments and analyses. It
wraps the modified Lifelong nnU-Net implementation and keeps machine paths out
of commands and source files.

## Files

| File | Purpose |
|---|---|
| `config.example.env` | Portable nnU-Net data, results, evaluation, and initialization paths |
| `experiments.json` | Canonical dataset and experiment registry |
| `preprocess.py` | Read-only dataset validation and opt-in nnU-Net preprocessing |
| `run_experiment.py` | Dry-run-by-default training and evaluation command renderer |
| `checkpoint_displacement.py` | Aggregate and per-block checkpoint movement |
| `bootstrap_metrics.py` | Paired old-task and new-task case-bootstrap intervals |
| `block_ablation.py` | Dry-run-by-default full encoder/decoder activation ablation |
| `reported_results.csv` | Patient-anonymous aggregate reference results |
| `reported_seed_summary.csv` | Aggregate fixed-LR seed summary |
| `run_*sh` | Sequential launchers for long single-GPU experiment groups |
| `MULTITALENT.md` | Joint baseline in the separate nnU-Net v2 environment |

## Minimal workflow

```bash
source reproducibility/config.example.env
python reproducibility/preprocess.py
python reproducibility/run_experiment.py --list
python reproducibility/run_experiment.py ecpc_middle_gradient_normalized
```

The last command prints training and evaluation commands. It does not execute
them. After inspecting the output:

```bash
python reproducibility/run_experiment.py ecpc_middle_gradient_normalized \
  --stage train --execute
python reproducibility/run_experiment.py ecpc_middle_gradient_normalized \
  --stage evaluate --execute
```

## What is reproducible

- independent single-task training;
- fixed-LR progressive unfreezing;
- bottleneck, middle, broad, and full scale-control conditions;
- UMD to UT-EndoMRI replication;
- Sequential, Rehearsal, LwF, and EWC baselines;
- additional fixed-LR seeds;
- checkpoint displacement and per-block allocation;
- case-level bootstrap intervals;
- inference-time block activation ablation.

The original conversion from source clinical exports to de-identified NIfTI
is not available. Reproduction therefore starts from nnU-Net-formatted raw or
compatible preprocessed data. See [`../docs/DATA.md`](../docs/DATA.md).

## Safety and interpretation

- Training, preprocessing, and ablation are opt-in with `--execute`.
- Do not analyze untrusted legacy checkpoint files; PyTorch loads them through
  pickle-compatible nnU-Net serialization.
- Gradient multipliers are calibrated once and remain static.
- Case bootstrap does not replace independent training seeds.
- Activation ablation measures functional reliance, not causal importance for
  continual adaptation.

For exact method definitions and reporting rules, see
[`../docs/METHOD.md`](../docs/METHOD.md) and
[`../docs/RESULTS.md`](../docs/RESULTS.md).
