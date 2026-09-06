# Experiment guide

## Registry-first workflow

Every paper configuration is defined once in
`reproducibility/experiments.json`. The runner validates experiment IDs, task
IDs, methods, and agreement between CLI indices and human-readable blocks
before rendering commands.

```bash
python reproducibility/run_experiment.py --list
python reproducibility/run_experiment.py --group fixed_lr_depth
```

Rendering is safe and does not start a process. Add `--execute` only to the
specific stage you intend to run.

## Common initialization

Controlled two-task experiments must start from the same UMD checkpoint. Set:

```bash
export PAPER_INIT_CHECKPOINT=/absolute/path/to/the/common/fold_0/checkpoint_directory
```

The path is a trainer output directory, not the `.model` file itself. Keep its
checksum in run metadata.

## Fixed-LR progressive sweep

```bash
python reproducibility/run_experiment.py --group fixed_lr_depth --stage train
```

The group includes frozen-body, bottleneck-only, nested `E/D` regions, and an
`E0/D0`-only diagnostic. The diagnostic is not part of the nested progression.

## Scale-aware controls

```bash
python reproducibility/run_experiment.py --group scale_control --stage train
```

The gradient-normalized controls use 20 gradient-only batches, bottleneck as
the reference block, seed 12345, and the same base LR schedule. The broad
RMS-targeted configuration instead uses base LR 0.008; final RMS is measured
after training and is not constrained online.

## Cross-transition replication

```bash
python reproducibility/run_experiment.py --group cross_transition
```

This changes the successor task from ECPC-IDS to UT-EndoMRI but retains UMD as
the first task. It is an additional transition, not a randomized or reversed
task-order experiment.

## Additional seeds

The launcher runs one job at a time for a single-GPU machine:

```bash
bash reproducibility/run_seed_replicates.sh
```

It creates seed-specific aliases of the original plans and refuses to replace
an alias whose content differs. The data split and initialization checkpoint
remain fixed; only adaptation stochasticity changes.

## Evaluation

```bash
python reproducibility/run_experiment.py EXPERIMENT_ID --stage evaluate
python reproducibility/run_experiment.py EXPERIMENT_ID --stage evaluate --execute
```

The runner evaluates every task/head registered for the sequence. Always
inspect the rendered `trained_on`, `use_model`, `use_head`, and `evaluate_on`
arguments before execution.

## Run record

For each completed job, archive:

- experiment ID and full rendered command;
- Git commit and dirty status;
- environment export and GPU information;
- start/end timestamps;
- task split and seed;
- initialization and final checkpoint checksums;
- calibration JSON when applicable;
- evaluation JSON files.

Large artifacts belong in controlled object storage, not Git.
