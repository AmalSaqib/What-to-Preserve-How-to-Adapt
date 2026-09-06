# Result and analysis outputs

## Training outputs

Gradient-normalized runs write:

- `gradient_lr_calibration_initial.json`: immutable initial calibration;
- `gradient_lr_calibration.json`: current calibration record;
- `gradient_lr_displacement.csv`: cumulative per-block diagnostic values;
- standard nnU-Net checkpoints and logs.

On resume, the trainer restores the initial multipliers instead of recalibrating
from a partially adapted checkpoint.

Compact, patient-anonymous summaries of the values reported in the manuscript
are provided in:

- `reproducibility/reported_results.csv`;
- `reproducibility/reported_seed_summary.csv`.

These files contain aggregate metrics only. They are reference outputs, not a
substitute for the checkpoint and per-case analyses that produced them.

## Checkpoint displacement

`checkpoint_displacement.py` reports one aggregate row and one row per selected
block. Task heads are excluded. The output records parameter count, RMS,
relative L2, total L2, maximum absolute displacement, and squared-displacement
share.

Because legacy nnU-Net checkpoints are pickle-based, analyze only trusted
files.

## Bootstrap output

`bootstrap_metrics.py` emits JSON containing:

- pre-adaptation old-task Dice;
- post-adaptation old-task Dice;
- forgetting and its paired case-bootstrap interval;
- new-task Dice and its case-bootstrap interval;
- patient counts, seed, and the explicit uncertainty limitation.

## Block-ablation output

`block_ablation.py` writes CSV and JSON rows with baseline Dice, ablated Dice,
and Dice reduction for every encoder and decoder block. The command is dry-run
by default because a full sweep launches many inference jobs.

## Reporting rules

- Report retention and new-task performance together.
- Distinguish RMS per parameter from total L2 movement.
- State whether a comparison uses shared or block-specific learning rates.
- Do not use case-bootstrap intervals as training-seed uncertainty.
- Do not call a final-checkpoint RMS comparison an online matched-update run.
- Treat block ablation as functional reliance, not causal training evidence.
