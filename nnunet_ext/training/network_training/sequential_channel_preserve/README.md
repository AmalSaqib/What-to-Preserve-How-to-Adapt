# Sequential Channel Preserve trainer

`nnUNetTrainerSequentialChannelPreserve` implements the paper's controlled
adaptation policies on top of Lifelong nnU-Net.

## Training policy

- Task 1 trains normally.
- For later tasks, the complete shared body is frozen first.
- Requested encoder and decoder blocks are then made trainable.
- Task-specific heads remain trainable.
- Freezing uses parameter identity rather than fragile name matching.

The installed entry point is:

```bash
nnUNet_train_sequential_channel_preserve
```

## Block indices

- Encoder `E0` is input-proximal; the final encoder index is bottleneck `B`.
- Decoder `D0` is output-proximal; larger indices move toward the bottleneck.
- Matching downsampling (`td`) and upsampling (`tu`) modules are included with
  their blocks when present.

Example: the paper's middle region `E4-E5-B-D5-D4` is selected with:

```bash
--encoder_unfreeze 4 5 6 --decoder_unfreeze 4 5
```

## Scale-aware mode

`--gradient_lr_control` runs gradient-only calibration before later-task
adaptation. For each block, it estimates the median initial relative gradient
norm and sets a static LR multiplier relative to bottleneck `B`. The common
polynomial decay is preserved throughout training.

Relevant options:

- `--gradient_calibration_batches` (default: 20)
- `--gradient_lr_reference` (default: `B`)
- `--gradient_lr_min_multiplier` (default: 0.1)
- `--gradient_lr_max_multiplier` (default: 10.0)
- `--gradient_displacement_log_interval` (default: 5 epochs)
- `--base_learning_rate`
- `--run_seed`

Calibration writes JSON metadata and periodic per-block displacement logs to
the trainer output directory. Resume restores the original calibration instead
of recalibrating from an adapted model.

## Recommended interface

Do not manually duplicate long trainer commands. Use the validated registry:

```bash
python reproducibility/run_experiment.py --list
python reproducibility/run_experiment.py ecpc_middle_gradient_normalized
```

See [`reproducibility/README.md`](../../../../reproducibility/README.md) and
[`docs/METHOD.md`](../../../../docs/METHOD.md) for full instructions and exact
metric definitions.
