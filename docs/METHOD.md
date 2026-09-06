# Method implementation

## Block convention

The shared 3D nnU-Net body is represented as encoder blocks
`E0, ..., E5`, bottleneck `B`, and decoder blocks `D5, ..., D0`.

- `E0` is closest to the input.
- `B` is the deepest encoder block.
- `D5` is closest to the bottleneck.
- `D0` is closest to the segmentation output.

Decoder CLI indices use this paper-facing convention even though nnU-Net's
internal localization list is ordered in the opposite direction.

## Depth-constrained adaptation

For Task 1, the shared body trains normally. For each later task, the trainer:

1. assembles the task model;
2. freezes every shared-body parameter by object identity;
3. unfreezes the requested encoder/decoder blocks and their transition modules;
4. keeps task-specific heads trainable;
5. builds the optimizer only after the final `requires_grad` policy is known.

`--full_body_freeze` trains heads only. `--encoder_unfreeze` and
`--decoder_unfreeze` select body blocks. Invalid indices raise an error rather
than being ignored.

## Static gradient-normalized learning rates

For a trainable block `b`, the calibration pass estimates the median initial
relative gradient norm over `K` current-task batches:

```text
r_b = median_k( ||g_b^(k)||_2 / (||theta_b||_2 + epsilon) ).
```

With bottleneck `B` as the reference, the static multiplier is

```text
alpha_b = clip(r_B / r_b, alpha_min, alpha_max),
eta_b(t) = alpha_b * eta(t).
```

The calibration batches perform forward/backward passes but no optimizer
steps. Multipliers are written to `gradient_lr_calibration_initial.json`, are
restored when training resumes, and remain fixed. The base learning rate
follows nnU-Net's polynomial decay. This is an initial gradient-scale control,
not an online displacement-matching algorithm.

## Displacement quantities

For selected trainable parameters `S`, the analysis reports:

```text
RMS(S)      = sqrt(sum_i (theta_i^after - theta_i^before)^2 / |S|)
Relative(S) = ||theta_S^after - theta_S^before||_2 / ||theta_S^before||_2
L2(S)       = ||theta_S^after - theta_S^before||_2
q_b         = ||Delta theta_b||_2^2 / sum_c ||Delta theta_c||_2^2
```

RMS is typical movement per selected parameter. L2 is total selected movement.
`q_b` describes how squared displacement is allocated among blocks. These
quantities are not interchangeable when configurations contain different
numbers of trainable parameters.

## Activation ablation

The evaluation code can zero the complete output of one encoder or decoder
block at inference. The reported reduction from unmodified Dice measures
functional reliance. It must not be interpreted as the causal effect of
updating, freezing, or removing that block during training.

## Uncertainty

The bootstrap utility resamples held-out patients with replacement. For
forgetting, the same sampled indices are applied to pre- and post-adaptation
old-task predictions. Percentile 95% intervals summarize evaluation-cohort
uncertainty only. Independently trained seeds are required to measure
optimization variability.
