# Where and How to Adapt

Research code for studying depth-constrained and scale-aware adaptation in
continual 3D medical image segmentation. The implementation extends
[Lifelong nnU-Net](https://github.com/MECLabTUDA/Lifelong-nnUNet) and nnU-Net
v1 with:

- explicit encoder, bottleneck, and decoder freezing policies;
- static block-specific learning rates calibrated from initial relative
  gradient norms;
- a machine-readable registry for every paper training configuration;
- reproducible preprocessing, evaluation, displacement, bootstrap, and
  activation-ablation utilities.

The repository contains **no medical images, patient identifiers, model
checkpoints, or per-patient predictions**.

## Repository status

This is research software built on the Lifelong nnU-Net codebase. The
paper-facing interface lives in [`reproducibility/`](reproducibility/); use it
instead of constructing long trainer commands manually. Expensive operations
are dry runs unless `--execute` is supplied.

## Quick start

The recorded experiments use Python 3.9 and nnU-Net v1. GPU/CUDA installation
is platform-specific; install the matching PyTorch build before or while
creating the environment.

```bash
conda env create -f environment.yml
conda activate depth-aware-cl
```

Configure local storage. The default places generated artifacts in the ignored
`data/` directory inside the clone:

```bash
source reproducibility/config.example.env
```

Validate nnU-Net-formatted raw data without changing it:

```bash
python reproducibility/preprocess.py
```

Inspect the registered experiments and render one command:

```bash
python reproducibility/run_experiment.py --list
python reproducibility/run_experiment.py ecpc_middle_gradient_normalized
```

After checking the rendered command, train and evaluate:

```bash
python reproducibility/run_experiment.py ecpc_middle_gradient_normalized \
  --stage train --execute
python reproducibility/run_experiment.py ecpc_middle_gradient_normalized \
  --stage evaluate --execute
```

## Data preparation

Reproduction starts from de-identified, nnU-Net-formatted NIfTI datasets. The
institution-specific export, conversion, and de-identification code is not
available and is not implied to be part of this release.

```text
data/nnUNet_raw_base/nnUNet_raw_data/TaskXXX_Name/
├── dataset.json
├── imagesTr/CASE_0000.nii.gz
├── imagesTr/CASE_0001.nii.gz
├── labelsTr/CASE.nii.gz
├── imagesTs/CASE_0000.nii.gz
└── labelsTs/CASE.nii.gz
```

The paper uses two input channels. ECPC-IDS uses PET and CT. For the
single-modality MRI tasks, the supplied nnU-Net representation must use the
same two-channel convention as the checkpoints. See [data documentation](docs/DATA.md).

To validate and then run standard nnU-Net planning/preprocessing:

```bash
python reproducibility/preprocess.py
python reproducibility/preprocess.py --execute
```

## Experiments

[`reproducibility/experiments.json`](reproducibility/experiments.json) is the
single source of truth for task order, trainable blocks, plans, learning rates,
seeds, and baseline hyperparameters.

| Group | Contents |
|---|---|
| `single_task` | Independent UMD, ECPC-IDS, and UT-EndoMRI models |
| `fixed_lr_depth` | Frozen, bottleneck, progressive nested regions, and outer-block diagnostic |
| `scale_control` | Gradient-normalized extents and approximate RMS-targeted broad run |
| `cross_transition` | UMD to UT-EndoMRI bottleneck/middle/broad replication |
| `seed_replicates` | Additional fixed-LR adaptation seeds |
| `continual_baselines` | Sequential, rehearsal, LwF, and EWC |

Render a complete group without starting it:

```bash
python reproducibility/run_experiment.py --group scale_control
```

The joint MultiTalent baseline uses a separate nnU-Net v2 environment; see
[`reproducibility/MULTITALENT.md`](reproducibility/MULTITALENT.md).

## Analysis

Measure aggregate and per-block displacement between trusted checkpoints:

```bash
python reproducibility/checkpoint_displacement.py \
  --initial /path/to/pre_adaptation.model \
  --final /path/to/post_adaptation.model \
  --blocks E4 E5 B D5 D4 \
  --output outputs/middle_displacement.csv
```

Compute held-out-case bootstrap intervals:

```bash
python reproducibility/bootstrap_metrics.py \
  --baseline /path/to/umd_before/val_metrics_eval.json \
  --retained /path/to/umd_after/val_metrics_eval.json \
  --new-task /path/to/ecpc_after/val_metrics_eval.json \
  --output outputs/middle_bootstrap.json
```

Render the complete activation-ablation command set without executing it:

```bash
python reproducibility/block_ablation.py \
  --checkpoint-path /path/to/model_final_checkpoint.model \
  --evaluation-root "$EVALUATION_FOLDER" \
  --raw-task-root "$nnUNet_raw_data_base/nnUNet_raw_data" \
  --output-dir outputs/block_ablation \
  --trained-on 666 555 --use-model 666 555 --use-head 666 \
  --evaluate-on 666 \
  --trainer nnUNetTrainerSequentialChannelPreserve \
  --plans nnUNetPlansv2.1_unfreeze_E456D45
```

Add `--execute` only after verifying paths and rendered commands. Activation
ablation measures functional reliance at inference; it is not evidence that
updating a block causes forgetting.

## Reproducibility boundaries

- `PAPER_INIT_CHECKPOINT` must identify the common pre-adaptation Task-1
  checkpoint used by controlled two-task experiments.
- Case bootstrap intervals quantify finite evaluation-cohort uncertainty, not
  training-seed variability.
- Gradient calibration is performed once on initial batches. The resulting
  block multipliers remain fixed while the common polynomial schedule decays.
- Approximate RMS matching is assessed at the final checkpoint; it is not
  enforced online.
- Dataset conversion and access remain governed by the original dataset
  providers and institutional approvals.

See the complete [method documentation](docs/METHOD.md),
[experiment guide](docs/EXPERIMENTS.md), and
[reported-result schema](docs/RESULTS.md). The exact recorded software stack is
listed in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Development checks

The lightweight checks do not require datasets or a GPU:

```bash
python -m unittest discover -s reproducibility/tests -v
python reproducibility/run_experiment.py --list > /dev/null
python -m compileall -q reproducibility
```

Full upstream tests may require the complete nnU-Net environment and a CUDA
device.

## Upstream attribution and license

This repository is derived from Lifelong nnU-Net and includes its original
continual-learning implementations. Please cite both the associated study and
the upstream framework when using the software. The code is distributed under
the [Apache License 2.0](LICENSE).
