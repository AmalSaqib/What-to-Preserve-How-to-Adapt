# Joint MultiTalent baseline

The paper's joint-learning baseline is stored under `MultiTalent_framework/` and uses the nnU-Net v2-style MultiTalent codebase. It must not share `nnUNet_preprocessed` with the Lifelong-nnUNet v1 pipeline.

## Recorded paper configuration

- combined dataset: `Dataset999_MultiTalent`;
- component datasets: 555 ECPC, 666 UMD, and 778 EndoMRI T1FS;
- trainer: `MultiTalent_trainer`;
- configuration: `3d_fullres`;
- plan: `nnUNetResEncUNetLPlansIso1x1x1`;
- fold: 0;
- epochs recorded in the checkpoint metadata: 300.

Do not use `Dataset901_Gyne_MT` for the reported T1FS comparison: that older artifact combines tasks 555, 666, and 777 (T2 EndoMRI), while the paper's selected EndoMRI task is 778 (T1FS).

## Environment and paths

```bash
cd MultiTalent_framework/MultiTalent
pip install -e .

export nnUNet_raw=/path/to/multitalent-data/nnUNet_raw
export nnUNet_preprocessed=/path/to/multitalent-data/nnUNet_preprocessed
export nnUNet_results=/path/to/multitalent-data/nnUNet_results
```

## Preparation and training

If `Dataset999_MultiTalent` has not been prepared:

```bash
prepare_MT_training MultiTalent 999 -d 555 666 778 \
  -p nnUNetResEncUNetLPlansIso1x1x1.json \
  --verify_dataset_integrity
```

Train fold 0:

```bash
multitalent_train 999 3d_fullres 0 \
  -p nnUNetResEncUNetLPlansIso1x1x1 \
  -tr MultiTalent_trainer
```

The exact training artifact currently resides at:

```text
MultiTalent_framework/datasets/nnUNet_results/
└── Dataset999_MultiTalent/
    └── MultiTalent_trainer__nnUNetResEncUNetLPlansIso1x1x1__3d_fullres/
        └── fold_0/
```

Inference should use `multitalent_predict_from_modelfolder` with the dataset-specific target head. Preserve prediction commands and metric aggregation in run metadata; unlike the Lifelong-nnUNet evaluations, this framework uses nnU-Net v2 dataset and label conventions.
