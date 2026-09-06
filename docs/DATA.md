# Data contract

## What is distributed

No patient data are included. Users must obtain each public dataset from its
provider and comply with its license and data-use conditions.

The reproducibility pipeline begins after conversion to de-identified nnU-Net
v1 NIfTI folders. Original clinical-export conversion and de-identification
were dataset-specific and are not available in this repository.

## Registered tasks

| ID | Dataset | Inputs | Targets | Train | Held out |
|---:|---|---|---|---:|---:|
| 666 | UMD | T2 MRI represented with two input channels | uterine wall, uterine cavity, myoma, nabothian cyst | 214 | 52 |
| 555 | ECPC-IDS | PET and CT | tumor | 108 | 26 |
| 778 | UT-EndoMRI | T1FS MRI represented with two input channels | uterus, ovaries, endometrioma, cyst | 83 | 20 |

Counts are checked against each task's `dataset.json`. They document the paper
configuration rather than granting permission to redistribute cases.

## Required layout

Set `nnUNet_raw_data_base` to a directory containing `nnUNet_raw_data/`:

```text
nnUNet_raw_data_base/
└── nnUNet_raw_data/
    ├── Task555_ECPC/
    ├── Task666_UMD/
    └── Task778_EndoMRI_T1FS/
```

Each task must include `dataset.json`, `imagesTr`, `labelsTr`, `imagesTs`, and
held-out reference labels for evaluation. Image filenames follow nnU-Net v1's
`CASE_0000.nii.gz` convention.

## Two-channel compatibility

All three tasks share one backbone, so their planned input channel count and
ordering must match. ECPC-IDS uses PET and CT. The MRI task representation must
match the channel convention recorded in the corresponding `dataset.json` and
pre-adaptation checkpoint. Do not change this convention after training: input
shape or normalization changes make the checkpoints incompatible.

## Validation and preprocessing

`preprocess.py` first performs read-only checks of task discovery,
`dataset.json`, declared counts, modality files, and training labels:

```bash
python reproducibility/preprocess.py
```

Only the explicit execution form invokes nnU-Net preprocessing:

```bash
python reproducibility/preprocess.py --execute
```

Preserve `plans.pkl`, `dataset_properties.pkl`, `splits_final.pkl`, the exact
environment, and preprocessing logs with each run. These generated files and
all NIfTI data are excluded by `.gitignore`.
