# Utility scripts

This directory contains the original Lifelong-nnUNet maintenance utilities.
Paper experiments are intentionally kept out of this package directory. Use
the validated experiment registry instead:

```bash
source reproducibility/config.example.env
python reproducibility/run_experiment.py --list
```

The canonical configurations are stored in
[`../../reproducibility/experiments.json`](../../reproducibility/experiments.json).
Machine-specific historical launchers are not distributed because they embed
local paths and can silently select obsolete block-index conventions.
