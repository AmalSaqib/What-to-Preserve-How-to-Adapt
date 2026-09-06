# Recorded software environment

The reported runs used the following core environment:

| Component | Recorded version |
|---|---|
| Python | 3.9.25 |
| PyTorch | 2.5.1+cu121 |
| CUDA runtime exposed to PyTorch | 12.1 |
| cuDNN | 9.1.0 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| scikit-learn | 1.6.1 |
| GPU | NVIDIA Quadro RTX 6000, 24 GB |
| Lifelong nnU-Net base commit | `9f26c566d948b88b35390e1c5cdd3253655bcf67` |

`environment.yml` captures the Python, PyTorch, and CUDA versions while
`requirements.txt` pins the upstream nnU-Net and auxiliary Git revisions.
Exact binary resolution can still vary by operating system and driver.

For every final run, save a local record alongside the checkpoint:

```bash
python --version > run_environment.txt
python -m pip freeze >> run_environment.txt
nvidia-smi >> run_environment.txt
git rev-parse HEAD >> run_environment.txt
git status --short >> run_environment.txt
```

`run_environment.txt` is ignored because it can contain local paths and system
details. Archive it with the controlled run artifacts rather than committing it.
