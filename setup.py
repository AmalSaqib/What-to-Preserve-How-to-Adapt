"""Package configuration for the Lifelong nnU-Net extensions."""

from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).resolve().parent


setup(
    name="nnunet_ext",
    version="0.1.0",
    description="Continual nnU-Net with depth-constrained and scale-aware adaptation",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["nnunet_ext", "nnunet_ext.*"]),
    python_requires=">=3.9,<3.11",
    install_requires=["tqdm"],
    license="Apache-2.0",
    project_urls={
        "Source code": "https://github.com/AmalSaqib/What-to-Preserve-How-to-Adapt",
        "Upstream framework": "https://github.com/MECLabTUDA/Lifelong-nnUNet",
    },
    entry_points={
        "console_scripts": [
            "nnUNet_dataset_label_mapping=nnunet_ext.experiment_planning.dataset_label_mapping:main",
            "nnUNet_delete_tasks=nnunet_ext.scripts.delete_specified_task:main",
            "nnUNet_update_checkpoints=nnunet_ext.scripts.update_checkpoints:main",
            "nnUNet_update_checkpoints_all=nnunet_ext.scripts.update_checkpoints:main_all",
            "nnUNet_train_multihead=nnunet_ext.run.run_training:main_multihead",
            "nnUNet_train_sequential=nnunet_ext.run.run_training:main_sequential",
            "nnUNet_train_rehearsal=nnunet_ext.run.run_training:main_rehearsal",
            "nnUNet_train_ewc=nnunet_ext.run.run_training:main_ewc",
            "nnUNet_train_rw=nnunet_ext.run.run_training:main_rw",
            "nnUNet_train_lwf=nnunet_ext.run.run_training:main_lwf",
            "nnUNet_train_mib=nnunet_ext.run.run_training:main_mib",
            "nnUNet_train_plop=nnunet_ext.run.run_training:main_plop",
            "nnUNet_train_pod=nnunet_ext.run.run_training:main_pod",
            "nnUNet_train_body_froz=nnunet_ext.run.run_training:main_frozen_body_seq",
            "nnUNet_train_ewc_ln=nnunet_ext.run.run_training:main_ewc_ln",
            "nnUNet_train_ewc_unet=nnunet_ext.run.run_training:main_ewc_unet",
            "nnUNet_train_ewc_vit=nnunet_ext.run.run_training:main_ewc_vit",
            "nnUNet_train_froz_ewc=nnunet_ext.run.run_training:main_froz_ewc",
            "nnUNet_train_frozen_nonln=nnunet_ext.run.run_training:main_frozen_nonln",
            "nnUNet_train_frozen_unet=nnunet_ext.run.run_training:main_frozen_unet",
            "nnUNet_train_frozen_vit=nnunet_ext.run.run_training:main_frozen_vit",
            "nnUNet_train_sequential_channel_preserve=nnunet_ext.run.run_training:main_sequential_channel_preserve",
            "nnUNet_parameter_search=nnunet_ext.run.run_param_search:main",
            "nnUNet_join_datasets=nnunet_ext.scripts.join_datasets:main",
            "nnUNet_inference=nnunet_ext.run.run_inference:main",
            "nnUNet_evaluate=nnunet_ext.run.run_evaluation:main",
            "nnUNet_evaluate2=nnunet_ext.run.run_evaluation:run_evaluation2",
        ]
    },
    keywords=[
        "continual learning",
        "medical image segmentation",
        "nnU-Net",
        "catastrophic forgetting",
    ],
)
