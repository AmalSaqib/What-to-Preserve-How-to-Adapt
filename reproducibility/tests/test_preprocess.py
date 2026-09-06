import json
import tempfile
import unittest
from pathlib import Path

from reproducibility.preprocess import case_name, validate_task


class PreprocessValidationTests(unittest.TestCase):
    def test_case_name_handles_nifti_suffix(self):
        self.assertEqual(case_name("./imagesTr/CASE_001.nii.gz"), "CASE_001")

    def test_minimal_two_channel_task_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            for folder in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
                (task / folder).mkdir()
            dataset = {
                "modality": {"0": "MRI", "1": "zero"},
                "labels": {"0": "background", "1": "target"},
                "numTraining": 1,
                "numTest": 1,
                "training": [
                    {"image": "./imagesTr/TRAIN.nii.gz", "label": "./labelsTr/TRAIN.nii.gz"}
                ],
                "test": ["./imagesTs/TEST.nii.gz"],
            }
            (task / "dataset.json").write_text(json.dumps(dataset))
            for path in (
                task / "imagesTr/TRAIN_0000.nii.gz",
                task / "imagesTr/TRAIN_0001.nii.gz",
                task / "labelsTr/TRAIN.nii.gz",
                task / "imagesTs/TEST_0000.nii.gz",
                task / "imagesTs/TEST_0001.nii.gz",
                task / "labelsTs/TEST.nii.gz",
            ):
                path.touch()
            self.assertEqual(validate_task(task, require_test_labels=True), [])

    def test_missing_modality_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            for folder in ("imagesTr", "labelsTr", "imagesTs"):
                (task / folder).mkdir()
            dataset = {
                "modality": {"0": "MRI", "1": "zero"},
                "labels": {"0": "background"},
                "numTraining": 0,
                "numTest": 1,
                "training": [],
                "test": ["./imagesTs/TEST.nii.gz"],
            }
            (task / "dataset.json").write_text(json.dumps(dataset))
            (task / "imagesTs/TEST_0000.nii.gz").touch()
            errors = validate_task(task)
            self.assertTrue(any("TEST_0001.nii.gz" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
