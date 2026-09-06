import os
import unittest
from unittest.mock import patch

from reproducibility.run_experiment import (
    evaluation_commands,
    experiment_by_id,
    load_manifest,
    training_command,
)


class ExperimentRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_manifest()
        cls.defaults = cls.manifest["defaults"]

    def test_ids_are_unique(self):
        ids = [experiment["id"] for experiment in self.manifest["experiments"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_registered_blocks_match_cli_indices(self):
        # load_manifest performs the authoritative validation. This assertion
        # ensures that every channel-preserve run declares human-readable blocks.
        for experiment in self.manifest["experiments"]:
            if experiment["method"] == "channel_preserve" and not experiment.get(
                "full_body_freeze"
            ):
                self.assertTrue(experiment.get("blocks"), experiment["id"])

    def test_gradient_normalized_command_contains_calibration_flags(self):
        experiment = experiment_by_id(
            self.manifest, "ecpc_middle_gradient_normalized"
        )
        with patch.dict(os.environ, {"PAPER_INIT_CHECKPOINT": "/tmp/base"}):
            command = training_command(experiment, self.defaults)
        self.assertIn("--gradient_lr_control", command)
        self.assertEqual(command[command.index("--gradient_calibration_batches") + 1], "20")
        self.assertEqual(command[command.index("--gradient_lr_reference") + 1], "B")

    def test_evaluation_covers_each_task(self):
        experiment = experiment_by_id(self.manifest, "ecpc_e3_gradient_normalized")
        commands = evaluation_commands(
            experiment, self.defaults, self.manifest["datasets"]
        )
        evaluated = [command[command.index("-evaluate_on") + 1] for command in commands]
        self.assertEqual(evaluated, ["666", "555"])

    def test_unknown_experiment_fails_cleanly(self):
        with self.assertRaises(SystemExit):
            experiment_by_id(self.manifest, "does_not_exist")


if __name__ == "__main__":
    unittest.main()
