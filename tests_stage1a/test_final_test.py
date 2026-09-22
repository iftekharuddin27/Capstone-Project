"""Static final-test safety checks; no GPU, checkpoint, or test CSV is read."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from corrected_pipeline.final_test import (
    FROZEN_RUNS, FROZEN_TEST_HASHES, frozen_run, verify_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


class FrozenFinalTestTests(unittest.TestCase):
    def test_six_preselected_configs_remain_validation_locked(self):
        for language in FROZEN_RUNS:
            for seed, (checkpoint_hash, _) in FROZEN_RUNS[language].items():
                with self.subTest(language=language, seed=seed):
                    frozen = frozen_run(language, seed, ROOT)
                    self.assertEqual(len(checkpoint_hash), 64)
                    self.assertEqual(len(frozen["paths"]), 3)
                    self.assertEqual(frozen["config"]["execution"]["evaluate_test"], False)
                    self.assertEqual(frozen["config"]["execution"]["test_locked"], True)
                    self.assertEqual(frozen["hashes"][f"{frozen['prefix']}_test"]["canonical_sha256"],
                                     FROZEN_TEST_HASHES[language])
                    self.assertEqual(frozen["config"]["dataset"]["paths"].keys(),
                                     frozen["config"]["dataset"]["hashes"].keys())

    def test_test_hashes_match_original_audited_manifest(self):
        full = json.loads((ROOT / "configs/corrected_full_xlmr_single_task.json").read_text())
        for language, prefix in (("english", "en"), ("bangla", "bn")):
            with self.subTest(language=language):
                self.assertEqual(full["dataset"]["hashes"][f"{prefix}_test"]["canonical_sha256"],
                                 FROZEN_TEST_HASHES[language])

    def test_reject_unselected_seed_and_changed_model_before_test(self):
        with self.assertRaises(ValueError):
            frozen_run("bangla", 13, ROOT)
        valid = frozen_run("english", 42, ROOT)["config"]
        tampered = deepcopy(valid)
        tampered["model"]["architecture"] = "mdistilbert_single_task"
        with patch("corrected_pipeline.final_test.load_config", return_value=tampered):
            with self.assertRaises(ValueError):
                frozen_run("english", 42, ROOT)

    def test_hash_mismatch_blocks_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "best_checkpoint_cpu.pt"
            checkpoint.write_bytes(b"incorrect checkpoint content")
            with self.assertRaisesRegex(ValueError, "Wrong checkpoint SHA-256"):
                verify_checkpoint(checkpoint, FROZEN_RUNS["bangla"][42][0])


if __name__ == "__main__":
    unittest.main()
