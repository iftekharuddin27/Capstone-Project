import json
import unittest
from copy import deepcopy
from pathlib import Path

from corrected_pipeline.config import ConfigError, validate_config


ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG_NAMES = {
    "pilot_bn_mdistilbert_single_seed42.json",
    "pilot_bn_xlmr_single_seed42.json",
}


class ConfigTests(unittest.TestCase):
    def test_all_corrected_configs_are_valid(self):
        paths = list((ROOT / "configs").glob("corrected_*.json"))
        paths.extend((ROOT / "configs").glob("pilot_*.json"))
        self.assertTrue(PILOT_CONFIG_NAMES.issubset({path.name for path in paths}))
        for path in sorted(paths):
            with self.subTest(path=path.name):
                validate_config(json.loads(path.read_text(encoding="utf-8")))

    def test_valid_pilot_configs_pass(self):
        for name in sorted(PILOT_CONFIG_NAMES):
            with self.subTest(config=name):
                path = ROOT / "configs" / name
                validate_config(json.loads(path.read_text(encoding="utf-8")))

    def test_unknown_run_kind_is_rejected(self):
        path = ROOT / "configs" / "pilot_bn_mdistilbert_single_seed42.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["run_kind"] = "unknown"
        with self.assertRaisesRegex(ConfigError, "smoke.*pilot.*full"):
            validate_config(config)

    def test_test_selection_is_rejected(self):
        path = ROOT / "configs" / "corrected_smoke_mdistilbert_multitask.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["training"]["selection_metric"] = "test_macro_f1"
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_silent_proxy_labels_are_rejected(self):
        path = ROOT / "configs" / "corrected_smoke_mdistilbert_multitask.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["auxiliary_labels"]["proxy_auxiliary_labels"] = False
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_legacy_scalar_dataset_hash_is_rejected(self):
        path = ROOT / "configs" / "corrected_smoke_mdistilbert_multitask.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["dataset"]["hashes"]["en_train"] = "0" * 64
        with self.assertRaisesRegex(ConfigError, "must be an object"):
            validate_config(config)

    def test_stage1b_pilot_cannot_add_test_split(self):
        path = ROOT / "configs" / "pilot_bn_mdistilbert_single_seed42.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["dataset"]["paths"]["bn_test"] = "Phase 2/Bangla data/bn_test.csv"
        config["dataset"]["hashes"]["bn_test"] = {
            "canonical_sha256": "c2ac6c268ad94284be0bff8857f9d2f90e68610075bc0c60dac9ff008496df0d"
        }
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_stage1b_pilot_cannot_enable_test_evaluation(self):
        path = ROOT / "configs" / "pilot_bn_xlmr_single_seed42.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["execution"]["evaluate_test"] = True
        with self.assertRaisesRegex(ConfigError, "evaluate_test=false"):
            validate_config(config)

    def test_stage1b_pilot_incorrect_result_status_is_rejected(self):
        path = ROOT / "configs" / "pilot_bn_mdistilbert_single_seed42.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["result_status"] = "CORRECTED FULL BASELINE — NEW RESULT"
        with self.assertRaisesRegex(ConfigError, "exact non-final status"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
