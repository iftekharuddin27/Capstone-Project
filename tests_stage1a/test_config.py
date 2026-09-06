import json
import unittest
from copy import deepcopy
from pathlib import Path

from corrected_pipeline.config import ConfigError, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_all_corrected_configs_are_valid(self):
        for path in sorted((ROOT / "configs").glob("corrected_*.json")):
            with self.subTest(path=path.name):
                validate_config(json.loads(path.read_text(encoding="utf-8")))

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


if __name__ == "__main__":
    unittest.main()
