import json
import unittest
from pathlib import Path

from corrected_pipeline.config import (
    ConfigError,
    STAGE2_BANGLA_SOURCES,
    STAGE2_MDISTILBERT_REVISION,
    STAGE2_REPORTABLE_SEEDS,
    STAGE2_XLMR_REVISION,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG_NAMES = {
    "pilot_bn_mdistilbert_single_seed42.json",
    "pilot_bn_xlmr_single_seed42.json",
}
STAGE2_CONFIG_NAMES = {
    f"stage2_reportable_{language}_{model}_seed{seed}.json"
    for language in ("en", "bn")
    for model in ("mdistilbert", "xlmr")
    for seed in STAGE2_REPORTABLE_SEEDS
}


def read_config(name):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


class ConfigTests(unittest.TestCase):
    def test_all_corrected_configs_are_valid(self):
        paths = list((ROOT / "configs").glob("corrected_*.json"))
        paths += list((ROOT / "configs").glob("pilot_*.json"))
        paths += list((ROOT / "configs").glob("stage2_reportable_*.json"))
        names = {path.name for path in paths}
        self.assertTrue(PILOT_CONFIG_NAMES.issubset(names))
        self.assertEqual({name for name in names if name.startswith("stage2_")}, STAGE2_CONFIG_NAMES)
        for path in sorted(paths):
            with self.subTest(path=path.name):
                validate_config(json.loads(path.read_text(encoding="utf-8")))

    def test_valid_pilot_configs_pass(self):
        for name in sorted(PILOT_CONFIG_NAMES):
            with self.subTest(config=name):
                validate_config(read_config(name))

    def test_unknown_run_kind_is_rejected(self):
        config = read_config("pilot_bn_mdistilbert_single_seed42.json")
        config["run_kind"] = "unknown"
        with self.assertRaisesRegex(ConfigError, "smoke.*pilot.*full"):
            validate_config(config)

    def test_test_selection_is_rejected(self):
        config = read_config("corrected_smoke_mdistilbert_multitask.json")
        config["training"]["selection_metric"] = "test_macro_f1"
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_silent_proxy_labels_are_rejected(self):
        config = read_config("corrected_smoke_mdistilbert_multitask.json")
        config["auxiliary_labels"]["proxy_auxiliary_labels"] = False
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_legacy_scalar_dataset_hash_is_rejected(self):
        config = read_config("corrected_smoke_mdistilbert_multitask.json")
        config["dataset"]["hashes"]["en_train"] = "0" * 64
        with self.assertRaisesRegex(ConfigError, "must be an object"):
            validate_config(config)

    def test_stage1b_pilot_cannot_add_test_split(self):
        config = read_config("pilot_bn_mdistilbert_single_seed42.json")
        config["dataset"]["paths"]["bn_test"] = "Phase 2/Bangla data/bn_test.csv"
        config["dataset"]["hashes"]["bn_test"] = {"canonical_sha256": "0" * 64}
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_stage1b_pilot_cannot_enable_test_evaluation(self):
        config = read_config("pilot_bn_xlmr_single_seed42.json")
        config["execution"]["evaluate_test"] = True
        with self.assertRaisesRegex(ConfigError, "evaluate_test=false"):
            validate_config(config)

    def test_stage1b_pilot_incorrect_result_status_is_rejected(self):
        config = read_config("pilot_bn_mdistilbert_single_seed42.json")
        config["result_status"] = "CORRECTED FULL BASELINE — NEW RESULT"
        with self.assertRaisesRegex(ConfigError, "exact non-final status"):
            validate_config(config)

    def test_only_approved_stage2_seeds_are_accepted(self):
        for seed in STAGE2_REPORTABLE_SEEDS:
            validate_config(read_config(f"stage2_reportable_en_xlmr_seed{seed}.json"))
        config = read_config("stage2_reportable_en_xlmr_seed42.json")
        config["training"]["random_seed"] = 13
        with self.assertRaisesRegex(ConfigError, "Stage 2 seed"):
            validate_config(config)

    def test_stage2_effective_batch_is_locked_to_64(self):
        for name in STAGE2_CONFIG_NAMES:
            training = read_config(name)["training"]
            self.assertEqual(training["effective_batch_size"], 64)
            self.assertEqual(training["batch_size"] * training["gradient_accumulation"], 64)
        config = read_config("stage2_reportable_en_mdistilbert_seed42.json")
        config["training"]["gradient_accumulation"] = 1
        with self.assertRaisesRegex(ConfigError, "effective batch 64"):
            validate_config(config)

    def test_stage2_test_access_is_rejected(self):
        config = read_config("stage2_reportable_bn_xlmr_seed42.json")
        config["execution"]["evaluate_test"] = True
        with self.assertRaisesRegex(ConfigError, "evaluate_test=false"):
            validate_config(config)
        config = read_config("stage2_reportable_bn_xlmr_seed42.json")
        config["dataset"]["paths"]["bn_test"] = "Phase 2/Bangla data/bn_test.csv"
        config["dataset"]["hashes"]["bn_test"] = {"canonical_sha256": "0" * 64}
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_stage2_proxy_labels_are_rejected(self):
        config = read_config("stage2_reportable_en_xlmr_seed42.json")
        config["auxiliary_labels"] = {"mode": "proxy", "proxy_auxiliary_labels": True}
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_stage2_outputs_are_unique(self):
        outputs = [read_config(name)["execution"]["output_directory"] for name in STAGE2_CONFIG_NAMES]
        self.assertEqual(len(outputs), len(set(outputs)))

    def test_stage2_revisions_are_immutable_and_approved(self):
        expected = {
            "mdistilbert_single_task": STAGE2_MDISTILBERT_REVISION,
            "xlmr_single_task": STAGE2_XLMR_REVISION,
        }
        for name in STAGE2_CONFIG_NAMES:
            model = read_config(name)["model"]
            self.assertRegex(model["revision"], r"^[0-9a-f]{40}$")
            self.assertEqual(model["revision"], expected[model["architecture"]])
            self.assertEqual(model["tokenizer_revision"], model["revision"])

    def test_stage2_uses_all_five_sources(self):
        for name in STAGE2_CONFIG_NAMES:
            values = read_config(name)["dataset"]["approved_bangla_sources"]
            self.assertEqual(tuple(values), STAGE2_BANGLA_SOURCES)


if __name__ == "__main__":
    unittest.main()
