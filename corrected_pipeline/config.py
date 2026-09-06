"""Configuration loading and validation with no ML-library dependencies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_SEEDS = (13, 21, 42, 87, 101)
ARCHITECTURES = {
    "xlmr_single_task",
    "mdistilbert_single_task",
    "xlmr_multi_task",
    "mdistilbert_multi_task",
}
SELECTION_METRIC = "validation_macro_f1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ConfigError(ValueError):
    """Raised when an experiment configuration is unsafe or incomplete."""


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required field: {context}.{key}")
    return mapping[key]


def _positive(value: Any, field: str, allow_zero: bool = False) -> None:
    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    valid = valid and (value >= 0 if allow_zero else value > 0)
    if not valid:
        comparator = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{field} must be a {comparator} number; got {value!r}")


def validate_config(config: Mapping[str, Any]) -> None:
    if _require(config, "schema_version", "config") != 1:
        raise ConfigError("config.schema_version must be 1")

    run_kind = _require(config, "run_kind", "config")
    if run_kind not in {"smoke", "pilot", "full"}:
        raise ConfigError("config.run_kind must be 'smoke', 'pilot', or 'full'")

    status = _require(config, "result_status", "config")
    if run_kind == "smoke" and status != "SMOKE TEST — NOT A REPORTED RESULT":
        raise ConfigError("Smoke configurations must use the exact non-reportable status label")
    if run_kind == "pilot" and status != "PILOT — NOT FINAL TEST RESULT":
        raise ConfigError("Pilot configurations must use the exact non-final status label")

    model = _require(config, "model", "config")
    architecture = _require(model, "architecture", "config.model")
    if architecture not in ARCHITECTURES:
        raise ConfigError(f"Unsupported architecture: {architecture!r}")
    for field in ("name", "revision", "tokenizer_name", "tokenizer_revision"):
        value = _require(model, field, "config.model")
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"config.model.{field} must be a non-empty string")

    dataset = _require(config, "dataset", "config")
    if _require(dataset, "class_2_semantic_status", "config.dataset") != "pending_supervisor_confirmation":
        raise ConfigError("class 2 must remain marked pending_supervisor_confirmation")
    languages = _require(dataset, "languages", "config.dataset")
    if not languages or any(language not in {"english", "bangla"} for language in languages):
        raise ConfigError("config.dataset.languages must contain english and/or bangla")
    if run_kind == "pilot" and languages != ["bangla"]:
        raise ConfigError("Stage 1B pilot configurations must contain only the Bangla language")
    paths = _require(dataset, "paths", "config.dataset")
    hashes = _require(dataset, "hashes", "config.dataset")
    required_splits = (
        {"bn_train", "bn_validation"}
        if run_kind == "pilot"
        else {
            "en_train", "en_validation", "en_test",
            "bn_train", "bn_validation", "bn_test",
        }
    )
    if set(paths) != required_splits:
        raise ConfigError(f"config.dataset.paths must have exactly {sorted(required_splits)}")
    if set(hashes) != required_splits:
        raise ConfigError(f"config.dataset.hashes must have exactly {sorted(required_splits)}")
    if run_kind == "pilot":
        if dataset.get("path_base") != "repository_root":
            raise ConfigError("Pilot dataset.path_base must be 'repository_root'")
        audited_paths = {
            "bn_train": "Phase 2/Bangla data/bn_train.csv",
            "bn_validation": "Phase 2/Bangla data/bn_val.csv",
        }
        if paths != audited_paths:
            raise ConfigError("Stage 1B pilots must use the audited repository-relative Bangla paths")
        for split, value in paths.items():
            if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
                raise ConfigError(f"config.dataset.paths.{split} must be repository-relative")
    for split, hash_manifest in hashes.items():
        if not isinstance(hash_manifest, Mapping):
            raise ConfigError(f"config.dataset.hashes.{split} must be an object")
        if set(hash_manifest).difference({"canonical_sha256", "raw_sha256"}):
            raise ConfigError(
                f"config.dataset.hashes.{split} may contain only canonical_sha256 and optional raw_sha256"
            )
        canonical = hash_manifest.get("canonical_sha256")
        if not isinstance(canonical, str) or not _SHA256.fullmatch(canonical.lower()):
            raise ConfigError(f"config.dataset.hashes.{split}.canonical_sha256 is not a SHA-256 digest")
        raw = hash_manifest.get("raw_sha256")
        if raw is not None and (not isinstance(raw, str) or not _SHA256.fullmatch(raw.lower())):
            raise ConfigError(f"config.dataset.hashes.{split}.raw_sha256 is not a SHA-256 digest")

    training = _require(config, "training", "config")
    required_training = (
        "maximum_sequence_length", "learning_rate", "batch_size",
        "gradient_accumulation", "epochs", "early_stopping_patience",
        "weight_decay", "warmup_ratio", "dropout", "gradient_clipping",
        "random_seed", "loss_weights", "class_weights", "selection_metric",
    )
    for field in required_training:
        _require(training, field, "config.training")
    for field in ("maximum_sequence_length", "learning_rate", "batch_size", "gradient_accumulation", "epochs", "dropout", "gradient_clipping"):
        _positive(training[field], f"config.training.{field}")
    for field in ("early_stopping_patience", "weight_decay", "warmup_ratio"):
        _positive(training[field], f"config.training.{field}", allow_zero=True)
    if training["random_seed"] not in SUPPORTED_SEEDS:
        raise ConfigError(f"random_seed must be one of {SUPPORTED_SEEDS}")
    if training["selection_metric"] != SELECTION_METRIC:
        raise ConfigError(f"selection_metric must be {SELECTION_METRIC!r}; test selection is forbidden")
    weights = training["loss_weights"]
    if set(weights) != {"main", "hate", "sarcasm"}:
        raise ConfigError("loss_weights must contain main, hate, and sarcasm")
    for name, weight in weights.items():
        _positive(weight, f"config.training.loss_weights.{name}", allow_zero=True)
    class_weights = training["class_weights"]
    if set(class_weights) != {"english", "bangla"}:
        raise ConfigError("class_weights must contain english and bangla")
    for language, values in class_weights.items():
        if values is not None and (len(values) != 3 or any(value <= 0 for value in values)):
            raise ConfigError(f"class_weights.{language} must be null or three positive values")

    auxiliary = _require(config, "auxiliary_labels", "config")
    mode = _require(auxiliary, "mode", "config.auxiliary_labels")
    proxy_flag = _require(auxiliary, "proxy_auxiliary_labels", "config.auxiliary_labels")
    if mode not in {"none", "proxy", "gold"}:
        raise ConfigError("auxiliary_labels.mode must be none, proxy, or gold")
    if (mode == "proxy") != (proxy_flag is True):
        raise ConfigError("Proxy mode must explicitly set proxy_auxiliary_labels=true, and only proxy mode may do so")
    if "multi_task" in architecture and mode == "none":
        raise ConfigError("Multi-task architectures require gold labels or an explicitly flagged proxy diagnostic")
    if "single_task" in architecture and mode != "none":
        raise ConfigError("Single-task architectures must use auxiliary_labels.mode='none'")

    execution = _require(config, "execution", "config")
    for field in ("blocked", "evaluate_test", "output_directory", "allow_overwrite"):
        _require(execution, field, "config.execution")
    if execution["allow_overwrite"] is not False:
        raise ConfigError("Corrected runs must set allow_overwrite=false")
    if not str(execution["output_directory"]).startswith("/kaggle/working/corrected_"):
        raise ConfigError("output_directory must be a new /kaggle/working/corrected_* path")
    if execution["blocked"] and not execution.get("blocked_reason"):
        raise ConfigError("Blocked configurations require blocked_reason")

    if run_kind == "pilot":
        if execution["evaluate_test"] is not False:
            raise ConfigError("Pilot configurations must set execution.evaluate_test=false")
        if architecture not in {"xlmr_single_task", "mdistilbert_single_task"}:
            raise ConfigError("Stage 1B pilots permit only single-task architectures")
        if mode != "none" or proxy_flag is not False:
            raise ConfigError("Stage 1B pilots must not use auxiliary labels")
        if training["random_seed"] != 42:
            raise ConfigError("Stage 1B pilots require random_seed=42")
        expected_model = {
            "mdistilbert_single_task": "distilbert-base-multilingual-cased",
            "xlmr_single_task": "xlm-roberta-base",
        }[architecture]
        if model["name"] != expected_model or model["tokenizer_name"] != expected_model:
            raise ConfigError(f"Stage 1B {architecture} must use {expected_model}")
        if weights != {"main": 1.0, "hate": 0.0, "sarcasm": 0.0}:
            raise ConfigError("Stage 1B pilots must optimize only the three-class main loss")

    if run_kind == "smoke":
        limits = _require(config, "limits", "config")
        if training["epochs"] != 1:
            raise ConfigError("Smoke tests must use exactly one epoch")
        if limits.get("train_rows", 0) <= 0 or limits["train_rows"] > 512:
            raise ConfigError("Smoke train_rows must be in 1..512")
        if limits.get("validation_rows", 0) <= 0 or limits["validation_rows"] > 256:
            raise ConfigError("Smoke validation_rows must be in 1..256")
        if execution["evaluate_test"]:
            raise ConfigError("Smoke tests may not evaluate the test split")


def assert_execution_allowed(config: Mapping[str, Any]) -> None:
    """Refuse intentionally blocked plans before any heavy import or model load."""
    validate_config(config)
    execution = config["execution"]
    if execution["blocked"]:
        raise ConfigError(f"Execution blocked: {execution['blocked_reason']}")
