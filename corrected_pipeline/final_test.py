"""Inference-only, frozen Stage 2 XLM-R final-test evaluator.

Run once per preselected language/seed checkpoint. The existing training runner
and validation-only configurations remain locked and are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import STAGE2_STATUS, STAGE2_XLMR_REVISION, load_config
from .data_validation import SPLIT_SPECS, validate_canonical_data
from .evaluation import sha256_artifact


FINAL_STATUS = "FINAL TEST — FROZEN VALIDATION SELECTION"
FROZEN_RUNS = {
    "bangla": {
        42: ("07137bcdc8674f48059134b9ccc565052ea5393e27972ec47847330c2d52801c", 0.8121981275460177),
        123: ("673aa3c660c2718c49036999cc596fc0cb3fc9edea447e784f53a5db188fe743", 0.8104287245666996),
        2026: ("6c1f4147723b01e5317ca0df1b4f03ceaeabb0c422d8669dee94101b5ebcc0f1", 0.8090208434925755),
    },
    "english": {
        42: ("26f4a8c92423ecc3ef51f5afb95395a00a9ec3ecfcfe6ec023d826655ea50e44", 0.8922108354193766),
        123: ("5aa933276a1aa5abd901e4ac92853e056837877e5dc3fa7ecb74b71df38846e8", 0.8902751881284828),
        2026: ("5fd1194ac864db98df6a58fa61345700fdd97b8c42ddc69e0583583f69f615f8", 0.8943968198418402),
    },
}
FROZEN_TEST_HASHES = {
    "bangla": "c2ac6c268ad94284be0bff8857f9d2f90e68610075bc0c60dac9ff008496df0d",
    "english": "a4647d638471bdee88c3da0d3d0c7110451a2265b89e6a32df9efd4a23bfd3eb",
}


def frozen_run(language: str, seed: int, repo: str | Path) -> dict:
    """Load a preselected config and its frozen checkpoint/test identities."""
    if language not in FROZEN_RUNS or seed not in FROZEN_RUNS[language]:
        raise ValueError("Only the six frozen XLM-R language/seed pairs are allowed")
    repo = Path(repo).resolve()
    prefix = "bn" if language == "bangla" else "en"
    config_path = repo / "configs" / f"stage2_reportable_{prefix}_xlmr_seed{seed}.json"
    config = load_config(config_path)
    expected_paths = {
        f"{prefix}_train": f"Phase 2/{'Bangla' if prefix == 'bn' else 'English'} data/{prefix}_train.csv",
        f"{prefix}_validation": f"Phase 2/{'Bangla' if prefix == 'bn' else 'English'} data/{prefix}_val.csv",
    }
    if (config["run_kind"] != "reportable_validation" or config["result_status"] != STAGE2_STATUS
            or config["dataset"]["languages"] != [language]
            or config["dataset"]["paths"] != expected_paths
            or config["model"]["architecture"] != "xlmr_single_task"
            or config["model"]["revision"] != STAGE2_XLMR_REVISION
            or config["training"]["random_seed"] != seed
            or config["auxiliary_labels"]["mode"] != "none"
            or config["execution"]["evaluate_test"] is not False
            or config["execution"]["test_locked"] is not True):
        raise ValueError("Stage 2 training configuration differs from the frozen selection")
    paths = {key: repo / value for key, value in expected_paths.items()}
    paths[f"{prefix}_test"] = repo / f"Phase 2/{'Bangla' if prefix == 'bn' else 'English'} data/{prefix}_test.csv"
    hashes = dict(config["dataset"]["hashes"])
    hashes[f"{prefix}_test"] = {"canonical_sha256": FROZEN_TEST_HASHES[language]}
    expected_checkpoint_sha256, validation_macro_f1 = FROZEN_RUNS[language][seed]
    return {
        "repo": repo, "config_path": config_path, "config": config,
        "prefix": prefix, "paths": paths, "hashes": hashes,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "validation_macro_f1": validation_macro_f1,
        "output": Path("/kaggle/working") / f"corrected_final_test_{prefix}_xlmr_seed{seed}",
    }


def verify_checkpoint(path: str | Path, expected_sha256: str) -> str:
    """Refuse to unpickle or load a checkpoint unless its frozen hash matches."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint missing or empty: {path}")
    actual = sha256_artifact(path)
    if actual != expected_sha256:
        raise ValueError(f"Wrong checkpoint SHA-256 for {path}: expected {expected_sha256}, found {actual}")
    return actual


def run_final_test(language: str, seed: int, checkpoint: str | Path, repo: str | Path) -> Path:
    frozen = frozen_run(language, seed, repo)
    output = frozen["output"]
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite final-test output: {output} or {partial}")
    checkpoint = Path(checkpoint).resolve()
    verified_hash = verify_checkpoint(checkpoint, frozen["checkpoint_sha256"])

    # Full train/validation/test validation checks all declared hashes, label
    # schemas, source vocabulary, row counts, and cross-split exact duplicates.
    specs = {key: SPLIT_SPECS[key] for key in frozen["paths"]}
    data_report = validate_canonical_data(frozen["paths"], frozen["hashes"], specs=specs)

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    from .evaluation import environment_information, evaluate_loader
    from .models import XLMRSingleTaskModel
    from .runner import BatchCollator, TextDataset

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator for final-test inference")
    config = frozen["config"]
    model_info = config["model"]
    encoder_config = AutoConfig.from_pretrained(model_info["name"], revision=model_info["revision"])
    # Construct the exact trained architecture; avoid downloading base weights
    # that will immediately be replaced by the verified training checkpoint.
    encoder = AutoModel.from_config(encoder_config)
    model = XLMRSingleTaskModel(encoder=encoder, dropout=config["training"]["dropout"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a state_dict of tensor weights")
    model.load_state_dict(state, strict=True)
    del state
    model.to(torch.device("cuda"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_info["tokenizer_name"], revision=model_info["tokenizer_revision"]
    )

    test_path = frozen["paths"][f"{frozen['prefix']}_test"]
    frame = pd.read_csv(test_path)
    label_column = "label" if language == "bangla" else "class"
    dataset = TextDataset(frame, label_column, "none", config["auxiliary_labels"])
    loader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"], shuffle=False,
        collate_fn=BatchCollator(tokenizer, config["training"]["maximum_sequence_length"]),
    )
    metrics, predictions = evaluate_loader(model, loader, torch.device("cuda"))
    if len(predictions) != SPLIT_SPECS[f"{frozen['prefix']}_test"].expected_rows:
        raise RuntimeError("Unexpected test prediction row count")
    if any(row["row_id"] != i or row["true_label"] != int(frame[label_column].iloc[i])
           for i, row in enumerate(predictions)):
        raise RuntimeError("Test predictions do not match the canonical row order and labels")
    metrics = {"result_status": FINAL_STATUS, **metrics}
    for row in predictions:
        row["result_status"] = FINAL_STATUS
    provenance = {
        "result_status": FINAL_STATUS,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": verified_hash,
        "training_config_path": str(frozen["config_path"]),
        "training_config_sha256": sha256_artifact(frozen["config_path"]),
        "selected_by": "validation_macro_f1", "validation_macro_f1": frozen["validation_macro_f1"],
        "language": language, "seed": seed, "model": model_info,
        "test_path": str(test_path), "test_canonical_sha256": FROZEN_TEST_HASHES[language],
        "test_rows": len(predictions), "training_performed": False,
    }

    partial.mkdir(parents=True)
    for filename, value in (
        ("metrics.json", metrics), ("provenance.json", provenance),
        ("data_validation.json", data_report), ("environment.json", environment_information()),
    ):
        (partial / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (partial / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    partial.rename(output)
    print(json.dumps({"output": str(output), "macro_f1": metrics["macro_f1"],
                      "sarcasm_f1": metrics["sarcasm_f1"], "checkpoint_sha256": verified_hash}, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=sorted(FROZEN_RUNS))
    parser.add_argument("--seed", required=True, type=int, choices=(42, 123, 2026))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    run_final_test(args.language, args.seed, args.checkpoint, args.repo)


if __name__ == "__main__":
    main()
