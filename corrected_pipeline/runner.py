"""Kaggle-only entry point for corrected smoke/full experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .config import assert_execution_allowed, load_config
from .data_validation import validate_canonical_data
from .labels import gold_auxiliary_targets, proxy_auxiliary_targets


def _prepare_output(path: str) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite corrected output directory: {output}")
    output.mkdir(parents=True)
    return output


def _stratified_limit(frame, label_column: str, limit: int, seed: int):
    if limit >= len(frame):
        return frame.reset_index(drop=True)
    per_class = max(limit // 3, 1)
    pieces = []
    for _, group in frame.groupby(label_column, sort=True):
        pieces.append(group.sample(n=min(per_class, len(group)), random_state=seed))
    limited = __import__("pandas").concat(pieces).sample(frac=1.0, random_state=seed)
    if len(limited) < limit:
        remaining = frame.drop(index=limited.index)
        limited = __import__("pandas").concat([
            limited, remaining.sample(n=min(limit - len(limited), len(remaining)), random_state=seed)
        ])
    return limited.head(limit).reset_index(drop=True)


class TextDataset:
    """Lightweight row container; tokenization happens per batch."""

    def __init__(self, frame, label_column: str, auxiliary_mode: str, auxiliary_config: Mapping[str, Any]):
        self.frame = frame.reset_index(drop=True)
        self.label_column = label_column
        self.auxiliary_mode = auxiliary_mode
        self.auxiliary_config = auxiliary_config
        labels = self.frame[label_column].tolist()
        if auxiliary_mode == "proxy":
            self.auxiliary = proxy_auxiliary_targets(
                labels, explicitly_enabled=auxiliary_config["proxy_auxiliary_labels"]
            )
        elif auxiliary_mode == "gold":
            hate_column = auxiliary_config["gold_hate_column"]
            sarcasm_column = auxiliary_config["gold_sarcasm_column"]
            missing = [column for column in (hate_column, sarcasm_column) if column not in self.frame.columns]
            if missing:
                raise ValueError(
                    f"Gold auxiliary columns {missing} are absent. The pipeline never merges annotations automatically."
                )
            self.auxiliary = gold_auxiliary_targets(
                self.frame[hate_column].tolist(), self.frame[sarcasm_column].tolist()
            )
        else:
            count = len(self.frame)
            self.auxiliary = gold_auxiliary_targets([None] * count, [None] * count)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return {
            "row_id": index,
            "text": row["text_clean"],
            "labels_main": int(row[self.label_column]),
            "labels_hate": self.auxiliary.hate[index],
            "labels_sarcasm": self.auxiliary.sarcasm[index],
            "mask_hate": self.auxiliary.hate_mask[index],
            "mask_sarcasm": self.auxiliary.sarcasm_mask[index],
        }


class BatchCollator:
    def __init__(self, tokenizer, maximum_sequence_length: int):
        self.tokenizer = tokenizer
        self.maximum_sequence_length = maximum_sequence_length

    def __call__(self, rows):
        import torch

        encoded = self.tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            padding=True,
            max_length=self.maximum_sequence_length,
            return_tensors="pt",
        )
        encoded.update({
            "row_id": torch.tensor([row["row_id"] for row in rows], dtype=torch.long),
            "labels_main": torch.tensor([row["labels_main"] for row in rows], dtype=torch.long),
            "labels_hate": torch.tensor([row["labels_hate"] for row in rows], dtype=torch.float),
            "labels_sarcasm": torch.tensor([row["labels_sarcasm"] for row in rows], dtype=torch.float),
            "mask_hate": torch.tensor([row["mask_hate"] for row in rows], dtype=torch.bool),
            "mask_sarcasm": torch.tensor([row["mask_sarcasm"] for row in rows], dtype=torch.bool),
        })
        return encoded


def _smoke_checks(model, loader, device, training_config):
    import torch
    from .training import compute_multitask_loss

    batch = next(iter(loader))
    device_batch = {name: value.to(device) for name, value in batch.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=device_batch["input_ids"],
            attention_mask=device_batch["attention_mask"],
        )
        class_weights = training_config["active_class_weights"]
        class_weights = None if class_weights is None else torch.tensor(class_weights, device=device)
        loss, components = compute_multitask_loss(
            outputs, device_batch, training_config["loss_weights"], class_weights
        )
        masked_batch = dict(device_batch)
        masked_batch["mask_hate"] = torch.zeros_like(device_batch["mask_hate"])
        masked_batch["mask_sarcasm"] = torch.zeros_like(device_batch["mask_sarcasm"])
        masked_loss, masked_components = compute_multitask_loss(
            outputs, masked_batch, training_config["loss_weights"], class_weights
        )
    return {
        "status": "SMOKE TEST — NOT A REPORTED RESULT",
        "forward_pass": outputs["logits_3class"].shape[-1] == 3,
        "finite_combined_loss": bool(torch.isfinite(loss)),
        "hate_target_wired": components.get("hate") is not None,
        "sarcasm_target_wired": components.get("sarcasm") is not None,
        "all_missing_hate_mask_skips_loss": masked_components.get("hate") is None,
        "all_missing_sarcasm_mask_skips_loss": masked_components.get("sarcasm") is None,
        "main_loss_survives_missing_auxiliary_labels": bool(torch.isfinite(masked_loss)),
    }


def _set_seed(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    assert_execution_allowed(config)

    # Strict full-file validation happens before pandas, torch, or transformers are imported.
    validation_report = validate_canonical_data(
        config["dataset"]["paths"], config["dataset"]["hashes"]
    )
    output_root = _prepare_output(config["execution"]["output_directory"])
    (output_root / "data_validation.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    from .evaluation import evaluate_loader, sha256_artifact, write_evaluation_artifacts
    from .models import build_model_and_tokenizer
    from .training import train_with_validation_selection

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Run this corrected pipeline on Kaggle.")
    seed = int(config["training"]["random_seed"])
    _set_seed(seed)
    device = torch.device("cuda")

    for language in config["dataset"]["languages"]:
        prefix = "en" if language == "english" else "bn"
        label_column = "class" if language == "english" else "label"
        train_frame = pd.read_csv(config["dataset"]["paths"][f"{prefix}_train"])
        validation_frame = pd.read_csv(config["dataset"]["paths"][f"{prefix}_validation"])
        if config["run_kind"] == "smoke":
            train_frame = _stratified_limit(train_frame, label_column, config["limits"]["train_rows"], seed)
            validation_frame = _stratified_limit(
                validation_frame, label_column, config["limits"]["validation_rows"], seed
            )

        model, tokenizer = build_model_and_tokenizer(config)
        model.to(device)
        auxiliary = config["auxiliary_labels"]
        train_dataset = TextDataset(train_frame, label_column, auxiliary["mode"], auxiliary)
        validation_dataset = TextDataset(validation_frame, label_column, auxiliary["mode"], auxiliary)
        collator = BatchCollator(tokenizer, config["training"]["maximum_sequence_length"])
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            collate_fn=collator,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collator,
        )

        active_training = deepcopy(config["training"])
        active_training["active_class_weights"] = config["training"]["class_weights"][language]
        smoke_checks_path = None
        if config["run_kind"] == "smoke":
            checks = _smoke_checks(model, train_loader, device, active_training)
            smoke_checks_path = output_root / f"{language}_smoke_checks.json"
            smoke_checks_path.write_text(
                json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        updates_per_epoch = math.ceil(
            len(train_loader) / config["training"]["gradient_accumulation"]
        )
        total_updates = updates_per_epoch * config["training"]["epochs"]
        warmup_steps = int(total_updates * config["training"]["warmup_ratio"])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

        def validation_fn(current_model, current_loader, current_device):
            return evaluate_loader(current_model, current_loader, current_device)[0]

        result = train_with_validation_selection(
            model,
            train_loader,
            validation_loader,
            optimizer,
            scheduler,
            device,
            active_training,
            validation_fn,
        )
        model.load_state_dict(result.best_state_cpu)
        language_output = output_root / language
        language_output.mkdir()
        checkpoint_path = language_output / "best_checkpoint_cpu.pt"
        torch.save(result.best_state_cpu, checkpoint_path)
        if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            raise RuntimeError("Checkpoint save verification failed")
        tokenizer.save_pretrained(language_output / "tokenizer")
        checkpoint_hash = sha256_artifact(checkpoint_path)
        if smoke_checks_path is not None:
            checks["checkpoint_saving"] = True
            checks["checkpoint_sha256"] = checkpoint_hash
            smoke_checks_path.write_text(
                json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

        validation_metrics, validation_predictions = evaluate_loader(model, validation_loader, device)
        validation_metrics = {"result_status": config["result_status"], **validation_metrics}
        for prediction in validation_predictions:
            prediction["result_status"] = config["result_status"]
        checkpoint_metadata = {
            "result_status": config["result_status"],
            "selected_by": "validation_macro_f1",
            "best_validation_macro_f1": result.best_validation_macro_f1,
            "best_epoch": result.best_epoch,
            "checkpoint_device": "cpu",
            "checkpoint_sha256": checkpoint_hash,
            "model_name": config["model"]["name"],
            "model_revision": config["model"]["revision"],
            "tokenizer_name": config["model"]["tokenizer_name"],
            "tokenizer_revision": config["model"]["tokenizer_revision"],
            "proxy_auxiliary_labels": auxiliary["proxy_auxiliary_labels"],
            "data_hashes": config["dataset"]["hashes"],
        }
        report_config = deepcopy(config)
        report_config["active_language"] = language
        report_config["training_history"] = result.history
        write_evaluation_artifacts(
            language_output / "validation",
            validation_metrics,
            validation_predictions,
            report_config,
            checkpoint_metadata,
        )

        if config["execution"]["evaluate_test"]:
            test_frame = pd.read_csv(config["dataset"]["paths"][f"{prefix}_test"])
            test_dataset = TextDataset(test_frame, label_column, auxiliary["mode"], auxiliary)
            test_loader = DataLoader(
                test_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                collate_fn=collator,
            )
            test_metrics, test_predictions = evaluate_loader(model, test_loader, device)
            test_metrics = {"result_status": config["result_status"], **test_metrics}
            for prediction in test_predictions:
                prediction["result_status"] = config["result_status"]
            write_evaluation_artifacts(
                language_output / "test",
                test_metrics,
                test_predictions,
                report_config,
                checkpoint_metadata,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
