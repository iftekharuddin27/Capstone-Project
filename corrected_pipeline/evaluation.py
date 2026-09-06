"""Reusable evaluation and artifact reporting for corrected experiments."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    true_positive = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    false_positive = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    false_negative = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return _safe_divide(2 * precision * recall, precision + recall)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] = (0, 1, 2),
    gold_hate_true: Sequence[int] | None = None,
    gold_hate_pred: Sequence[int] | None = None,
    gold_sarcasm_true: Sequence[int] | None = None,
    gold_sarcasm_pred: Sequence[int] | None = None,
) -> dict[str, Any]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and have equal length")
    matrix = [[0 for _ in labels] for _ in labels]
    positions = {label: index for index, label in enumerate(labels)}
    for truth, prediction in zip(y_true, y_pred):
        if truth not in positions or prediction not in positions:
            raise ValueError(f"Unexpected evaluation label pair: {truth}, {prediction}")
        matrix[positions[truth]][positions[prediction]] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    class_f1s: list[float] = []
    supports: list[int] = []
    for label in labels:
        index = positions[label]
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(labels)) if row != index)
        false_negative = sum(matrix[index][column] for column in range(len(labels)) if column != index)
        support = sum(matrix[index])
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        class_f1s.append(f1)
        supports.append(support)

    accuracy = _safe_divide(sum(matrix[i][i] for i in range(len(labels))), len(y_true))
    macro_f1 = sum(class_f1s) / len(class_f1s)
    weighted_f1 = _safe_divide(sum(f1 * support for f1, support in zip(class_f1s, supports)), sum(supports))
    main_hate_true = [int(label == 1) for label in y_true]
    main_hate_pred = [int(label == 1) for label in y_pred]
    main_sarcasm_true = [int(label == 2) for label in y_true]
    main_sarcasm_pred = [int(label == 2) for label in y_pred]

    hate_f1 = _binary_f1(gold_hate_true, gold_hate_pred) if gold_hate_true is not None else _binary_f1(main_hate_true, main_hate_pred)
    sarcasm_f1 = _binary_f1(gold_sarcasm_true, gold_sarcasm_pred) if gold_sarcasm_true is not None else _binary_f1(main_sarcasm_true, main_sarcasm_pred)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "hate_f1": hate_f1,
        "hate_f1_source": "available_auxiliary_axis" if gold_hate_true is not None else "main_class_1_one_vs_rest",
        "sarcasm_f1": sarcasm_f1,
        "sarcasm_f1_source": "available_auxiliary_axis" if gold_sarcasm_true is not None else "main_class_2_one_vs_rest",
        "hateful_class_recall": per_class["1"]["recall"],
    }


def evaluate_loader(model, loader, device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict[str, Any]] = []
    gold_hate_true: list[int] = []
    gold_hate_pred: list[int] = []
    gold_sarcasm_true: list[int] = []
    gold_sarcasm_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs["logits_3class"], dim=-1).cpu()
            predicted = probabilities.argmax(dim=-1)
            truths = batch["labels_main"].cpu()
            y_true.extend(int(value) for value in truths)
            y_pred.extend(int(value) for value in predicted)

            hate_probabilities = torch.sigmoid(outputs["logit_hate"]).cpu() if "logit_hate" in outputs else None
            sarcasm_probabilities = torch.sigmoid(outputs["logit_sarcasm"]).cpu() if "logit_sarcasm" in outputs else None
            for index in range(len(truths)):
                record = {
                    "row_id": int(batch["row_id"][index]),
                    "true_label": int(truths[index]),
                    "predicted_label": int(predicted[index]),
                    "probability_0": float(probabilities[index, 0]),
                    "probability_1": float(probabilities[index, 1]),
                    "probability_2": float(probabilities[index, 2]),
                }
                if hate_probabilities is not None:
                    record["hate_probability"] = float(hate_probabilities[index])
                    if bool(batch["mask_hate"][index]):
                        gold_hate_true.append(int(batch["labels_hate"][index]))
                        gold_hate_pred.append(int(hate_probabilities[index] >= 0.5))
                if sarcasm_probabilities is not None:
                    record["sarcasm_probability"] = float(sarcasm_probabilities[index])
                    if bool(batch["mask_sarcasm"][index]):
                        gold_sarcasm_true.append(int(batch["labels_sarcasm"][index]))
                        gold_sarcasm_pred.append(int(sarcasm_probabilities[index] >= 0.5))
                predictions.append(record)

    metrics = classification_metrics(
        y_true,
        y_pred,
        gold_hate_true=gold_hate_true or None,
        gold_hate_pred=gold_hate_pred or None,
        gold_sarcasm_true=gold_sarcasm_true or None,
        gold_sarcasm_pred=gold_sarcasm_pred or None,
    )
    return metrics, predictions


def environment_information() -> dict[str, Any]:
    packages = {}
    for package in ("torch", "transformers", "numpy", "pandas", "scikit-learn", "accelerate"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def sha256_artifact(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evaluation_artifacts(
    output_directory: str | Path,
    metrics: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    run_config: Mapping[str, Any],
    checkpoint_metadata: Mapping[str, Any],
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    environment = environment_information()
    environment["result_status"] = run_config["result_status"]
    (output / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "checkpoint_metadata.json").write_text(json.dumps(checkpoint_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if predictions:
        with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
