"""Strict canonical dataset validation using only the Python standard library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

from .labels import BANGLA_SOURCES, CLASS_2_SEMANTIC_STATUS, MAIN_LABELS


@dataclass(frozen=True)
class SplitSpec:
    language: str
    split: str
    expected_rows: int
    text_column: str
    label_column: str
    source_column: str | None = None


SPLIT_SPECS = {
    "en_train": SplitSpec("english", "train", 83455, "text_clean", "class"),
    "en_validation": SplitSpec("english", "validation", 10432, "text_clean", "class"),
    "en_test": SplitSpec("english", "test", 10432, "text_clean", "class"),
    "bn_train": SplitSpec("bangla", "train", 67009, "text_clean", "label", "source"),
    "bn_validation": SplitSpec("bangla", "validation", 8376, "text_clean", "label", "source"),
    "bn_test": SplitSpec("bangla", "test", 8377, "text_clean", "label", "source"),
}


class DataValidationError(ValueError):
    """Raised when canonical data does not match its required contract."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_split(path: str | Path, spec: SplitSpec) -> tuple[dict[str, Any], set[str], list[str]]:
    path = Path(path)
    errors: list[str] = []
    if not path.is_file():
        return {"path": str(path), "exists": False}, set(), [f"Missing file: {path}"]

    row_count = 0
    missing_text = 0
    missing_label = 0
    missing_source = 0
    unexpected_labels: set[str] = set()
    unexpected_sources: set[str] = set()
    observed_sources: set[str] = set()
    texts: set[str] = set()
    within_split_duplicates = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        required = {spec.text_column, spec.label_column}
        if spec.source_column:
            required.add(spec.source_column)
        missing_columns = sorted(required.difference(columns))
        unexpected_columns = sorted(set(columns).difference(required))
        if missing_columns:
            errors.append(f"{path}: missing required columns {missing_columns}; found {columns}")
            return {"path": str(path), "exists": True, "columns": columns}, set(), errors
        if unexpected_columns:
            errors.append(f"{path}: unexpected columns {unexpected_columns}; expected exactly {sorted(required)}")

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            text = row.get(spec.text_column)
            if text is None or text.strip() == "":
                missing_text += 1
            else:
                if text in texts:
                    within_split_duplicates += 1
                texts.add(text)

            raw_label = row.get(spec.label_column)
            if raw_label is None or raw_label.strip() == "":
                missing_label += 1
            else:
                try:
                    label = int(raw_label)
                except ValueError:
                    unexpected_labels.add(raw_label)
                else:
                    if label not in MAIN_LABELS:
                        unexpected_labels.add(raw_label)

            if spec.source_column:
                source = row.get(spec.source_column)
                if source is None or source.strip() == "":
                    missing_source += 1
                else:
                    observed_sources.add(source)
                    if source not in BANGLA_SOURCES:
                        unexpected_sources.add(source)

    if row_count != spec.expected_rows:
        errors.append(f"{path}: expected {spec.expected_rows:,} rows, found {row_count:,}")
    if missing_text:
        errors.append(f"{path}: {missing_text} missing/empty {spec.text_column} values")
    if missing_label:
        errors.append(f"{path}: {missing_label} missing {spec.label_column} values")
    if missing_source:
        errors.append(f"{path}: {missing_source} missing {spec.source_column} values")
    if unexpected_labels:
        errors.append(f"{path}: unexpected labels {sorted(unexpected_labels)}")
    if unexpected_sources:
        errors.append(f"{path}: unexpected Bangla sources {sorted(unexpected_sources)}")

    report = {
        "path": str(path),
        "exists": True,
        "columns": columns,
        "row_count": row_count,
        "expected_row_count": spec.expected_rows,
        "sha256": sha256_file(path),
        "missing_text": missing_text,
        "missing_label": missing_label,
        "missing_source": missing_source,
        "unexpected_labels": sorted(unexpected_labels),
        "observed_sources": sorted(observed_sources),
        "unexpected_sources": sorted(unexpected_sources),
        "within_split_exact_duplicates": within_split_duplicates,
    }
    return report, texts, errors


def validate_canonical_data(
    paths: Mapping[str, str | Path],
    expected_hashes: Mapping[str, str],
    *,
    specs: Mapping[str, SplitSpec] = SPLIT_SPECS,
) -> dict[str, Any]:
    """Validate schema, counts, labels, sources, hashes, and split isolation.

    The function validates complete canonical files before any experiment may
    take a subset. Every mismatch is collected into one actionable exception.
    """
    errors: list[str] = []
    if set(paths) != set(specs):
        errors.append(f"Path keys must be exactly {sorted(specs)}; got {sorted(paths)}")
    if set(expected_hashes) != set(specs):
        errors.append(f"Hash keys must be exactly {sorted(specs)}; got {sorted(expected_hashes)}")
    if errors:
        raise DataValidationError("Canonical data validation failed:\n- " + "\n- ".join(errors))

    reports: dict[str, dict[str, Any]] = {}
    text_sets: dict[str, set[str]] = {}
    for key, spec in specs.items():
        report, texts, split_errors = inspect_split(paths[key], spec)
        reports[key] = report
        text_sets[key] = texts
        errors.extend(split_errors)
        actual_hash = report.get("sha256")
        expected_hash = expected_hashes[key].lower()
        if actual_hash is not None and actual_hash != expected_hash:
            errors.append(f"{paths[key]}: SHA-256 mismatch; expected {expected_hash}, found {actual_hash}")

    for language, keys in {
        "english": ("en_train", "en_validation", "en_test"),
        "bangla": ("bn_train", "bn_validation", "bn_test"),
    }.items():
        for left, right in combinations(keys, 2):
            overlap = text_sets[left].intersection(text_sets[right])
            if overlap:
                errors.append(f"{language}: {len(overlap)} exact text duplicates across {left} and {right}")

    observed_bangla_sources = set().union(
        *(set(reports[key].get("observed_sources", [])) for key in ("bn_train", "bn_validation", "bn_test"))
    )
    if observed_bangla_sources != set(BANGLA_SOURCES):
        errors.append(
            "Bangla source vocabulary mismatch; expected "
            f"{sorted(BANGLA_SOURCES)}, found {sorted(observed_bangla_sources)}"
        )

    if errors:
        raise DataValidationError("Canonical data validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "passed",
        "label_mapping": MAIN_LABELS,
        "class_2_semantic_status": CLASS_2_SEMANTIC_STATUS,
        "bangla_sources": list(BANGLA_SOURCES),
        "splits": reports,
        "cross_split_exact_duplicates": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Corrected experiment JSON configuration")
    parser.add_argument("--output", help="Optional JSON validation report path")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    report = validate_canonical_data(config["dataset"]["paths"], config["dataset"]["hashes"])
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
