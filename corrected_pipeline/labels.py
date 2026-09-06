"""Canonical label meanings and explicit auxiliary-label provenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence


MAIN_LABELS = {0: "Non-hateful", 1: "Hateful", 2: "Sarcastic"}
CLASS_2_SEMANTIC_STATUS = "pending_supervisor_confirmation"
BANGLA_SOURCES = ("ALERT", "BD_SHS", "BenSarc", "BanglaSarc3", "BIDWESH")
MISSING_AUXILIARY_VALUES = (None, "", "NA", "N/A", -1, "-1")


class LabelError(ValueError):
    """Raised for ambiguous, missing, or unexpected label values."""


class LabelProvenance(str, Enum):
    MAIN_DATASET = "main_dataset"
    PROXY_AUXILIARY = "proxy_auxiliary"
    HUMAN_GOLD = "human_gold"


@dataclass(frozen=True)
class AuxiliaryTargets:
    hate: tuple[float, ...]
    sarcasm: tuple[float, ...]
    hate_mask: tuple[bool, ...]
    sarcasm_mask: tuple[bool, ...]
    provenance: LabelProvenance
    proxy_auxiliary_labels: bool


def normalize_main_labels(values: Iterable[Any]) -> tuple[int, ...]:
    normalized = []
    for index, value in enumerate(values):
        try:
            label = int(value)
        except (TypeError, ValueError) as exc:
            raise LabelError(f"Main label at index {index} is not an integer: {value!r}") from exc
        if label not in MAIN_LABELS:
            raise LabelError(f"Unexpected main label at index {index}: {label}")
        normalized.append(label)
    return tuple(normalized)


def proxy_auxiliary_targets(
    main_labels: Iterable[Any], *, explicitly_enabled: bool
) -> AuxiliaryTargets:
    if not explicitly_enabled:
        raise LabelError("Proxy auxiliary labels require proxy_auxiliary_labels=true")
    labels = normalize_main_labels(main_labels)
    return AuxiliaryTargets(
        hate=tuple(float(label == 1) for label in labels),
        sarcasm=tuple(float(label == 2) for label in labels),
        hate_mask=tuple(True for _ in labels),
        sarcasm_mask=tuple(True for _ in labels),
        provenance=LabelProvenance.PROXY_AUXILIARY,
        proxy_auxiliary_labels=True,
    )


def _is_missing_auxiliary(value: Any) -> bool:
    if value is None:
        return True
    if value.__class__.__name__ == "NAType":  # pandas.NA without importing pandas.
        return True
    if isinstance(value, str):
        return value.strip().upper() in {"", "NA", "N/A", "-1"}
    try:
        if bool(value == -1):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(value != value)  # NaN without importing pandas/numpy.
    except (TypeError, ValueError):
        return False


def _gold_axis(values: Sequence[Any], axis_name: str) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    targets: list[float] = []
    masks: list[bool] = []
    for index, value in enumerate(values):
        if _is_missing_auxiliary(value):
            targets.append(0.0)
            masks.append(False)
            continue
        try:
            target = int(value)
        except (TypeError, ValueError) as exc:
            raise LabelError(f"Gold {axis_name} label at index {index} is invalid: {value!r}") from exc
        if target not in (0, 1):
            raise LabelError(f"Gold {axis_name} label at index {index} must be 0, 1, or missing")
        targets.append(float(target))
        masks.append(True)
    return tuple(targets), tuple(masks)


def gold_auxiliary_targets(
    hate_values: Sequence[Any], sarcasm_values: Sequence[Any]
) -> AuxiliaryTargets:
    if len(hate_values) != len(sarcasm_values):
        raise LabelError("Gold hate and sarcasm axes must have equal length")
    hate, hate_mask = _gold_axis(hate_values, "hate")
    sarcasm, sarcasm_mask = _gold_axis(sarcasm_values, "sarcasm")
    return AuxiliaryTargets(
        hate=hate,
        sarcasm=sarcasm,
        hate_mask=hate_mask,
        sarcasm_mask=sarcasm_mask,
        provenance=LabelProvenance.HUMAN_GOLD,
        proxy_auxiliary_labels=False,
    )
