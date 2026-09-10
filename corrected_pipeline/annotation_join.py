"""Fail-closed validation for future gold-annotation joins.

Text is deliberately excluded as a join identifier. This module performs no
automatic merge and has no dependency outside the Python standard library.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


STABLE_ID_CANDIDATES = ("sample_id", "record_id", "document_id", "uuid", "id")


class AnnotationJoinError(ValueError):
    """Raised when a gold-annotation join is absent, ambiguous, or incomplete."""


def require_stable_join_identifier(
    annotation_columns: Iterable[str], canonical_columns: Iterable[str]
) -> str:
    """Return a shared approved ID column or reject the join.

    Free text, row position, and fuzzy similarity are never accepted as keys.
    """
    shared = set(annotation_columns).intersection(canonical_columns)
    candidates = [name for name in STABLE_ID_CANDIDATES if name in shared]
    if len(candidates) != 1:
        raise AnnotationJoinError(
            "Gold annotations require exactly one shared stable ID column; "
            f"found {candidates}. Text-only and fuzzy joins are forbidden."
        )
    return candidates[0]


def validate_one_to_one_join(
    annotation_ids: Sequence[Any], canonical_ids: Sequence[Any]
) -> Mapping[str, int]:
    """Require complete, non-missing, one-to-one annotation/canonical IDs."""
    if any(value is None or str(value).strip() == "" for value in annotation_ids):
        raise AnnotationJoinError("Annotation stable IDs contain missing values")
    if any(value is None or str(value).strip() == "" for value in canonical_ids):
        raise AnnotationJoinError("Canonical stable IDs contain missing values")
    annotation_keys = [str(value) for value in annotation_ids]
    canonical_keys = [str(value) for value in canonical_ids]
    annotation_duplicates = sum(count - 1 for count in Counter(annotation_keys).values() if count > 1)
    canonical_duplicates = sum(count - 1 for count in Counter(canonical_keys).values() if count > 1)
    if annotation_duplicates or canonical_duplicates:
        raise AnnotationJoinError(
            "Stable ID join is not one-to-one: "
            f"annotation duplicate excess={annotation_duplicates}, "
            f"canonical duplicate excess={canonical_duplicates}"
        )
    canonical_set = set(canonical_keys)
    unmatched = sum(key not in canonical_set for key in annotation_keys)
    if unmatched:
        raise AnnotationJoinError(f"{unmatched} annotation stable IDs are unmatched")
    return {
        "annotation_rows": len(annotation_keys),
        "matched_rows": len(annotation_keys),
        "unmatched_rows": 0,
    }
