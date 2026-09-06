"""Import-independent declaration of multi-task loss/target wiring."""

from __future__ import annotations


LOSS_BINDINGS = {
    "main": {"logit": "logits_3class", "target": "labels_main", "mask": None, "loss": "cross_entropy"},
    "hate": {"logit": "logit_hate", "target": "labels_hate", "mask": "mask_hate", "loss": "masked_bce"},
    "sarcasm": {"logit": "logit_sarcasm", "target": "labels_sarcasm", "mask": "mask_sarcasm", "loss": "masked_bce"},
}


def validate_loss_bindings(bindings=LOSS_BINDINGS) -> None:
    expected = {
        "main": ("logits_3class", "labels_main", None),
        "hate": ("logit_hate", "labels_hate", "mask_hate"),
        "sarcasm": ("logit_sarcasm", "labels_sarcasm", "mask_sarcasm"),
    }
    if set(bindings) != set(expected):
        raise ValueError("Loss bindings must contain exactly main, hate, and sarcasm")
    for task, (logit, target, mask) in expected.items():
        actual = bindings[task]
        if (actual.get("logit"), actual.get("target"), actual.get("mask")) != (logit, target, mask):
            raise ValueError(f"Incorrect {task} loss wiring: {actual}")


validate_loss_bindings()
