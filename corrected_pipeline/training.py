"""Corrected loss wiring and training loop for Kaggle execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from .loss_wiring import LOSS_BINDINGS, validate_loss_bindings


@dataclass
class TrainingResult:
    best_state_cpu: dict[str, torch.Tensor]
    best_validation_macro_f1: float
    best_epoch: int
    history: list[dict[str, float]]


def masked_binary_cross_entropy(logits, targets, mask):
    mask = mask.to(dtype=torch.bool)
    if mask.numel() == 0 or not bool(mask.any()):
        return None
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask].float())


def compute_multitask_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    loss_weights: Mapping[str, float],
    class_weights: torch.Tensor | None,
):
    """Wire each head only to its matching target and optional mask."""
    validate_loss_bindings(LOSS_BINDINGS)
    main_loss = F.cross_entropy(
        outputs["logits_3class"], batch["labels_main"], weight=class_weights
    )
    components: dict[str, torch.Tensor | None] = {"main": main_loss}
    total = loss_weights["main"] * main_loss

    if "logit_hate" in outputs:
        hate_loss = masked_binary_cross_entropy(
            outputs["logit_hate"], batch["labels_hate"], batch["mask_hate"]
        )
        components["hate"] = hate_loss
        if hate_loss is not None:
            total = total + loss_weights["hate"] * hate_loss

    if "logit_sarcasm" in outputs:
        sarcasm_loss = masked_binary_cross_entropy(
            outputs["logit_sarcasm"], batch["labels_sarcasm"], batch["mask_sarcasm"]
        )
        components["sarcasm"] = sarcasm_loss
        if sarcasm_loss is not None:
            total = total + loss_weights["sarcasm"] * sarcasm_loss
    return total, components


def clone_state_dict_to_cpu(model) -> dict[str, torch.Tensor]:
    """Copy the selected checkpoint off GPU to avoid retaining duplicate VRAM."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_with_validation_selection(
    model,
    train_loader,
    validation_loader,
    optimizer,
    scheduler,
    device,
    training_config: Mapping[str, Any],
    validation_fn: Callable[..., Mapping[str, float]],
) -> TrainingResult:
    """Train without touching the test set; select by validation macro F1."""
    epochs = int(training_config["epochs"])
    accumulation = int(training_config["gradient_accumulation"])
    patience = int(training_config["early_stopping_patience"])
    class_weight_values = training_config["active_class_weights"]
    class_weights = None
    if class_weight_values is not None:
        class_weights = torch.tensor(class_weight_values, dtype=torch.float, device=device)

    best_score = -math.inf
    best_epoch = 0
    best_state_cpu = None
    history: list[dict[str, float]] = []
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items() if name != "row_id"}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss, _ = compute_multitask_loss(
                outputs, batch, training_config["loss_weights"], class_weights
            )
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_config["gradient_clipping"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu())

        metrics = validation_fn(model, validation_loader, device)
        score = float(metrics["macro_f1"])
        history.append({
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            "validation_macro_f1": score,
        })
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state_cpu = clone_state_dict_to_cpu(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                break

    if best_state_cpu is None:
        raise RuntimeError("Training produced no selectable validation checkpoint")
    return TrainingResult(best_state_cpu, best_score, best_epoch, history)

