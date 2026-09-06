"""Corrected model definitions. No model is downloaded at module import time."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


class SingleTaskThreeClassModel(nn.Module):
    """One shared encoder and one three-class classification head."""

    def __init__(self, encoder: nn.Module, dropout: float = 0.3, num_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.three_class_head = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, **encoder_kwargs):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **encoder_kwargs)
        pooled = self.dropout(encoded.last_hidden_state[:, 0, :])
        return {"logits_3class": self.three_class_head(pooled)}


class SharedEncoderMultiTaskModel(nn.Module):
    """One encoder with independent three-class, hate, and sarcasm heads."""

    def __init__(self, encoder: nn.Module, dropout: float = 0.3, num_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.three_class_head = nn.Linear(hidden_size, num_classes)
        self.hate_head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )
        self.sarcasm_head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, **encoder_kwargs):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **encoder_kwargs)
        pooled = self.dropout(encoded.last_hidden_state[:, 0, :])
        return {
            "logits_3class": self.three_class_head(pooled),
            "logit_hate": self.hate_head(pooled).squeeze(-1),
            "logit_sarcasm": self.sarcasm_head(pooled).squeeze(-1),
        }


class XLMRSingleTaskModel(SingleTaskThreeClassModel):
    """XLM-R three-class baseline (encoder supplied by the builder)."""


class MultilingualDistilBERTSingleTaskModel(SingleTaskThreeClassModel):
    """Multilingual DistilBERT three-class baseline."""


class XLMRMultiTaskModel(SharedEncoderMultiTaskModel):
    """Corrected single-encoder XLM-R multi-task model."""


class MultilingualDistilBERTMultiTaskModel(SharedEncoderMultiTaskModel):
    """Corrected single-encoder multilingual DistilBERT multi-task model."""


MODEL_CLASSES = {
    "xlmr_single_task": XLMRSingleTaskModel,
    "mdistilbert_single_task": MultilingualDistilBERTSingleTaskModel,
    "xlmr_multi_task": XLMRMultiTaskModel,
    "mdistilbert_multi_task": MultilingualDistilBERTMultiTaskModel,
}


def build_model_and_tokenizer(config: Mapping[str, Any]):
    """Perform the explicit heavy model/tokenizer load when called on Kaggle."""
    from transformers import AutoModel, AutoTokenizer

    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer_name"], revision=model_config["tokenizer_revision"]
    )
    encoder = AutoModel.from_pretrained(model_config["name"], revision=model_config["revision"])
    model_class = MODEL_CLASSES[model_config["architecture"]]
    model = model_class(encoder=encoder, dropout=config["training"]["dropout"])
    return model, tokenizer

