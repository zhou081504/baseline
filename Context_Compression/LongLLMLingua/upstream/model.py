"""Question-aware token compression head built over a BERT encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class LongLLMLinguaBinaryModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.token_importance = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        weights = torch.sigmoid(self.token_importance(hidden))
        weighted = hidden * mask * weights
        pooled = weighted.sum(dim=1) / (mask * weights).sum(dim=1).clamp(min=1)
        return self.classifier(pooled)[:, 1]
