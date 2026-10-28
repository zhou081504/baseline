"""BERT encoder with CogLTX-style block selection and a binary head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class CogLTXBertBinary(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.block_score = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, block_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()

        token_score = self.block_score(hidden).squeeze(-1)
        max_block = int(block_ids.max().item()) + 1
        block_scores = []
        for block in range(max_block):
            block_mask = (block_ids == block).float()
            denominator = block_mask.sum(dim=1).clamp(min=1)
            block_scores.append((token_score * block_mask).sum(dim=1) / denominator)
        block_scores = torch.stack(block_scores, dim=1)

        token_weight = F.softmax(block_scores, dim=1)
        selected_weight = token_weight.gather(1, block_ids).unsqueeze(-1)
        weighted = (hidden * mask * selected_weight).sum(dim=1)
        normalizer = (mask * selected_weight).sum(dim=1).clamp(min=1)
        return self.classifier(weighted / normalizer)[:, 1]
