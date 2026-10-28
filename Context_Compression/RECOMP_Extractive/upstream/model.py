"""BERT encoder with extractive sentence selection and a binary head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class RECOMPBinaryModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.sentence_score = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, sentence_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        token_score = self.sentence_score(hidden).squeeze(-1)

        max_sentence = int(sentence_ids.max().item()) + 1
        sentence_scores = []
        for sentence in range(max_sentence):
            sentence_mask = (sentence_ids == sentence).float()
            denom = sentence_mask.sum(dim=1).clamp(min=1)
            sentence_scores.append((token_score * sentence_mask).sum(dim=1) / denom)
        sentence_scores = torch.stack(sentence_scores, dim=1)
        token_weight = F.softmax(sentence_scores, dim=1).gather(1, sentence_ids).unsqueeze(-1)
        pooled = (hidden * mask * token_weight).sum(dim=1)
        normalizer = (mask * token_weight).sum(dim=1).clamp(min=1)
        return self.classifier(pooled / normalizer)[:, 1]
