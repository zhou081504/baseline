"""Dataset with sentence identifiers for the RECOMP baseline."""

from __future__ import annotations

import re
from bisect import bisect_right

import torch
from torch.utils.data import Dataset


class SentenceRiskDataset(Dataset):
    def __init__(self, records, tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = torch.tensor([record.label for record in records], dtype=torch.float)
        self.ids = [record.id for record in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        raw = self.records[index].raw_text
        encoding = self.tokenizer(
            raw,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoding.pop("offset_mapping").squeeze(0)
        sentence_ids = torch.zeros(self.max_length, dtype=torch.long)
        boundaries = [match.start() for match in re.finditer(r"[。！？!?；;\n]", raw)]
        for position, (start, end) in enumerate(offsets.tolist()):
            sentence_ids[position] = bisect_right(boundaries, start)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "sentence_ids": sentence_ids,
        }, self.labels[index]

    @staticmethod
    def collate(batch):
        inputs, labels = zip(*batch)
        return {
            "input_ids": torch.stack([item["input_ids"] for item in inputs]),
            "attention_mask": torch.stack([item["attention_mask"] for item in inputs]),
            "sentence_ids": torch.stack([item["sentence_ids"] for item in inputs]),
        }, torch.stack(labels)


def build_dataset(records, tokenizer, max_length: int = 512):
    return SentenceRiskDataset(records, tokenizer, max_length)
