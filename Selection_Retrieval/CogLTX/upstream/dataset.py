"""Dataset with token block identifiers for the CogLTX baseline."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class BlockRiskDataset(Dataset):
    def __init__(self, records, tokenizer, max_length: int = 512, block_size: int = 64):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.block_size = block_size
        self.labels = torch.tensor([record.label for record in records], dtype=torch.float)
        self.ids = [record.id for record in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.records[index].raw_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids = encoding["input_ids"].squeeze(0)
        mask = encoding["attention_mask"].squeeze(0)
        positions = torch.arange(self.max_length)
        block_ids = positions // self.block_size
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "block_ids": block_ids,
        }, self.labels[index]

    @staticmethod
    def collate(batch):
        inputs, labels = zip(*batch)
        return {
            "input_ids": torch.stack([item["input_ids"] for item in inputs]),
            "attention_mask": torch.stack([item["attention_mask"] for item in inputs]),
            "block_ids": torch.stack([item["block_ids"] for item in inputs]),
        }, torch.stack(labels)


def build_dataset(records, tokenizer, max_length: int = 512, block_size: int = 64):
    return BlockRiskDataset(records, tokenizer, max_length, block_size)
