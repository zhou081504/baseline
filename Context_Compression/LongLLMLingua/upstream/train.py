"""Train the independent LongLLMLingua baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.risk_data import TextRiskDataset, load_splits
from common.risk_trainer import train_model
from model import LongLLMLinguaBinaryModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "merged_contact_orders.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/longllmlingua"))
    parser.add_argument("--model-name", default="hfl/chinese-bert-wwm-ext")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    splits = load_splits(
        args.csv,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    datasets = {
        name: TextRiskDataset(records, tokenizer, args.max_length)
        for name, records in {
            "train": splits.train,
            "val": splits.val,
            "test": splits.test,
        }.items()
    }
    loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=name == "train", collate_fn=TextRiskDataset.collate)
        for name, dataset in datasets.items()
    }
    model = LongLLMLinguaBinaryModel(args.model_name).to(device)
    train_model(
        model,
        tokenizer,
        loaders["train"],
        loaders["val"],
        loaders["test"],
        args.output_dir,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )
    with (args.output_dir / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"model_name": args.model_name}, handle, ensure_ascii=False)


if __name__ == "__main__":
    main()
