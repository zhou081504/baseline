"""Run inference with the independent LongLLMLingua baseline."""

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
from common.risk_trainer import infer_dataset
from model import LongLLMLinguaBinaryModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "merged_contact_orders.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/longllmlingua"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with (args.output_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        threshold = json.load(handle).get("threshold", 0.5)
    with (args.output_dir / "model_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir / "tokenizer")
    model = LongLLMLinguaBinaryModel(config["model_name"])
    model.load_state_dict(torch.load(args.output_dir / "pytorch_model.bin", map_location=device))
    model.to(device).eval()

    records = load_splits(args.csv, seed=args.seed)
    selected = records.train + records.val + records.test if args.split == "all" else getattr(records, args.split)
    dataset = TextRiskDataset(selected, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=TextRiskDataset.collate)
    infer_dataset(model, loader, args.output_dir, device=device, threshold=threshold, prefix=f"infer_{args.split}")


if __name__ == "__main__":
    main()
