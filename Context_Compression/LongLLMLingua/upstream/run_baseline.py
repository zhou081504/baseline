"""Compress JSONL records and retain source fields for downstream evaluation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
from longllmlingua_baseline import LongLLMLingua
from longllmlingua_baseline.baseline import DEFAULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='NousResearch/Llama-2-7b-hf')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'], default='cuda')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--rate', type=float, default=0.5)
    group.add_argument('--target-token', type=int, default=-1)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('input and output must differ')
    if args.output.exists():
        parser.error('output already exists; choose a new path')
    if not 0 < args.rate <= 1 or (args.target_token != -1 and args.target_token <= 0):
        parser.error('invalid compression budget')
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error('input is empty')
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not {'context', 'question'} <= row.keys():
            parser.error(f'record {i} requires context and question')
    baseline = LongLLMLingua(args.model, args.device)
    import torch
    def synchronize():
        if args.device == 'cuda':
            torch.cuda.synchronize()
        elif args.device == 'mps':
            torch.mps.synchronize()
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'transformers', 'accelerate', 'tiktoken', 'nltk', 'numpy']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        for index, row in enumerate(rows):
            synchronize()
            start = time.perf_counter()
            result = baseline.compress(row['context'], row['question'],
                instruction=row.get('instruction', ''), rate=args.rate,
                target_token=args.target_token)
            synchronize()
            record = {'id': row.get('id', index), 'source': row, 'compression': result,
                'compression_seconds': time.perf_counter() - start,
                'config': dict(DEFAULTS, model=args.model, device=args.device,
                    rate=args.rate, target_token=args.target_token), 'versions': versions}
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
    print(f'Saved {len(rows)} records to {args.output}')


if __name__ == '__main__':
    main()
