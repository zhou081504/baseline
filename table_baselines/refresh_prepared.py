"""Refresh labels/splits without rerunning label-independent CARROT selection.

Reuse is permitted only after reconstructing the old source CSV byte-for-byte
(verified by SHA256) by restoring its old labels. Any text/ID/order change fails.
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

from common.data import ID2LABEL, LABEL_MAP
from .runtime import empty_output, sha256, split_records, write_json


def restored_hash(rows, fields, old_labels, newline='\n', bom=True):
    digest = hashlib.sha256(b'\xef\xbb\xbf' if bom else b'')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fields, lineterminator=newline)
    writer.writeheader()
    digest.update(stream.getvalue().encode('utf-8'))
    for row in rows:
        restored = dict(row, label=ID2LABEL[old_labels[row['id']]])
        stream.seek(0); stream.truncate(0)
        writer.writerow(restored)
        digest.update(stream.getvalue().encode('utf-8'))
    return digest.hexdigest()


def refresh(args):
    old = Path(args.prepared)
    meta = json.loads((old/'prepare_config.json').read_text())
    if meta.get('method') != 'carrot' or meta.get('selection_uses_gold_labels') is not False or meta.get('smoke_test'):
        raise ValueError('Only complete, explicitly label-independent CARROT preparation can be reused')
    selected = [json.loads(s) for s in (old/'selected.jsonl').read_text().splitlines()]
    by_id = {r['id']: r for r in selected}
    with Path(args.csv).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if len(by_id) != len(selected) or len(rows) != len(selected) or len({r['id'] for r in rows}) != len(rows) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('Duplicate/missing/changed IDs: full preparation is required')
    old_labels = {k:r['label'] for k,r in by_id.items()}
    proof = restored_hash(rows, fields, old_labels)
    if proof != meta['csv_sha256']:
        raise ValueError('Old-source reconstruction hash mismatch: text/order/dialect changed; full preparation required')
    splits = split_records(args)
    output = empty_output(args.output_dir)
    changed = sum(LABEL_MAP[r['label']] != old_labels[r['id']] for r in rows)
    provenance = dict(old_csv_sha256=meta['csv_sha256'], restored_csv_sha256=proof,
        old_prepared_sha256=sha256(old/'selected.jsonl'), label_changes=changed,
        verification='restore old labels; UTF-8 BOM; LF; QUOTE_MINIMAL; exact SHA256 match',
        selection_reused=True, texts_unchanged=True)
    meta.update(csv=str(args.csv), csv_sha256=sha256(args.csv), seed=args.seed,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        output_dir=str(output), split_ids={k:[r.id for r in rs] for k,rs in splits.items()},
        reuse_provenance=provenance)
    shutil.copytree(old/'tokenizer', output/'tokenizer')
    with (output/'selected.jsonl.partial').open('x', encoding='utf-8') as handle:
        for split, records in splits.items():
            for record in records:
                row = dict(by_id[record.id], label=record.label, split=split,
                    text_sha256=hashlib.sha256(record.raw_text.encode()).hexdigest())
                if not row['token_ids'] or len(row['token_ids']) > meta['classifier_max_length']:
                    raise ValueError('Invalid cached selection budget')
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    (output/'selected.jsonl.partial').rename(output/'selected.jsonl')
    meta['selected_sha256'] = sha256(output/'selected.jsonl')
    write_json(output/'prepare_config.json', meta)
    print(json.dumps(dict(output=str(output), provenance=provenance,
        splits={k:dict(samples=len(rs), positive=sum(r.label for r in rs)) for k,rs in splits.items()})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    from .cli import source_args
    source_args(parser)
    refresh(parser.parse_args())
