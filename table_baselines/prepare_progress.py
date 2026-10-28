"""Validate a preparation prefix before append; never silently skip or replace rows."""
from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def preparation_lock(output):
    # A sibling lock does not make a fresh output directory nonempty.
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + '.prepare.lock').open('a+b') as handle:
        if os.name == 'posix':
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another preparation is using this output directory') from None
        else:
            import msvcrt
            handle.write(b'0'); handle.flush(); handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise ValueError('Another preparation is using this output directory') from None
        yield


def validate_resume(output, config, expected):
    output = Path(output)
    if (output / 'selected.jsonl').exists():
        raise ValueError('Preparation already complete; refusing to overwrite selected.jsonl')
    saved = json.loads((output / 'prepare_config.json').read_text(encoding='utf-8'))
    ignored = {'resume'}
    changed = sorted(k for k in set(saved) | set(config)
                     if k not in ignored and saved.get(k) != config.get(k))
    if changed:
        raise ValueError('Resume configuration/source mismatch: ' + ', '.join(changed))
    count = 0
    with (output / 'selected.jsonl.partial').open('rb') as handle:
        for line in handle:
            if not line.endswith(b'\n'):
                raise ValueError('Partial file has an incomplete final line; preserve and inspect before repair')
            try:
                row = json.loads(line)
                split, record = expected[count]
                tokens = row['token_ids']
                correct = (row['id'] == record.id and row['label'] == record.label
                           and row['split'] == split and isinstance(tokens, list)
                           and 0 < len(tokens) <= config['classifier_max_length']
                           and all(type(t) is int and t >= 0 for t in tokens))
            except (ValueError, KeyError, TypeError, IndexError):
                correct = False
            if not correct:
                raise ValueError(f'Invalid or mismatched prepared row {count + 1}; file left unchanged')
            count += 1
    return count
