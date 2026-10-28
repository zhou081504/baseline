"""Reproducible classification runtime, with dynamic padding and bounded batches."""
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from .long_context import load_classifier, resolve_amp, autocast_context
from .input_budget import LENGTH_SCHEMA

from common.data import load_records, make_splits
from common.metrics import compute_binary_metrics, find_best_f1_threshold, save_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer):
    payload = {'vocab':tokenizer.get_vocab(), 'special_tokens':tokenizer.special_tokens_map}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def save_model(model, path):
    # Some pretrained checkpoints have transposed tensor strides.
    # safetensors requires contiguous storage even after successful training.
    model.save_pretrained(path, state_dict={key:value.contiguous() for key,value in model.state_dict().items()})


def empty_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output {path} is not empty; use a new directory to preserve previous work')
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_records(args):
    records = load_records(args.csv)
    if any(not r.id for r in records) or len({r.id for r in records}) != len(records):
        raise ValueError('Data source requires unique, nonempty IDs')
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(not 0 < x < 1 for x in ratios) or abs(sum(ratios)-1) > 1e-6:
        raise ValueError('Split ratios must be in (0,1) and sum to 1')
    splits = make_splits(records, train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                         test_ratio=args.test_ratio, seed=args.seed)
    if args.smoke_test:
        splits = {name:sum(([r for r in rows if r.label == label][:2] for label in (0,1)), [])
                  for name, rows in splits.items()}
        if any({r.label for r in rows} != {0,1} for rows in splits.values()):
            raise ValueError('Smoke test needs both classes in every split')
    return splits


def validate_length(tokenizer, config, length, name='classifier_max_length'):
    if length <= tokenizer.num_special_tokens_to_add(pair=False):
        raise ValueError(f'{name} must leave room for content and special tokens')
    positions = getattr(config, 'max_position_embeddings', length)
    if getattr(config, 'model_type', '') in ('roberta', 'xlm-roberta'):
        positions -= (getattr(config, 'pad_token_id', 1) or 0) + 1
    limit = min(tokenizer.model_max_length, positions)
    if length > limit:
        raise ValueError(f'{name}={length} exceeds encoder limit {limit}; no automatic position extension is applied')


class PackedDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows, self.tokenizer = rows, tokenizer
        self.labels = torch.tensor([r['label'] for r in rows], dtype=torch.float)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {'input_ids':row['token_ids'], 'attention_mask':[1]*len(row['token_ids'])}, row['label']

    def collate(self, batch):
        features, labels = zip(*batch)
        return self.tokenizer.pad(list(features), padding=True, return_tensors='pt'), torch.tensor(labels, dtype=torch.float)


def loader(rows, tokenizer, batch_size, shuffle=False):
    dataset = PackedDataset(rows, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=dataset.collate)


def binary_logits(model, batch):
    # Subtract and apply sigmoid in FP32: BF16 rounds close binary scores.
    logits = model(**batch).logits.float()
    return logits[:,1]-logits[:,0]


@torch.inference_mode()
def predict(model, data_loader, device, amp='none'):
    model.eval()
    labels, probs = [], []
    for batch, target in data_loader:
        batch = {key:value.to(device) for key,value in batch.items()}
        with autocast_context(device, amp):
            probability = torch.sigmoid(binary_logits(model, batch)).float().cpu().numpy()
        labels.extend(target.tolist())
        probs.extend(probability.tolist())
    return np.asarray(labels), np.asarray(probs)


def save_evaluation(output, prefix, rows, labels, probabilities, threshold, smoke_test=False):
    metrics = compute_binary_metrics(labels, probabilities, threshold)
    metrics['smoke_test'] = bool(smoke_test)
    metrics['samples'] = len(labels)
    save_metrics(metrics, Path(output)/f'{prefix}_metrics.json')
    with (Path(output)/f'{prefix}_predictions.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id','gold','probability','prediction'])
        for row, label, prob in zip(rows, labels, probabilities):
            writer.writerow([row['id'],int(label),float(prob),int(prob>=threshold)])
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


def train_classifier(args):
    """All selection/compression methods use the guarded stable trainer."""
    from .stable_training import train
    return train(args)


def _train_classifier_legacy(args):
    set_seed(args.seed)
    prepared = Path(args.prepared)
    meta = json.loads((prepared/'prepare_config.json').read_text(encoding='utf-8'))
    if meta.get('length_schema') != LENGTH_SCHEMA:
        raise ValueError('Old prepared data uses max_length as classifier length; rerun prepare with --max_length and --classifier_max_length')
    if meta['method'] == 'cogltx':
        raise ValueError('Use train-cogltx: its jointly trained reasoner must not be replaced by a new classifier')
    rows = [json.loads(line) for line in (prepared/'selected.jsonl').read_text(encoding='utf-8').splitlines()]
    splits = {name:[r for r in rows if r['split'] == name] for name in ('train','val','test')}
    if any(not v for v in splits.values()):
        raise ValueError('Prepared data must contain nonempty train/val/test splits')
    output = empty_output(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(prepared/'tokenizer', local_files_only=True)
    model_tokenizer = AutoTokenizer.from_pretrained(args.classifier_model)
    if tokenizer_fingerprint(tokenizer) != tokenizer_fingerprint(model_tokenizer):
        raise ValueError('Classifier tokenizer differs from the tokenizer used for selection/budgeting; prepare data again with the same model')
    model = load_classifier(args.classifier_model, num_labels=2).to(args.device)
    validate_length(tokenizer, model.config, meta['classifier_max_length'])
    if any(not r['token_ids'] or len(r['token_ids'])>meta['classifier_max_length'] for r in rows):
        raise ValueError('Prepared input violates classifier_max_length')
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    args.amp = resolve_amp(args.amp, args.device)
    loaders = {name:loader(items, tokenizer, args.batch_size, name=='train') for name,items in splits.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    positive = sum(r['label'] for r in splits['train'])
    if positive == 0 or positive == len(splits['train']):
        raise ValueError('Training split must contain both labels')
    pos_weight = min(20.0, max(1.0, (len(splits['train'])-positive)/positive))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=args.device))
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp == 'fp16')
    best = -1.0
    epochs = 1 if meta['smoke_test'] else args.epochs
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = len(loaders['train'])
        for step, (batch, labels) in enumerate(loaders['train']):
            batch = {key:value.to(args.device) for key,value in batch.items()}
            labels = labels.to(args.device)
            window_start = step//args.gradient_accumulation_steps*args.gradient_accumulation_steps
            # Weight each micro-batch by its actual number of samples; the last
            # partial batch/window must match a single equivalent large batch.
            window_end = min(window_start+args.gradient_accumulation_steps, batches)
            window_samples = min(window_end*args.batch_size, len(splits['train']))-window_start*args.batch_size
            with autocast_context(args.device, args.amp):
                raw_loss = criterion(binary_logits(model,batch), labels)
                loss = raw_loss*len(labels)/window_samples
            scaler.scale(loss).backward()
            if step+1 == window_end:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += raw_loss.item()*len(labels)
            if (step + 1) % 100 == 0:
                print(f'epoch={epoch+1} batch={step+1}/{batches}', flush=True)
        val_labels, val_probs = predict(model, loaders['val'], args.device, args.amp)
        score = compute_binary_metrics(val_labels,val_probs)['auprc']
        print(f'epoch={epoch+1} loss={total_loss/len(splits["train"]):.6f} val_auprc={score:.6f}', flush=True)
        if score > best:
            best = score
            save_model(model,output/'classifier')
    del model, optimizer
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    model = load_classifier(output/'classifier', local_files_only=True).to(args.device)
    tokenizer.save_pretrained(output/'tokenizer')
    labels, probs = predict(model, loaders['val'], args.device, args.amp)
    # 只用验证集选择阈值，随后固定该阈值计算测试集指标。
    threshold = find_best_f1_threshold(labels, probs)
    labels, probs = predict(model, loaders['test'], args.device, args.amp)
    metrics = save_evaluation(output,'test',splits['test'],labels,probs,threshold,meta['smoke_test'])
    config = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(preparation=meta, prepared_sha256=sha256(prepared/'selected.jsonl'), threshold=threshold,
                  max_length=meta['max_length'], classifier_max_length=meta['classifier_max_length'],
                  length_schema=LENGTH_SCHEMA, input_truncation=meta.get('input_truncation', 'recent'), method=meta['method'], epochs=epochs,
                  pos_weight=pos_weight, tokenizer_fingerprint=tokenizer_fingerprint(tokenizer))
    write_json(output/'run_config.json', config)
    return metrics


def evaluate_classifier(args):
    output, prepared = Path(args.model_dir), Path(args.prepared)
    config = json.loads((output/'run_config.json').read_text(encoding='utf-8'))
    requested = config.get('amp','auto') if args.amp=='auto' and args.device=='cuda' else args.amp
    args.amp = resolve_amp(requested, args.device)
    if config.get('length_schema') != LENGTH_SCHEMA:
        raise ValueError('Legacy checkpoint uses the old length semantics; use a newly prepared/trained model')
    if config.get('joint_cogltx'):
        raise ValueError('Use infer-cogltx for a jointly trained CogLTX checkpoint')
    if sha256(prepared/'selected.jsonl') != config['prepared_sha256']:
        raise ValueError('Prepared dataset differs from the one saved with this checkpoint')
    rows = [json.loads(line) for line in (prepared/'selected.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = [r for r in rows if args.split == 'all' or r['split'] == args.split]
    tokenizer = AutoTokenizer.from_pretrained(output/'tokenizer', local_files_only=True)
    model = load_classifier(output/'classifier', local_files_only=True).to(args.device)
    if config.get('stable_training'):
        from .stable_training import probabilities
        labels = np.asarray([r['label'] for r in rows])
        probs = probabilities(model,rows,tokenizer,args.device,amp=args.amp,max_batch=args.batch_size)
    else:
        labels, probs = predict(model,loader(rows,tokenizer,args.batch_size),args.device,args.amp)
    save_evaluation(output,f'infer_{args.split}',rows,labels,probs,config['threshold'],config['preparation']['smoke_test'])
