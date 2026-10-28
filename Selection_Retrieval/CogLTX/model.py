"""User-requested CogLTX adaptation: input cap -> blocks -> judge -> concatenate.

The judge retains train-only block-erasure supervision. Selection uses one
scoring pass over all blocks, with no memory replay or four-buffer expansion.
The downstream classifier is BGE-M3, with a separate default 4096-token budget.
"""
from common.data import sample_unit
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig

from table_baselines.budget import Selection
from table_baselines.input_budget import recent_input, LENGTH_SCHEMA
from table_baselines.long_context import load_classifier, resolve_amp, autocast_context
from table_baselines.runtime import (empty_output, set_seed, sha256, split_records, validate_length,
    write_json, save_evaluation, tokenizer_fingerprint, save_model)
from common.metrics import compute_binary_metrics, find_best_f1_threshold
from common.data import load_records

PIPELINE = 'rank_blocks_v1'


def make_blocks(text, tokenizer, block_size, max_length):
    if block_size < 1 or block_size+3 > max_length:
        raise ValueError('block_size must be positive and leave room for block SEP and encoder special tokens')
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [ids[i:i+block_size]+[tokenizer.sep_token_id] for i in range(0,len(ids),block_size)] or [[tokenizer.sep_token_id]]


def pack_blocks(blocks, indices, tokenizer, device):
    ids = [tokenizer.cls_token_id]
    spans = []
    for i in sorted(indices):
        start = len(ids)
        ids.extend(blocks[i])
        spans.append((i,start,len(ids)))
    ids.append(tokenizer.sep_token_id)
    batch = {'input_ids':torch.tensor([ids],dtype=torch.long,device=device),
             'attention_mask':torch.ones((1,len(ids)),dtype=torch.long,device=device)}
    return batch,spans


def fit(blocks, order, capacity, initial=()):
    kept = list(initial)
    used = sum(len(blocks[i]) for i in kept)
    for i in order:
        if i not in kept and used+len(blocks[i])<=capacity:
            kept.append(i)
            used += len(blocks[i])
    return sorted(kept)


def dual_blocks(text, selector_tokenizer, classifier_tokenizer, block_size, selector_max_length, max_length):
    """同一原文块保留两份 token IDs，绝不把 BGE IDs 输入内部 BERT。"""
    make_blocks('', selector_tokenizer, block_size, selector_max_length)
    encoded = selector_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    selected, classified = [], []
    for start in range(0, len(ids), block_size):
        end = min(start + block_size, len(ids))
        fragment = text[offsets[start][0]:offsets[end-1][1]]
        selected.append(ids[start:end] + [selector_tokenizer.sep_token_id])
        other = classifier_tokenizer.encode(fragment, add_special_tokens=False)
        classified.append(other + [classifier_tokenizer.sep_token_id])
        if len(classified[-1]) > max_length - 2:
            raise ValueError('A CogLTX block exceeds classifier budget; reduce --block-size')
    if not selected:
        selected = [[selector_tokenizer.sep_token_id]]
        classified = [[classifier_tokenizer.sep_token_id]]
    return selected, classified


def selector_windows(blocks, indices, max_length):
    """选块器按自身 512 上限分批打分；最终分类器的预算可以为 8192。"""
    window, used = [], 2
    for i in sorted(indices):
        if len(blocks[i]) + 2 > max_length:
            raise ValueError('An introspector block exceeds --selector-max-length')
        if used + len(blocks[i]) > max_length:
            yield window
            window, used = [], 2
        window.append(i)
        used += len(blocks[i])
    if window:
        yield window


@torch.inference_mode()
def rank_blocks(judge, blocks, final_blocks, selector_tokenizer, classifier_max_length,
                device, amp='none', top_k=None):
    """每块单独由judge打分，高分优先装入预算，最终恢复原文顺序。"""
    judge.eval()
    scores = []
    for i in range(len(blocks)):
        batch, spans = pack_blocks(blocks, [i], selector_tokenizer, device)
        _, start, end = spans[0]
        with autocast_context(device, amp):
            logits = judge(**batch).logits[0, start:end, 0]
            scores.append(float(torch.sigmoid(logits).float().mean()))
    order = sorted(range(len(blocks)), key=lambda i: (-scores[i], i))
    if top_k is not None:
        order = order[:top_k]
    return fit(final_blocks, order, classifier_max_length-2), scores


def sample_blocks(text, tokenizer, selector_tokenizer, config):
    # max_length只限制原始完整样本；不会拿它与分类器的位置上限比较。
    text, info = recent_input(text, tokenizer, config['max_length'])
    blocks, final_blocks = dual_blocks(text, selector_tokenizer, tokenizer,
        config['block_size'], config['selector_max_length'], config['classifier_max_length'])
    return blocks, final_blocks, info


def check_config(config):
    if config.get('length_schema') != LENGTH_SCHEMA or config.get('pipeline') != PIPELINE:
        raise ValueError('Legacy CogLTX checkpoint uses memory replay/old length semantics; retrain with the new two-budget pipeline')


class CogSelector:
    def __init__(self, checkpoint, budget, device='cpu', input_max_length=None):
        if checkpoint is None:
            raise ValueError('CogLTX requires a trained judge checkpoint')
        self.path = Path(checkpoint)
        self.config = json.loads((self.path/'run_config.json').read_text())
        check_config(self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path/'tokenizer', local_files_only=True)
        if tokenizer_fingerprint(budget.tokenizer) != tokenizer_fingerprint(self.tokenizer):
            raise ValueError('CogLTX downstream tokenizer differs from the trained classifier')
        self.config['classifier_max_length'] = budget.max_length
        if input_max_length is not None:
            self.config['max_length'] = input_max_length
        validate_length(self.tokenizer, AutoConfig.from_pretrained(self.path/'classifier'),
                        budget.max_length)
        self.selector_tokenizer = AutoTokenizer.from_pretrained(self.path/'selector_tokenizer', local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.path/'introspector',
            local_files_only=True).to(device).eval()
        self.device, self.amp = device, resolve_amp('auto', device)
        validate_length(self.selector_tokenizer, self.model.config, self.config['selector_max_length'], name='selector_max_length')

    def select(self, text):
        blocks, final_blocks, info = sample_blocks(text, self.tokenizer, self.selector_tokenizer, self.config)
        indices, scores = rank_blocks(self.model, blocks, final_blocks, self.selector_tokenizer,
            self.config['classifier_max_length'], self.device, self.amp, self.config.get('top_k'))
        batch, _ = pack_blocks(final_blocks, indices, self.tokenizer, 'cpu')
        info.update(block_ranking=True, memory_replay=False)
        return Selection(batch['input_ids'][0].tolist(), indices, scores, info)


def train(args):
    """Default entry uses guarded joint training, including head warmup."""
    from table_baselines.stable_cogltx import train as stable_train
    return stable_train(args)


def _train_legacy(args):
    set_seed(args.seed)
    args.amp = resolve_amp(args.amp, args.device)
    output = empty_output(args.output_dir)
    splits = split_records(args)
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_model)
    selector_tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    reasoner = load_classifier(args.classifier_model, num_labels=2).to(args.device)
    judge = AutoModelForTokenClassification.from_pretrained(args.encoder, num_labels=1).to(args.device)
    # 两个模型分别校验自身预算；原文max_length可以大于二者的位置上限。
    validate_length(tokenizer, reasoner.config, args.classifier_max_length)
    validate_length(selector_tokenizer, judge.config, args.selector_max_length, name='selector_max_length')
    make_blocks('', selector_tokenizer, args.block_size, args.selector_max_length)
    if args.gradient_checkpointing:
        reasoner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        judge.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    ropt = torch.optim.AdamW(reasoner.parameters(), lr=args.lr, foreach=False)
    jopt = torch.optim.AdamW(judge.parameters(), lr=args.lr, foreach=False)
    rscaler = torch.amp.GradScaler('cuda', enabled=args.amp=='fp16')
    jscaler = torch.amp.GradScaler('cuda', enabled=args.amp=='fp16')
    epochs = 1 if args.smoke_test else args.epochs
    config = {key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items()}
    config.update(joint_cogltx=True, method='cogltx', pipeline=PIPELINE, length_schema=LENGTH_SCHEMA,
        sample_unit=sample_unit(args.csv), turn_label_used=False, input_truncation='recent',
        selection_uses_gold_labels=False, epochs=epochs, csv_sha256=sha256(args.csv),
        split_ids={k:[r.id for r in rows] for k,rows in splits.items()})
    tokenizer.save_pretrained(output/'tokenizer')
    selector_tokenizer.save_pretrained(output/'selector_tokenizer')
    relevance = {}
    best = -1.
    for epoch in range(epochs):
        records = list(splits['train'])
        random.shuffle(records)
        total = 0.
        for number, record in enumerate(records, 1):
            blocks, final_blocks, _ = sample_blocks(record.raw_text, tokenizer, selector_tokenizer, config)
            # 选块阶段不传入真实标签，训练和推理使用同一个judge排序流程。
            indices, _ = rank_blocks(judge, blocks, final_blocks, selector_tokenizer,
                args.classifier_max_length, args.device, args.amp, args.top_k)
            batch, spans = pack_blocks(final_blocks, indices, tokenizer, args.device)
            target = torch.tensor([record.label], device=args.device)
            reasoner.train()
            ropt.zero_grad(set_to_none=True)
            with autocast_context(args.device, args.amp):
                loss = reasoner(**batch, labels=target).loss
            rscaler.scale(loss).backward()
            rscaler.unscale_(ropt)
            torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.)
            rscaler.step(ropt)
            rscaler.update()
            ropt.zero_grad(set_to_none=True)
            total += loss.item()
            # 仅训练样本使用“删除该块后分类损失如何变化”产生judge监督。
            # 不把整通电话风险标签直接复制成每个块的标签。
            rel = relevance.setdefault(record.id, [0]*len(blocks))
            reasoner.eval()
            with torch.inference_mode(), autocast_context(args.device, args.amp):
                base_loss = reasoner(**batch, labels=target).loss.item()
                for i, start, end in spans:
                    changed = {key:value.clone() for key,value in batch.items()}
                    changed['attention_mask'][:, start:end] = 0
                    delta = reasoner(**changed, labels=target).loss.item() - base_loss
                    if delta >= args.levelup_threshold:
                        rel[i] = min(2, rel[i]+1)
                    elif delta <= args.leveldown_threshold:
                        rel[i] = max(-1, rel[i]-1)
            # 只监督已评估过的块，未选块不被擅自贴成负例。
            judge.train()
            jopt.zero_grad(set_to_none=True)
            tokens = sum(len(blocks[i]) for i in indices)
            for i in indices:
                jbatch, jspans = pack_blocks(blocks, [i], selector_tokenizer, args.device)
                _, start, end = jspans[0]
                with autocast_context(args.device, args.amp):
                    logits = judge(**jbatch).logits[0, start:end, 0]
                    labels = torch.full_like(logits, float(rel[i]>0))
                    jloss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels, reduction='sum')/tokens
                jscaler.scale(jloss).backward()
            jscaler.unscale_(jopt)
            torch.nn.utils.clip_grad_norm_(judge.parameters(), 1.)
            jscaler.step(jopt)
            jscaler.update()
            jopt.zero_grad(set_to_none=True)
            if number % 100 == 0:
                print(f'cogltx epoch={epoch+1} records={number}/{len(records)}', flush=True)
        labels, probs, _ = predict_records(reasoner, judge, splits['val'], tokenizer,
                                           selector_tokenizer, config, args.device, args.amp)
        score = compute_binary_metrics(labels, probs)['auprc']
        print(f'cogltx epoch={epoch+1} total_loss={total:.6f} val_auprc={score:.6f}', flush=True)
        if score > best:
            best = score
            save_model(reasoner, output/'classifier')
            save_model(judge, output/'introspector')
    del reasoner, judge, ropt, jopt, loss, jloss
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    reasoner = load_classifier(output/'classifier', local_files_only=True).to(args.device)
    judge = AutoModelForTokenClassification.from_pretrained(output/'introspector', local_files_only=True).to(args.device)
    labels, probs, _ = predict_records(reasoner, judge, splits['val'], tokenizer,
                                       selector_tokenizer, config, args.device, args.amp)
    threshold = find_best_f1_threshold(labels, probs)
    labels, probs, rows = predict_records(reasoner, judge, splits['test'], tokenizer,
                                          selector_tokenizer, config, args.device, args.amp)
    save_evaluation(output, 'test', rows, labels, probs, threshold, args.smoke_test)
    config['threshold'] = threshold
    write_json(output/'run_config.json', config)


@torch.inference_mode()
def predict_records(reasoner, judge, records, tokenizer, selector_tokenizer, config, device, amp='none'):
    reasoner.eval()
    judge.eval()
    if config.get('stable_training'):
        from table_baselines.stable_cogltx import select_rows
        from table_baselines.stable_training import probabilities as batch_probabilities
        reasoner._stable_amp = judge._stable_amp = amp
        items = []
        for record in records:
            blocks, final_blocks, info = sample_blocks(record.raw_text, tokenizer, selector_tokenizer, config)
            items.append(dict(id=record.id, label=record.label, blocks=blocks, final_blocks=final_blocks, info=info))
        rows = select_rows(judge, items, tokenizer, selector_tokenizer, config, device)
        probs = batch_probabilities(reasoner, rows, tokenizer, device, amp=amp)
        return np.asarray([r.label for r in records]), probs, rows
    probabilities, rows = [], []
    for record in records:
        blocks, final_blocks, info = sample_blocks(record.raw_text, tokenizer, selector_tokenizer, config)
        indices, scores = rank_blocks(judge, blocks, final_blocks, selector_tokenizer,
            config['classifier_max_length'], device, amp, config.get('top_k'))
        batch, _ = pack_blocks(final_blocks, indices, tokenizer, device)
        with autocast_context(device, amp):
            probabilities.append(float(reasoner(**batch).logits.float().softmax(-1)[0, 1]))
        rows.append(dict(id=record.id, selected_indices=indices, scores=scores, **info))
    return np.asarray([r.label for r in records]), np.asarray(probabilities), rows


def infer(args):
    output = Path(args.model_dir)
    config = json.loads((output/'run_config.json').read_text())
    requested = config.get('amp','auto') if args.amp=='auto' and args.device=='cuda' else args.amp
    args.amp = resolve_amp(requested, args.device)
    check_config(config)
    if args.split != 'all' and sha256(args.csv) != config['csv_sha256']:
        raise ValueError('CSV differs from saved training data')
    records = load_records(args.csv)
    if args.split != 'all':
        by_id = {r.id:r for r in records}
        records = [by_id[i] for i in config['split_ids'][args.split]]
    tokenizer = AutoTokenizer.from_pretrained(output/'tokenizer', local_files_only=True)
    selector_tokenizer = AutoTokenizer.from_pretrained(output/'selector_tokenizer', local_files_only=True)
    reasoner = load_classifier(output/'classifier', local_files_only=True).to(args.device)
    judge = AutoModelForTokenClassification.from_pretrained(output/'introspector', local_files_only=True).to(args.device)
    validate_length(tokenizer, reasoner.config, config['classifier_max_length'])
    validate_length(selector_tokenizer, judge.config, config['selector_max_length'], name='selector_max_length')
    labels, probs, rows = predict_records(reasoner, judge, records, tokenizer,
                                          selector_tokenizer, config, args.device, args.amp)
    save_evaluation(output, f'infer_{args.split}', rows, labels, probs,
                    config['threshold'], config['smoke_test'])
