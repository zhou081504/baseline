"""Stable binary training: cached head warmup, differential LR and larger batches."""
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from common.metrics import compute_binary_metrics, find_best_f1_threshold
from .long_context import load_classifier, autocast_context, resolve_amp
from .runtime import (set_seed, empty_output, sha256, tokenizer_fingerprint,
    validate_length, write_json, save_model, save_evaluation)
from .input_budget import LENGTH_SCHEMA


def event(kind, **values):
    print(json.dumps(dict(event=kind, **values), ensure_ascii=False), flush=True)


def validate_training_args(args):
    for name in ('epochs', 'head_epochs', 'effective_batch_size', 'train_token_budget'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('encoder_lr', 'head_lr', 'finetune_head_lr'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f'{name} must be finite and positive')
    args.amp = resolve_amp(getattr(args, 'amp', 'auto'), args.device)
    if args.amp == 'fp16':
        raise ValueError('Stable training supports bf16 or none; use --amp none on GPUs without BF16')
    args.batch_size = getattr(args, 'batch_size', 16)
    if args.batch_size < 1:
        raise ValueError('batch_size must be positive')


def model_amp(model, device):
    return getattr(model, '_stable_amp', resolve_amp('auto', device))


def head_features(model, batch):
    out = model.base_model(**batch)
    if model.config.model_type in ('roberta', 'xlm-roberta'):
        return out.last_hidden_state[:, 0]
    if model.config.model_type == 'bert':
        return out.pooler_output
    raise ValueError('Stable head warmup supports BERT, RoBERTa and XLM-R/BGE-M3 classifiers')


def head_logits(model, features):
    if model.config.model_type in ('roberta', 'xlm-roberta'):
        return model.classifier(features[:, None, :]).float()
    return model.classifier(model.dropout(features)).float()


def collapse_reason(stats):
    if stats['unique_probabilities'] < 2:
        return 'constant probabilities'
    if stats['probability_std'] < 1e-6:
        return 'nearly constant probabilities (std < 1e-6)'
    return None


def require_healthy(stats, output, stage):
    reason = collapse_reason(stats)
    if reason:
        write_json(output/'failure.json', dict(stage=stage, reason=reason, diagnostics=stats))
        raise ValueError(f'{stage}: {reason}; refusing to publish an invalid checkpoint')


def validate_prepared(prepared, meta, rows):
    from common.data import load_records
    source = Path(meta['csv'])
    candidates = [source] if source.is_absolute() else [Path.cwd()/source, Path(__file__).resolve().parents[1]/source]
    source = next((p for p in candidates if p.is_file() and sha256(p) == meta['csv_sha256']), None)
    if source is None:
        raise ValueError('Data source changed after preparation or cannot be found; prepare current data again')
    if meta.get('selected_sha256') and sha256(prepared/'selected.jsonl') != meta['selected_sha256']:
        raise ValueError('Prepared selection checksum mismatch')
    original = {r.id:r.label for r in load_records(source)}
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Prepared data has duplicate IDs / overlapping splits')
    if any(r['id'] not in original or r['label'] != original[r['id']] for r in rows):
        raise ValueError('Prepared labels differ from the current data source')
    if not meta.get('smoke_test') and len(rows) != len(original):
        raise ValueError('Full prepared data must cover every sample')
    splits = {s:[r for r in rows if r['split'] == s] for s in ('train','val','test')}
    if sum(map(len,splits.values())) != len(rows) or any(not v for v in splits.values()):
        raise ValueError('Prepared data requires nonempty train/val/test splits')
    if any({r['label'] for r in splits[s]} != {0,1} for s in ('train','val')):
        raise ValueError('Training and validation splits must each contain both labels')
    if meta.get('split_ids') != {s:[r['id'] for r in rs] for s,rs in splits.items()}:
        raise ValueError('Prepared split IDs differ from the manifest')
    return splits


def pad(rows, tokenizer, device):
    return tokenizer.pad([dict(input_ids=r['token_ids'], attention_mask=r.get('attention_mask', [1]*len(r['token_ids'])))
        for r in rows], padding=True, return_tensors='pt').to(device)


def batches(rows, token_budget=8192, max_batch=32):
    """Bound padding, attention memory and batch size; preserve returned indices."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]['token_ids']))
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and end-start < max_batch and (end-start+1)*len(rows[order[end]]['token_ids']) <= token_budget:
            end += 1
        yield order[start:end]
        start = end


@torch.inference_mode()
def probabilities(model, rows, tokenizer, device='cuda', token_budget=8192, amp=None, max_batch=32):
    model.eval()
    result = np.empty(len(rows), dtype=np.float64)
    for ix in batches(rows, token_budget, max_batch=max_batch):
        batch = pad([rows[i] for i in ix], tokenizer, device)
        with autocast_context(device, amp or model_amp(model, device)):
            logits = model(**batch).logits.float()
        result[ix] = (logits[:,1]-logits[:,0]).sigmoid().cpu().numpy()
    return result


def diagnostics(rows, probs, threshold=.5):
    y = np.asarray([r['label'] for r in rows])
    if not np.isfinite(probs).all():
        raise ValueError('Non-finite model probabilities')
    out = compute_binary_metrics(y, probs, threshold)
    out.update(probability_std=float(np.std(probs)), unique_probabilities=int(len(np.unique(probs))))
    return out


def warmup(model, splits, tokenizer, output, device='cuda', epochs=8, lr=1e-3):
    """Fit only the head on original pretrained features; validation selects epoch."""
    model.eval()
    features = {}
    for split in ('train','val'):
        rows = splits[split]
        features[split] = torch.empty((len(rows), model.config.hidden_size), dtype=torch.float32)
        done = 0
        for ix in batches(rows):
            with torch.inference_mode(), autocast_context(device, model_amp(model, device)):
                vec = head_features(model, pad([rows[i] for i in ix],tokenizer,device)).float().cpu()
            features[split][ix] = vec
            done += len(ix)
            if done % 2048 < len(ix): event('head_cache', split=split, done=done, total=len(rows))
        torch.save(dict(ids=[r['id'] for r in rows], features=features[split]), output/f'{split}_cls.pt')
    head = model.classifier
    x = {k:v.to(device) for k,v in features.items()}
    y = {k:torch.tensor([r['label'] for r in splits[k]],dtype=torch.float32,device=device) for k in x}
    positive = float(y['train'].sum())
    if positive == 0 or positive == len(y['train']):
        raise ValueError('Training split must contain both labels')
    weight = min(20., (len(y['train'])-positive)/positive)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=device))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, foreach=False)
    best, state, history = -1., None, []
    for epoch in range(epochs):
        head.train()
        if hasattr(model, 'dropout'): model.dropout.train()
        total=0.
        for ix in torch.randperm(len(x['train']),device=device).split(64):
            opt.zero_grad(set_to_none=True)
            z=head_logits(model,x['train'][ix])
            loss=criterion(z[:,1]-z[:,0],y['train'][ix]); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            opt.step(); total+=loss.item()*len(ix)
        head.eval()
        if hasattr(model, 'dropout'): model.dropout.eval()
        with torch.no_grad():
            z=torch.cat([head_logits(model,a) for a in x['val'].split(256)])
            p=(z[:,1]-z[:,0]).sigmoid().cpu().numpy()
        metrics=diagnostics(splits['val'],p)
        history.append(dict(stage='head',epoch=epoch+1,loss=total/len(x['train']),**metrics))
        event('head_epoch',**history[-1])
        write_json(output/'warmup_history.json',history)
        if collapse_reason(metrics) is None and metrics['auprc']>best:
            best=metrics['auprc']; state=copy.deepcopy(head.state_dict())
    if state is None:
        require_healthy(metrics, output, 'head_warmup')
        raise ValueError('No valid warmup checkpoint')
    head.load_state_dict(state)
    write_json(output/'warmup_history.json',history)
    return best, weight


def optimizer_for(model, encoder_lr, head_lr):
    head_ids={id(p) for p in model.classifier.parameters()}
    return torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in head_ids],lr=encoder_lr),
        dict(params=list(model.classifier.parameters()),lr=head_lr)],foreach=False)


def scheduler_for(opt, steps):
    warm=max(1,math.ceil(.1*steps))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda n: (n+1)/warm if n<warm else max(0.,(steps-n)/max(1,steps-warm)))


def train_window(model, rows, tokenizer, optimizer, criterion, device='cuda', token_budget=2048, max_batch=16):
    model.train(); optimizer.zero_grad(set_to_none=True); total=0.
    for ix in batches(rows,token_budget,max_batch=max_batch):
        batch=pad([rows[i] for i in ix],tokenizer,device)
        y=torch.tensor([rows[i]['label'] for i in ix],device=device,dtype=torch.float32)
        with autocast_context(device,model_amp(model,device)):
            logits=model(**batch).logits.float()
            loss=criterion(logits[:,1]-logits[:,0],y)
        (loss*len(ix)/len(rows)).backward(); total+=loss.item()*len(ix)
    norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
    if not math.isfinite(norm):
        raise ValueError('Non-finite gradient; optimizer step aborted')
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return total/len(rows),norm


def train(args):
    validate_training_args(args)
    set_seed(args.seed); torch.set_num_threads(args.num_threads)
    args.prepared=Path(args.prepared).resolve()
    args.output_dir=Path(args.output_dir).resolve()
    prepared=args.prepared; output=empty_output(args.output_dir)
    meta=json.loads((prepared/'prepare_config.json').read_text())
    if meta.get('length_schema')!=LENGTH_SCHEMA or meta['method']=='cogltx':
        raise ValueError('Use current prepared data and the joint CogLTX trainer for CogLTX')
    rows=[json.loads(s) for s in (prepared/'selected.jsonl').read_text().splitlines()]
    splits=validate_prepared(prepared,meta,rows)
    tok=AutoTokenizer.from_pretrained(prepared/'tokenizer',local_files_only=True)
    if tokenizer_fingerprint(tok)!=tokenizer_fingerprint(AutoTokenizer.from_pretrained(args.classifier_model,local_files_only=True)):
        raise ValueError('Tokenizer mismatch')
    model=load_classifier(args.classifier_model,local_files_only=True).to(args.device)
    model._stable_amp=args.amp
    validate_length(tok,model.config,meta['classifier_max_length'])
    if any(not r['token_ids'] or len(r['token_ids'])>meta['classifier_max_length'] for r in rows):
        raise ValueError('Prepared budget violation')
    best,weight=warmup(model,splits,tok,output,args.device,args.head_epochs,args.head_lr)
    warm_stats=diagnostics(splits['val'],probabilities(model,splits['val'],tok,args.device))
    require_healthy(warm_stats,output,'warmup_model_validation')
    best=warm_stats['auprc']
    save_model(model,output/'classifier'); best_stage='head_warmup'
    tok.save_pretrained(output/'tokenizer')
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    opt=optimizer_for(model,args.encoder_lr,args.finetune_head_lr)
    schedule=scheduler_for(opt,args.epochs*math.ceil(len(splits['train'])/args.effective_batch_size))
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=args.device))
    history=[]; stopped_for_collapse=False
    for epoch in range(args.epochs):
        order=list(splits['train']);random.shuffle(order);loss_sum=0.
        for start in range(0,len(order),args.effective_batch_size):
            window=order[start:start+args.effective_batch_size]
            loss,norm=train_window(model,window,tok,opt,criterion,args.device,args.train_token_budget,args.batch_size)
            schedule.step();loss_sum+=loss*len(window)
            if start//args.effective_batch_size%50==0:
                event('finetune',epoch=epoch+1,done=start+len(window),total=len(order),loss=loss,gradient_norm=norm)
        p=probabilities(model,splits['val'],tok,args.device)
        stats=diagnostics(splits['val'],p)
        history.append(dict(epoch=epoch+1,loss=loss_sum/len(order),**stats));event('validation',**history[-1])
        write_json(output/'finetune_history.json',history)
        if collapse_reason(stats):
            stopped_for_collapse=True
            write_json(output/'collapse_guard.json',dict(epoch=epoch+1,reason=collapse_reason(stats),action='restore best healthy checkpoint'))
            event('collapse_guard',action='keep previous healthy checkpoint; stop fine tuning');break
        if stats['auprc']>best:
            best=stats['auprc'];best_stage=f'finetune_epoch_{epoch+1}';save_model(model,output/'classifier')
    del model,opt,schedule
    if args.device=='cuda':torch.cuda.empty_cache()
    model=load_classifier(output/'classifier',local_files_only=True).to(args.device)
    model._stable_amp=args.amp
    p=probabilities(model,splits['val'],tok,args.device)
    require_healthy(diagnostics(splits['val'],p),output,'saved_model_validation')
    threshold=find_best_f1_threshold([r['label'] for r in splits['val']],p)
    save_evaluation(output,'val',splits['val'],[r['label'] for r in splits['val']],p,threshold,meta['smoke_test'])
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(stable_training=True,best_stage=best_stage,best_val_auprc=best,threshold=threshold,
        preparation=meta,prepared_sha256=sha256(prepared/'selected.jsonl'),length_schema=LENGTH_SCHEMA,
        method=meta['method'],max_length=meta['max_length'],classifier_max_length=meta['classifier_max_length'],
        pos_weight=weight,input_truncation=meta.get('input_truncation', 'recent'),tokenizer_fingerprint=tokenizer_fingerprint(tok),
        epochs_completed=len(history),stopped_for_collapse=stopped_for_collapse)
    write_json(output/'run_config.json',config)
    p=probabilities(model,splits['test'],tok,args.device)
    save_evaluation(output,'test',splits['test'],[r['label'] for r in splits['test']],p,threshold,meta['smoke_test'])
    require_healthy(diagnostics(splits['test'],p,threshold),output,'test_output_check_no_retuning')
    write_json(output/'completion.json',dict(status='complete',test=diagnostics(splits['test'],p,threshold),best_stage=best_stage))


def add_args(parser):
    parser.add_argument('--head-epochs',type=int,default=8)
    parser.add_argument('--head-lr',type=float,default=1e-3)
    parser.add_argument('--encoder-lr',type=float,default=2e-6)
    parser.add_argument('--finetune-head-lr',type=float,default=2e-5)
    parser.add_argument('--effective-batch-size',type=int,default=64)
    parser.add_argument('--train-token-budget',type=int,default=2048)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--classifier-model',required=True)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num-threads',type=int,default=2)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--batch-size',type=int,default=16,help='maximum physical microbatch; effective batch is separate')
    p.add_argument('--amp',choices=['auto','none','bf16'],default='auto')
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    add_args(p)
    train(p.parse_args())
