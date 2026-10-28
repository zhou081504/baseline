"""Batched CogLTX training with cached tokenization and train-only deletion targets."""
from common.data import sample_unit
import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

from Selection_Retrieval.CogLTX.model import sample_blocks, fit, PIPELINE
from .input_budget import LENGTH_SCHEMA
from .long_context import load_classifier, autocast_context
from .runtime import (empty_output, set_seed, split_records, sha256, write_json,
    save_model, save_evaluation, tokenizer_fingerprint, validate_length)
from .stable_training import (event, batches, pad, warmup, optimizer_for, scheduler_for,
    train_window, probabilities, diagnostics, add_args, validate_training_args,
    model_amp, collapse_reason, require_healthy)
from common.metrics import find_best_f1_threshold


def packed(blocks, indices, tokenizer):
    ids=[tokenizer.cls_token_id]; spans=[]
    for i in sorted(indices):
        start=len(ids);ids.extend(blocks[i]);spans.append((i,start,len(ids)))
    ids.append(tokenizer.sep_token_id)
    return ids,spans


@torch.inference_mode()
def block_scores(judge, block_rows, tokenizer, device, token_budget=8192):
    judge.eval(); scores=np.empty(len(block_rows),dtype=np.float64)
    for ix in batches(block_rows,token_budget):
        batch=pad([block_rows[i] for i in ix],tokenizer,device)
        with autocast_context(device,model_amp(judge,device)):
            z=judge(**batch).logits[:,:,0].float().sigmoid()
        for j,i in enumerate(ix):
            scores[i]=float(z[j,1:len(block_rows[i]['token_ids'])-1].mean())
    return scores


def select_rows(judge, items, tokenizer, selector_tokenizer, config, device):
    """Never reads labels to rank blocks. All-fit shortcut is algebraically exact."""
    requests=[]; locations=[]; selected={}; capacity=config['classifier_max_length']-2
    for n,item in enumerate(items):
        blocks=item['blocks']; final=item['final_blocks']
        if config.get('top_k') is None and sum(map(len,final))<=capacity:
            selected[n]=list(range(len(final)))
        else:
            for i,block in enumerate(blocks):
                requests.append(dict(token_ids=[selector_tokenizer.cls_token_id]+block+[selector_tokenizer.sep_token_id]))
                locations.append((n,i))
    values=block_scores(judge,requests,selector_tokenizer,device) if requests else []
    score_map={}
    for (n,i),value in zip(locations,values):score_map.setdefault(n,{})[i]=value
    for n,scores in score_map.items():
        order=sorted(scores,key=lambda i:(-scores[i],i))
        if config.get('top_k') is not None:order=order[:config['top_k']]
        selected[n]=fit(items[n]['final_blocks'],order,capacity)
    rows=[]
    for n,item in enumerate(items):
        ids,spans=packed(item['final_blocks'],selected[n],tokenizer)
        rows.append(dict(id=item['id'],label=item['label'],token_ids=ids,
            selected_indices=selected[n],spans=spans,blocks=item['blocks']))
    return rows


def prepare_cache(args, tokenizer, selector_tokenizer, config):
    target=Path(args.cache_dir)
    expected=dict(csv_sha256=sha256(args.csv),tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
        selector_tokenizer_fingerprint=tokenizer_fingerprint(selector_tokenizer),
        max_length=args.max_length,classifier_max_length=args.classifier_max_length,
        selector_max_length=args.selector_max_length,block_size=args.block_size,
        split_ids=config['split_ids'],cache_schema='cog_dual_blocks_v1')
    if (target/'complete.json').exists():
        saved=json.loads((target/'complete.json').read_text())
        if saved['config']!=expected or saved['cache_sha256']!=sha256(target/'blocks.jsonl'):
            raise ValueError('CogLTX token cache provenance mismatch; use a new cache directory')
        items=[json.loads(s) for s in (target/'blocks.jsonl').read_text().splitlines()]
        event('cog_cache_reused',samples=len(items))
        return {s:[r for r in items if r['split']==s] for s in ('train','val','test')}
    target=empty_output(target);splits=split_records(args)
    def convert(pair):
        split,record=pair
        blocks,final,info=sample_blocks(record.raw_text,tokenizer,selector_tokenizer,config)
        return dict(id=record.id,label=record.label,split=split,blocks=blocks,final_blocks=final,info=info)
    pairs=[(s,r) for s,records in splits.items() for r in records]
    items=[];start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool, (target/'blocks.jsonl.partial').open('x',encoding='utf-8') as handle:
        for offset in range(0,len(pairs),512):
            for item in pool.map(convert,pairs[offset:offset+512]):
                items.append(item);handle.write(json.dumps(item,ensure_ascii=False)+'\n')
            if offset%4096==0:event('cog_prepare',done=len(items),total=len(pairs),seconds=round(time.monotonic()-start,1))
    (target/'blocks.jsonl.partial').rename(target/'blocks.jsonl')
    lengths=[sum(map(len,r['final_blocks']))+2 for r in items]
    stats=dict(samples=len(items),length_percentiles=np.percentile(lengths,[0,50,90,95,99,100]).tolist(),
        all_fit=sum(x<=args.classifier_max_length for x in lengths),seconds=time.monotonic()-start)
    write_json(target/'complete.json',dict(config=expected,cache_sha256=sha256(target/'blocks.jsonl'),stats=stats))
    event('cog_prepared',**stats)
    return {s:[r for r in items if r['split']==s] for s in splits}


@torch.inference_mode()
def erasure_losses(reasoner, rows, tokenizer, device, token_budget=8192):
    reasoner.eval();losses=np.empty(len(rows))
    for ix in batches(rows,token_budget):
        batch=pad([rows[i] for i in ix],tokenizer,device)
        y=torch.tensor([rows[i]['label'] for i in ix],device=device)
        with autocast_context(device,model_amp(reasoner,device)):
            z=reasoner(**batch).logits.float()
        losses[ix]=F.cross_entropy(z,y,reduction='none').cpu().numpy()
    return losses


def deletion_targets(reasoner, rows, tokenizer, selector_tokenizer, relevance, config, device):
    variants=[];groups=[]
    for row in rows:
        base=len(variants);variants.append(row);changes=[]
        for i,start,end in row['spans']:
            mask=[1]*len(row['token_ids']);mask[start:end]=[0]*(end-start)
            changes.append((i,len(variants)))
            variants.append(dict(token_ids=row['token_ids'],attention_mask=mask,label=row['label']))
        groups.append((row,base,changes))
    losses=erasure_losses(reasoner,variants,tokenizer,device)
    targets=[];positive=0
    for row,base,changes in groups:
        rel=relevance.setdefault(row['id'],[0]*len(row['blocks']))
        tokens=sum(len(row['blocks'][i]) for i in row['selected_indices'])
        for i,variant in changes:
            delta=losses[variant]-losses[base]
            if delta>=config['levelup_threshold']:rel[i]=min(2,rel[i]+1)
            elif delta<=config['leveldown_threshold']:rel[i]=max(-1,rel[i]-1)
            target=float(rel[i]>0);positive+=int(target)
            targets.append(dict(token_ids=[selector_tokenizer.cls_token_id]+row['blocks'][i]+[selector_tokenizer.sep_token_id],
                target=target,weight=1./(tokens*len(rows))))
    return targets,positive


def train_judge(judge, targets, tokenizer, optimizer, device):
    judge.train();optimizer.zero_grad(set_to_none=True);total=0.
    for ix in batches(targets,4096,max_batch=32):
        batch=pad([targets[i] for i in ix],tokenizer,device)
        with autocast_context(device,model_amp(judge,device)):
            logits=judge(**batch).logits[:,:,0].float()
        loss=logits.sum()*0.
        for j,i in enumerate(ix):
            row=targets[i];z=logits[j,1:len(row['token_ids'])-1]
            loss=loss+F.binary_cross_entropy_with_logits(z,torch.full_like(z,row['target']),reduction='sum')*row['weight']
        loss.backward();total+=loss.item()
    norm=float(torch.nn.utils.clip_grad_norm_(judge.parameters(),1.))
    if not math.isfinite(norm):raise ValueError('Non-finite judge gradient')
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    return total


def train(args):
    validate_training_args(args)
    if any(not math.isfinite(v) or v<=0 for v in (args.judge_encoder_lr,args.judge_head_lr)):
        raise ValueError('Judge learning rates must be finite and positive')
    set_seed(args.seed);torch.set_num_threads(args.num_threads)
    output=empty_output(args.output_dir)
    tok=AutoTokenizer.from_pretrained(args.classifier_model,local_files_only=True)
    stok=AutoTokenizer.from_pretrained(args.encoder,local_files_only=True)
    records=split_records(args)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(joint_cogltx=True,stable_training=True,method='cogltx',pipeline=PIPELINE,length_schema=LENGTH_SCHEMA,
        sample_unit=sample_unit(args.csv), turn_label_used=False,input_truncation='recent',selection_uses_gold_labels=False,
        csv_sha256=sha256(args.csv),split_ids={s:[r.id for r in rs] for s,rs in records.items()},amp=args.amp)
    del records
    splits=prepare_cache(args,tok,stok,config)
    reasoner=load_classifier(args.classifier_model,local_files_only=True).to(args.device)
    judge=AutoModelForTokenClassification.from_pretrained(args.encoder,num_labels=1,local_files_only=True).to(args.device)
    reasoner._stable_amp=judge._stable_amp=args.amp
    validate_length(tok,reasoner.config,args.classifier_max_length)
    validate_length(stok,judge.config,args.selector_max_length,name='selector_max_length')
    tok.save_pretrained(output/'tokenizer');stok.save_pretrained(output/'selector_tokenizer')
    warm_splits={s:select_rows(judge,splits[s],tok,stok,config,args.device) for s in ('train','val')}
    warm_best,weight=warmup(reasoner,warm_splits,tok,output,args.device,args.head_epochs,args.head_lr)
    # This is a recovery artifact, never presented as a jointly trained checkpoint.
    torch.save(reasoner.classifier.state_dict(),output/'warmup_head.pt')
    del warm_splits
    if args.gradient_checkpointing:
        reasoner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        judge.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ropt=optimizer_for(reasoner,args.encoder_lr,args.finetune_head_lr)
    jopt=optimizer_for(judge,args.judge_encoder_lr,args.judge_head_lr)
    steps=args.epochs*math.ceil(len(splits['train'])/args.effective_batch_size)
    rsched=scheduler_for(ropt,steps);jsched=scheduler_for(jopt,steps)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device=args.device))
    relevance={};best=-1.;history=[];stopped_for_collapse=False
    config.update(pos_weight=weight,warmup_val_auprc=warm_best)
    write_json(output/'training_config.json',config)
    for epoch in range(args.epochs):
        order=list(splits['train']);random.shuffle(order);rloss=0.;jloss=0.;positive=0;blocks_seen=0
        start_time=time.monotonic()
        for start in range(0,len(order),args.effective_batch_size):
            items=order[start:start+args.effective_batch_size]
            rows=select_rows(judge,items,tok,stok,config,args.device)
            loss,norm=train_window(reasoner,rows,tok,ropt,criterion,args.device,args.train_token_budget,args.batch_size)
            targets,npos=deletion_targets(reasoner,rows,tok,stok,relevance,config,args.device)
            jl=train_judge(judge,targets,stok,jopt,args.device)
            rsched.step();jsched.step();rloss+=loss*len(rows);jloss+=jl*len(rows);positive+=npos;blocks_seen+=len(targets)
            if start//args.effective_batch_size%20==0:
                event('cog_joint',epoch=epoch+1,done=start+len(rows),total=len(order),loss=loss,
                    judge_loss=jl,gradient_norm=norm,positive_blocks=positive,blocks_seen=blocks_seen,seconds=round(time.monotonic()-start_time,1))
        val=select_rows(judge,splits['val'],tok,stok,config,args.device)
        p=probabilities(reasoner,val,tok,args.device);stats=diagnostics(val,p)
        entry=dict(epoch=epoch+1,reasoner_loss=rloss/len(order),judge_loss=jloss/len(order),
            positive_blocks=positive,blocks_seen=blocks_seen,seconds=time.monotonic()-start_time,**stats)
        history.append(entry);write_json(output/'joint_history.json',history);event('cog_validation',**entry)
        if collapse_reason(stats):
            stopped_for_collapse=True
            if best<0:
                require_healthy(stats,output,'first_joint_epoch')
            write_json(output/'collapse_guard.json',dict(epoch=epoch+1,reason=collapse_reason(stats),action='restore best healthy paired checkpoint'))
            break
        if stats['auprc']>best:
            best=stats['auprc'];config.update(best_epoch=epoch+1,best_val_auprc=best)
            save_model(reasoner,output/'classifier');save_model(judge,output/'introspector')
        write_json(output/'relevance.json',relevance)
        write_json(output/'training_config.json',config)
    del reasoner,judge,ropt,jopt,rsched,jsched
    if args.device=='cuda':torch.cuda.empty_cache()
    reasoner=load_classifier(output/'classifier',local_files_only=True).to(args.device)
    judge=AutoModelForTokenClassification.from_pretrained(output/'introspector',local_files_only=True).to(args.device)
    reasoner._stable_amp=judge._stable_amp=args.amp
    val=select_rows(judge,splits['val'],tok,stok,config,args.device)
    p=probabilities(reasoner,val,tok,args.device)
    require_healthy(diagnostics(val,p),output,'saved_joint_model_validation')
    threshold=find_best_f1_threshold([r['label'] for r in val],p)
    save_evaluation(output,'val',val,[r['label'] for r in val],p,threshold,args.smoke_test)
    config.update(threshold=threshold,epochs_completed=len(history),stopped_for_collapse=stopped_for_collapse)
    write_json(output/'run_config.json',config)
    test=select_rows(judge,splits['test'],tok,stok,config,args.device)
    p=probabilities(reasoner,test,tok,args.device)
    save_evaluation(output,'test',test,[r['label'] for r in test],p,threshold,args.smoke_test)
    require_healthy(diagnostics(test,p,threshold),output,'test_output_check_no_retuning')
    write_json(output/'completion.json',dict(status='complete',test=diagnostics(test,p,threshold),best_epoch=config['best_epoch']))


if __name__=='__main__':
    from .cli import source_args,device_args,pipeline_length_args
    p=argparse.ArgumentParser(description=__doc__)
    source_args(p);device_args(p);pipeline_length_args(p);add_args(p)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--classifier-model',required=True)
    p.add_argument('--encoder',required=True)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--amp',choices=['auto','none','bf16'],default='auto')
    p.add_argument('--block-size',type=int,default=256)
    p.add_argument('--selector-max-length',type=int,default=512)
    p.add_argument('--top-k',type=int,default=None)
    p.add_argument('--judge-encoder-lr',type=float,default=2e-6)
    p.add_argument('--judge-head-lr',type=float,default=1e-4)
    p.add_argument('--levelup-threshold',type=float,default=.2)
    p.add_argument('--leveldown-threshold',type=float,default=-.05)
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    train(p.parse_args())
