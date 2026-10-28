import argparse
from dataclasses import asdict
import json
from pathlib import Path

METHODS = ['recent','dense','cogltx','sage','carrot','longllmlingua','recomp']
ROOT = Path(__file__).resolve().parents[1]
from common.data import sample_unit
from .long_context import DEFAULT_CLASSIFIER
from .input_budget import (DEFAULT_INPUT_MAX_LENGTH, DEFAULT_CLASSIFIER_MAX_LENGTH, LENGTH_SCHEMA, recent_input)
# 方法内部的短文本分段/选块编码器，与最终分类器分开。
DEFAULT_INTERNAL_ENCODER = '/data/models/chinese-bert-wwm-ext'


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return result


def device_args(parser):
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--num-threads', type=positive, default=2)


def source_args(parser):
    parser.add_argument('--csv','--data', type=Path, default=ROOT/'merged_最终版.xlsx')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-ratio', type=float, default=.7)
    parser.add_argument('--val-ratio', type=float, default=.15)
    parser.add_argument('--test-ratio', type=float, default=.15)
    parser.add_argument('--smoke-test', action='store_true', help='two examples/class/split; not paper metrics')


def length_arg(parser, default=128):
    parser.add_argument('--max-length','--max_length', dest='max_length', type=positive, default=default,
                        help='final classifier input budget, INCLUDING special tokens')


def pipeline_length_args(parser):
    parser.add_argument('--max-length','--max_length',dest='max_length',type=nonnegative,
        default=DEFAULT_INPUT_MAX_LENGTH,
        help='complete-sample CONTENT-token cap BEFORE selection; keep latest text; RECOMP accepts 0 for unlimited original text')
    parser.add_argument('--classifier-max-length','--classifier_max_length',dest='classifier_max_length',
        type=positive,default=DEFAULT_CLASSIFIER_MAX_LENGTH,
        help='selected/concatenated classifier-token cap INCLUDING special tokens; default 4096')


def prepare(args):
    from .prepare_progress import preparation_lock
    with preparation_lock(args.output_dir):
        return _prepare(args)


def _prepare(args):
    from transformers import AutoConfig, AutoTokenizer
    from .budget import TokenBudget
    from .runtime import empty_output, set_seed, sha256, split_records, validate_length, write_json, tokenizer_fingerprint
    from .selectors import make_selector
    if args.method == 'sage' and not args.segmenter_checkpoint:
        raise ValueError('SAGE requires a trained --segmenter-checkpoint; use train-segmenter with paragraph-derived pairs')
    if args.method == 'cogltx' and not args.selector_checkpoint:
        raise ValueError('CogLTX requires --selector-checkpoint from train-cogltx')
    if args.method not in ('recent','cogltx') and not args.query.strip():
        raise ValueError('Query must not be empty')
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.classifier_model)
    validate_length(tokenizer,AutoConfig.from_pretrained(args.classifier_model),args.classifier_max_length)
    splits = split_records(args)
    output = Path(args.output_dir)
    config = {key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    config.update(csv_sha256=sha256(args.csv), tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
                  split_ids={name:[r.id for r in rows] for name,rows in splits.items()},
                  selection_uses_gold_labels=False, length_schema=LENGTH_SCHEMA,
                  sample_unit=sample_unit(args.csv), turn_label_used=False, input_truncation='none' if args.max_length == 0 else 'recent')
    expected = [(split, record) for split, records in splits.items() for record in records]
    resume = getattr(args, 'resume', False)
    if resume:
        from .prepare_progress import validate_resume
        count = validate_resume(output, config, expected)
        print(f'Resume verified: {count}/{len(expected)} completed rows retained', flush=True)
        history_file = output/'resume_history.json'
        history = json.loads(history_file.read_text()) if history_file.exists() else []
        history.append({'completed_rows': count, 'prefix_sha256': sha256(output/'selected.jsonl.partial'),
                        'feedback_parser': 'strict_json_bounded_retry_v1', 'csv_sha256': config['csv_sha256']})
        write_json(history_file, history)
    else:
        output = empty_output(output)
        write_json(output/'prepare_config.json',config)
        tokenizer.save_pretrained(output/'tokenizer')
        count = 0
    budget = TokenBudget(tokenizer,args.classifier_max_length)
    selector = make_selector(args,budget) if count < len(expected) else None
    partial = output/'selected.jsonl.partial'
    with partial.open('a' if resume else 'x',encoding='utf-8') as handle:
        for split,record in expected[count:]:
            # Never pass labels, IDs or split names into text selection.
            text, input_info = recent_input(record.raw_text, tokenizer, args.max_length)
            try:
                selected = selector.select(text)
            except Exception as exc:
                write_json(output/'prepare_failure.json', {'completed_rows': count, 'next_id': record.id,
                           'error_type': type(exc).__name__, 'resume_supported': True})
                raise
            selected.details.update(input_info)
            if not selected.token_ids or len(selected.token_ids)>args.classifier_max_length:
                raise ValueError('Selector violated classifier_max_length')
            row = dict(asdict(selected), id=record.id, label=record.label, split=split)
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
            handle.flush()
            count += 1
            if count % 100 == 0:
                print(f'prepared={count}',flush=True)
    partial.rename(output/'selected.jsonl')
    (output/'prepare_failure.json').unlink(missing_ok=True)
    print(f'Prepared {count} records: {output}; smoke_test={args.smoke_test}',flush=True)


def main(argv=None):
    from .selectors import DEFAULT_QUERY, DEFAULT_LM
    from .stable_training import add_args as stable_args
    parser = argparse.ArgumentParser(description='Requested table baselines; TurnRank and EviCompose intentionally excluded')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare',help='select/compress complete calls before downstream training')
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='validate and continue selected.jsonl.partial without reprocessing completed rows')
    source_args(p); device_args(p); pipeline_length_args(p)
    p.add_argument('--classifier-model',default=DEFAULT_CLASSIFIER)
    p.add_argument('--query',default=DEFAULT_QUERY)
    p.add_argument('--top-k',type=positive,default=None,
                   help='optional sentence cap for RECOMP (default: fill classifier token budget); other selectors default to 10')
    p.add_argument('--selector-batch-size',type=positive,default=1)
    p.add_argument('--retrieval-max-length',type=positive,default=512)
    p.add_argument('--dense-model',default='BAAI/bge-m3')
    p.add_argument('--recomp-model',default='BAAI/bge-m3')
    p.add_argument('--recomp-encoder-style',choices=['bge','official'],default='bge',
                   help='bge: Chinese-compatible CLS/cosine; official: original RECOMP mean/dot (use with fangyuan model)')
    p.add_argument('--reranker-model',default='BAAI/bge-reranker-v2-m3')
    p.add_argument('--lm-model',default=DEFAULT_LM)
    p.add_argument('--feedback-device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--selector-checkpoint',type=Path)
    p.add_argument('--segmenter-checkpoint',type=Path)
    p.add_argument('--segmentation-threshold',type=float,default=.55)
    p.add_argument('--coarse-length',type=positive,default=400)
    p.add_argument('--min-k',type=positive,default=7)
    p.add_argument('--gradient-divisor',type=float,default=.3,
                   help='literal SAGE equation: current > previous/g; paper reports .3 (see README ambiguity)')
    p.add_argument('--feedback-threshold',type=float,default=9)
    p.add_argument('--mcts-iterations',type=positive,default=30)
    p.add_argument('--exploration',type=float,default=1.41)
    p.add_argument('--cost-penalty',type=float,default=1.0)
    p = sub.add_parser('train',help='fine-tune the shared long-context classifier on selected/compressed inputs')
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--classifier-model',default=DEFAULT_CLASSIFIER)
    p.add_argument('--batch-size',type=positive,default=16,help='maximum physical microbatch; effective batch defaults to 64')
    p.add_argument('--epochs',type=positive,default=2)
    p.add_argument('--lr',type=float,default=None,help='obsolete; use --encoder-lr and --finetune-head-lr')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--gradient-accumulation-steps',type=positive,default=None,help='obsolete; use --effective-batch-size')
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--amp',choices=['auto','none','bf16'],default='auto')
    stable_args(p)
    device_args(p)
    p = sub.add_parser('infer',help='offline evaluation on saved selected inputs and split IDs')
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--split',choices=['train','val','test','all'],default='test')
    p.add_argument('--amp',choices=['auto','none','fp16','bf16'],default='auto')
    p.add_argument('--batch-size',type=positive,default=32)
    device_args(p)
    p = sub.add_parser('train-segmenter',help='train SAGE on paragraph-derived sentence pairs, not call risk labels')
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--encoder',default=DEFAULT_INTERNAL_ENCODER)
    p.add_argument('--hidden-size',type=positive,default=256)
    p.add_argument('--batch-size',type=positive,default=1)
    p.add_argument('--epochs',type=positive,default=3)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--seed',type=int,default=42)
    length_arg(p,128); device_args(p)
    p = sub.add_parser('train-cogltx',help='complete input -> recent cap -> blocks -> judge scores -> concatenate -> classifier')
    p.add_argument('--classifier-model',default=DEFAULT_CLASSIFIER)
    p.add_argument('--selector-max-length',type=positive,default=512)
    p.add_argument('--amp',choices=['auto','none','bf16'],default='auto')
    source_args(p); device_args(p); pipeline_length_args(p)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--encoder',default=DEFAULT_INTERNAL_ENCODER)
    p.add_argument('--block-size',type=positive,default=256)
    p.add_argument('--epochs',type=positive,default=2)
    p.add_argument('--lr',type=float,default=None,help='obsolete; use --encoder-lr and --finetune-head-lr')
    p.add_argument('--cache-dir',type=Path,help='verified tokenization cache; default is beside output directory')
    p.add_argument('--batch-size',type=positive,default=16)
    p.add_argument('--judge-encoder-lr',type=float,default=2e-6)
    p.add_argument('--judge-head-lr',type=float,default=1e-4)
    stable_args(p)
    p.add_argument('--top-k',type=positive,default=None,help='optional block-count cap; by default fill classifier budget')
    p.add_argument('--levelup-threshold',type=float,default=.2)
    p.add_argument('--leveldown-threshold',type=float,default=-.05)
    p.add_argument('--gradient-checkpointing',action=argparse.BooleanOptionalAction,default=True)
    p = sub.add_parser('infer-cogltx',help='evaluate the saved joint CogLTX checkpoint')
    p.add_argument('--amp',choices=['auto','none','fp16','bf16'],default='auto')
    p.add_argument('--csv','--data',type=Path,default=ROOT/'merged_最终版.xlsx')
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--split',choices=['train','val','test','all'],default='test')
    device_args(p)
    args = parser.parse_args(argv)
    if getattr(args, 'max_length', None) == 0 and not (args.command == 'prepare' and args.method == 'recomp'):
        parser.error('--max_length 0 is supported only for RECOMP prepare')
    if args.command == 'prepare' and args.method != 'recomp' and args.top_k is None:
        args.top_k = 10
    import torch
    torch.set_num_threads(args.num_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    if args.command in ('train','train-cogltx'):
        if args.lr is not None:
            parser.error('--lr is obsolete: use --encoder-lr (default 2e-6) and --finetune-head-lr (default 2e-5)')
        if getattr(args,'gradient_accumulation_steps',None) is not None:
            parser.error('--gradient-accumulation-steps is obsolete: use --effective-batch-size (default 64); --batch-size only caps memory use')
        if args.command=='train-cogltx' and args.cache_dir is None:
            args.cache_dir=args.output_dir.with_name(args.output_dir.name+'_blocks')
    if getattr(args,'lr',None) is not None and args.lr<=0:
        parser.error('lr must be positive')
    if args.command=='prepare' and args.method=='sage':
        if not 0<=args.segmentation_threshold<=1 or args.gradient_divisor<=0 or not 1<=args.feedback_threshold<=10:
            parser.error('Invalid SAGE thresholds')
        if args.gradient_divisor<=1:
            print('SAGE paper ambiguity: g<=1 with positive descending scores retains only min_k; literal published equation is used. See docs/PROTOCOL.md.',flush=True)
    try:
        from .models import resolve_models
        resolve_models(args)
        if args.command=='prepare':
            prepare(args)
        elif args.command=='train':
            from .runtime import train_classifier
            train_classifier(args)
        elif args.command=='infer':
            from .runtime import evaluate_classifier
            evaluate_classifier(args)
        elif args.command=='train-segmenter':
            from .segmenter import train_segmenter
            train_segmenter(args)
        elif args.command=='train-cogltx':
            from .cogltx import train
            train(args)
        elif args.command=='infer-cogltx':
            from .cogltx import infer
            infer(args)
    except torch.OutOfMemoryError:
        parser.exit(2,'Out of memory: use --device cpu or a smaller batch/classifier_max_length; other processes are never killed.\n')
    except (ValueError,FileNotFoundError,ImportError) as exc:
        parser.error(str(exc))
