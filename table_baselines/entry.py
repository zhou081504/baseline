"""Per-model command entry points with local output defaults and method checks."""
import json
import os
from pathlib import Path
import sys


def option(argv,name,default=None):
    result=default
    for i,arg in enumerate(argv):
        if arg==name and i+1<len(argv):result=argv[i+1]
        elif arg.startswith(name+'='):result=arg.split('=',1)[1]
    return result


def stable_model_default(model_dir):
    """Prefer completed stable weights, never a partially written new run."""
    artifacts=model_dir/'artifacts'
    from .runtime import sha256
    source=model_dir.parents[1]/'merged_最终版.xlsx'
    digest=sha256(source) if source.is_file() else None
    candidates=[artifacts/'model_final']+sorted(artifacts.glob('model_*'),reverse=True)
    for path in candidates:
        config=path/'run_config.json'; done=path/'completion.json'
        if config.is_file() and done.is_file():
            saved=json.loads(config.read_text())
            source_digest=saved.get('csv_sha256') or saved.get('preparation',{}).get('csv_sha256')
            if source_digest != digest or digest is None:
                continue
            if saved.get('stable_training') and json.loads(done.read_text()).get('status')=='complete':
                return path
    return artifacts/'model_final'


def prepared_default(model_dir):
    """Prefer complete preparations that match the current project workbook."""
    from .runtime import sha256
    source=model_dir.parents[1]/'merged_最终版.xlsx'
    if source.is_file():
        digest=sha256(source)
        candidates=sorted((model_dir/'artifacts').glob('prepared*'),reverse=True)
        for path in candidates:
            meta=path/'prepare_config.json'
            if meta.is_file() and (path/'selected.jsonl').is_file() and json.loads(meta.read_text()).get('csv_sha256')==digest:
                return path
    return model_dir/'artifacts/prepared_final'


def run(method,command,model_dir,argv=None):
    model_dir=Path(model_dir).resolve()
    root=model_dir.parents[1]
    argv=list(sys.argv[1:] if argv is None else argv)
    os.environ.setdefault('TIKTOKEN_CACHE_DIR',str(root/'.cache/tiktoken'))
    if option(argv,'--method') is not None:
        raise SystemExit('This folder fixes the method; use the target model folder instead of --method.')
    defaults={}
    if command=='prepare':
        defaults={'--output-dir':model_dir/'artifacts/prepared_final'}
        if method=='sage':defaults['--segmenter-checkpoint']=model_dir/'artifacts/segmenter'
    elif command=='train':
        prep=option(argv,'--prepared') or prepared_default(model_dir)
        defaults={'--prepared':prep,'--output-dir':model_dir/'artifacts/model_final'}
    elif command=='infer':
        checkpoint=Path(option(argv,'--model-dir') or stable_model_default(model_dir))
        prep=model_dir/'artifacts/prepared_final'
        config=checkpoint/'run_config.json'
        if config.is_file():
            saved=json.loads(config.read_text()).get('prepared')
            if saved:
                prep=Path(saved)
                if not prep.is_absolute():prep=root/prep
        defaults={'--prepared':prep,'--model-dir':checkpoint}
    elif command=='train-cogltx':
        defaults={'--output-dir':model_dir/'artifacts/model_final'}
    elif command=='infer-cogltx':
        defaults={'--model-dir':option(argv,'--model-dir') or stable_model_default(model_dir)}
    elif command=='train-segmenter':
        defaults={'--pairs':model_dir/'data/wikipedia/pairs.jsonl','--output-dir':model_dir/'artifacts/segmenter'}
    elif command=='collect-wikipedia':
        defaults={'--output-dir':model_dir/'data/wikipedia'}
    prefix=[]
    for key,value in defaults.items():
        if option(argv,key) is None:prefix.extend([key,str(value)])
    args=prefix+argv
    if not any(x in ('--help','-h') for x in argv):
        if command in ('train','infer'):
            meta=Path(option(args,'--prepared'))/'prepare_config.json'
            if meta.exists() and json.loads(meta.read_text(encoding='utf-8')).get('method')!=method:
                raise SystemExit(f'Prepared inputs belong to another method; this folder is {method}.')
        if command in ('infer','infer-cogltx'):
            meta=Path(option(args,'--model-dir'))/'run_config.json'
            if meta.exists() and json.loads(meta.read_text(encoding='utf-8')).get('method')!=method:
                raise SystemExit(f'Checkpoint belongs to another method; this folder is {method}.')
    if command=='collect-wikipedia':
        from .wiki_pairs import main
        main(args)
    else:
        from .cli import main
        main([command]+(['--method',method] if command=='prepare' else [])+args)
