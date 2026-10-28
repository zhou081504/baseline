"""Model locations and provenance for official pretrained weights."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPOSITORIES={
    'dense':'BAAI/bge-m3',
    'recomp':'fangyuan/nq_extractive_compressor',
    'reranker':'BAAI/bge-reranker-v2-m3',
    'lm':'Qwen/Qwen2.5-3B-Instruct',
}


def resolve_models(args):
    fields=['classifier_model'] if args.command in ('prepare','train') else []
    if args.command in ('train-cogltx','train-segmenter'):
        fields=['encoder'] + (['classifier_model'] if args.command=='train-cogltx' else [])
    if args.command=='prepare':
        fields += {'recent':[], 'cogltx':[], 'dense':['dense_model'],
                   'recomp':['recomp_model'], 'carrot':['dense_model','reranker_model'],
                   'longllmlingua':['lm_model'],
                   'sage':['dense_model','reranker_model','lm_model']}[args.method]
    sources={}
    for field in fields:
        value=getattr(args,field)
        candidate=ROOT/'models'/value.replace('/','--')
        if value in REPOSITORIES.values() and candidate.exists():
            if not (candidate/'download_manifest.json').is_file():
                raise ValueError(f'Model download is incomplete: {candidate}. Resume with python -m table_baselines.download_models; no random substitute will be used.')
            value=str(candidate)
            setattr(args,field,value)
        manifest=Path(value)/'download_manifest.json'
        sources[field]=json.loads(manifest.read_text()) if manifest.is_file() else {'model':value}
    args.model_sources=sources
