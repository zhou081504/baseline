"""Download official inference files, pin revisions and keep resumable HF state."""
import argparse
import json
import os
from pathlib import Path

from .models import REPOSITORIES,ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=list(REPOSITORIES),default=list(REPOSITORIES))
    parser.add_argument('--output-root',type=Path,default=ROOT/'models')
    args=parser.parse_args()
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT','60')
    from huggingface_hub import HfApi,snapshot_download
    api=HfApi()
    for key in args.models:
        repo=REPOSITORIES[key]
        target=args.output_root/repo.replace('/','--')
        target.mkdir(parents=True,exist_ok=True)
        plan=target/'download_plan.json'
        if plan.exists():
            metadata=json.loads(plan.read_text())
        else:
            info=api.model_info(repo)
            names=[x.rfilename for x in info.siblings if '/' not in x.rfilename]
            weights=[n for n in names if n.endswith('.safetensors')]
            if not weights:
                weights=[n for n in names if n.startswith('pytorch_model') and n.endswith('.bin')]
            if not weights:
                raise ValueError(f'No supported weights found in {repo}')
            files=weights+[n for n in names if n.endswith('.json') or n in
                          ['vocab.txt','sentencepiece.bpe.model','tokenizer.model','merges.txt','LICENSE','README.md']]
            metadata={'repo':repo,'revision':info.sha,'files':files}
            plan.write_text(json.dumps(metadata,indent=2))
        print('DOWNLOAD_START',repo,metadata['revision'],flush=True)
        snapshot_download(repo_id=repo,revision=metadata['revision'],local_dir=target,
                          allow_patterns=metadata['files'],max_workers=2)
        (target/'download_manifest.json').write_text(json.dumps(metadata,indent=2))
        print('DOWNLOAD_COMPLETE',repo,flush=True)


if __name__=='__main__':
    main()
