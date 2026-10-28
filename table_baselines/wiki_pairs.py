"""Collect public Chinese Wikipedia paragraphs, with resumable/offline caching.

Same-paragraph adjacent sentences are positive; paragraph boundaries negative.
Only public Wikipedia data is used. No task call data is read or uploaded.
This module requires requests, but does not import torch or transformers.
"""
import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import requests
from .text import sentences

API = 'https://zh.wikipedia.org/w/api.php'
SCHEMA = 'wiki_pairs_v2'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def read_pages(path):
    pages=[];seen=set()
    if not path.exists():return pages
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        page=json.loads(line)
        if not isinstance(page.get('pageid'),int) or not isinstance(page.get('title'),str) or not isinstance(page.get('extract'),str):
            raise ValueError('Page cache needs pageid, title and extract: '+str(path))
        if page['pageid'] not in seen:pages.append(page);seen.add(page['pageid'])
    return pages


def fetch(session, params, args):
    for attempt in range(args.retries+1):
        try:
            response=session.get(API,params=params,timeout=args.timeout)
            response.raise_for_status()
            body=response.json()
            if 'error' in body:
                error=body['error']
                if error.get('code') in ('maxlag','ratelimited'):
                    raise requests.ConnectionError('Wikipedia is temporarily busy: '+error.get('code',''))
                raise ValueError('Wikipedia API error: '+str(error))
            return body
        except requests.RequestException as exc:
            if attempt==args.retries:
                raise ValueError('Cannot reach zh.wikipedia.org (DNS/network/HTTP failure). Cached pages were kept. Retry to resume, or collect on an Internet-connected machine and copy the complete data directory; --offline-pages pages.jsonl can rebuild pairs without network.') from exc
            time.sleep(min(8.,args.retry_delay*2**attempt))


def collect(args):
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    pair_file=output/'pairs.jsonl';manifest_file=output/'provenance.json'
    cache=output/'pages.jsonl';state_file=output/'collection_state.json'
    requested=dict(pages=args.pages,max_pairs=args.max_pairs,seed=args.seed)
    if manifest_file.exists() and pair_file.exists():
        manifest=json.loads(manifest_file.read_text(encoding='utf-8'))
        if (manifest.get('schema')==SCHEMA and manifest.get('requested')==requested
                and manifest.get('pairs_sha256')==sha256(pair_file)):
            print(f"Already complete and verified: {manifest['pairs']} pairs from {len(manifest['pages'])} pages",flush=True)
            return manifest
        raise ValueError('Output already contains different or modified data; choose a new output directory')
    allowed={'pages.jsonl','collection_state.json','collection_error.json','pairs.jsonl.partial','provenance.json.tmp','collection_state.json.tmp','collection_error.json.tmp'}
    # A final pair file without its final manifest is recoverable only when it
    # belongs to this collector's saved state, never an arbitrary user dataset.
    if state_file.exists():allowed.add('pairs.jsonl')
    if any(p.name not in allowed for p in output.iterdir()):
        raise ValueError('Output contains unrelated files; choose a new output directory')
    state=json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {}
    if state and (state.get('schema')!=SCHEMA or state.get('requested')!=requested):
        raise ValueError('Cached collection settings differ; choose a new output directory')
    pages=read_pages(cache);seen={p['pageid'] for p in pages}
    continuation=state.get('continuation',{})
    def checkpoint():
        write_json(state_file,dict(schema=SCHEMA,requested=requested,pages_cached=len(pages),continuation=continuation))
    checkpoint()
    def append(page,handle):
        if len(pages)>=args.pages or page['pageid'] in seen:return
        if not page.get('extract','').strip():return
        # Store the original API extract and attribution, not private inputs.
        clean={k:page.get(k) for k in ('pageid','title','extract','fullurl','revisions')}
        if page.get('origin'):clean['origin']=page['origin']
        pages.append(clean);seen.add(page['pageid'])
        handle.write(json.dumps(clean,ensure_ascii=False)+'\n');handle.flush()
    with cache.open('a',encoding='utf-8') as handle:
        if args.offline_pages:
            for page in read_pages(args.offline_pages):append(page,handle)
            checkpoint()
            if len(pages)<args.pages:
                raise ValueError(f'Offline cache has only {len(pages)} usable pages; reduce --pages or supply a larger cache')
        else:
            stalled=0
            with requests.Session() as session:
                session.headers['User-Agent']='BaselineResearch/0.2 (SAGE public paragraph segmentation experiment)'
                while len(pages)<args.pages:
                    params=dict(action='query',format='json',formatversion=2,generator='random',
                        grnnamespace=0,grnlimit=1,prop='extracts|info|revisions',
                        explaintext=1,exlimit=1,inprop='url',rvprop='ids',maxlag=5)
                    params.update(continuation)
                    body=fetch(session,params,args);before=len(pages)
                    for page in body.get('query',{}).get('pages',[]):append(page,handle)
                    continuation=body.get('continue',{})
                    checkpoint()
                    stalled=stalled+1 if len(pages)==before else 0
                    if stalled>=10:raise ValueError('Wikipedia returned no new usable pages for 10 requests; cache retained, retry later')
                    if len(pages)//100>before//100 or len(pages)==args.pages:
                        print(f'wikipedia pages={len(pages)}/{args.pages}',flush=True)
                    time.sleep(args.request_delay)
    positives=[];negatives=[];sources=[]
    for page in pages[:args.pages]:
        paragraphs=[sentences(p) for p in re.split(r'\n+',page['extract']) if p.strip() and not p.lstrip().startswith('==')]
        paragraphs=[p for p in paragraphs if p]
        source=dict(pageid=page['pageid'],title=page['title'],url=page.get('fullurl') or f"https://zh.wikipedia.org/?curid={page['pageid']}",
            revision=((page.get('revisions') or [{}])[0]).get('revid'))
        if page.get('origin'):source['corpus']=page['origin']
        sources.append(source)
        for paragraph in paragraphs:
            positives.extend(dict(left=a,right=b,label=1,source=source) for a,b in zip(paragraph,paragraph[1:]))
        negatives.extend(dict(left=a[-1],right=b[0],label=0,source=source) for a,b in zip(paragraphs,paragraphs[1:]))
    rng=random.Random(args.seed);rng.shuffle(positives);rng.shuffle(negatives)
    count=min(len(positives),len(negatives),args.max_pairs//2)
    if not count:raise ValueError('No balanced paragraph pairs found; collect more pages into a new directory')
    pairs=positives[:count]+negatives[:count];rng.shuffle(pairs)
    partial=output/'pairs.jsonl.partial'
    with partial.open('w',encoding='utf-8') as handle:
        for pair in pairs:handle.write(json.dumps(pair,ensure_ascii=False)+'\n')
    partial.replace(pair_file)
    dump_sources=[p['origin'] for p in pages[:args.pages] if p.get('origin')]
    manifest=dict(schema=SCHEMA,source='Chinese Wikipedia public dataset snapshot' if dump_sources else 'Chinese Wikipedia API',api=None if dump_sources else API,retrieved_unix=time.time(),seed=args.seed,
        requested=requested,pairs=len(pairs),class_counts={'same_paragraph':count,'paragraph_boundary':count},
        available_pairs={'same_paragraph':len(positives),'paragraph_boundary':len(negatives)},pages=sources,
        pairs_sha256=sha256(pair_file),pages_sha256=sha256(cache),
        license='CC BY-SA; see each page history and https://zh.wikipedia.org/wiki/Wikipedia:著作权',
        task_call_data_used=False,sampling='See page corpus metadata for seeded dump-row sampling; frozen pairs are reproducible' if dump_sources else 'Wikipedia random-page API with continuation; cache and frozen pairs enable reproducibility')
    write_json(manifest_file,manifest)
    print(f"Complete: {len(sources)} pages, {len(pairs)} balanced pairs; pairs SHA256={manifest['pairs_sha256']}",flush=True)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--pages',type=int,default=100)
    parser.add_argument('--max-pairs',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--offline-pages',type=Path,help='original API page JSONL; build pairs without network')
    parser.add_argument('--retries',type=int,default=3)
    parser.add_argument('--timeout',type=float,default=20.)
    parser.add_argument('--retry-delay',type=float,default=1.)
    parser.add_argument('--request-delay',type=float,default=.2)
    args=parser.parse_args(argv)
    if args.pages<1 or args.max_pairs<2 or args.retries<0 or args.timeout<=0 or args.retry_delay<0 or args.request_delay<0:
        parser.error('pages/timeout must be positive; max-pairs >= 2; retries and delays >= 0')
    try:return collect(args)
    except (ValueError,OSError,requests.RequestException) as exc:
        if args.output_dir.is_dir():write_json(args.output_dir/'collection_error.json',dict(error=str(exc),time=time.time()))
        parser.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
