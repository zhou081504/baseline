"""Network failures and offline recovery; no network or neural dependencies."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
import requests
from table_baselines.wiki_pairs import collect,sha256


def page(index):
    return dict(pageid=index,title=f'公开条目{index}',extract=f'甲{index}。乙。\n== 小节 ==\n丙。丁。',
                fullurl=f'https://zh.wikipedia.org/?curid={index}',revisions=[dict(revid=index+100)])


def response(pages,cursor=None):
    result=MagicMock();body={'query':{'pages':pages}}
    if cursor:body['continue']={'grncontinue':cursor,'continue':'grncontinue||'}
    result.json.return_value=body
    return result


class WikipediaCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.raw=self.root/'offline.jsonl'
        self.raw.write_text('\n'.join(json.dumps(page(i),ensure_ascii=False) for i in (1,2)),encoding='utf-8')

    def tearDown(self):self.temp.cleanup()

    def args(self,**kwargs):
        values=dict(output_dir=self.root/'out',pages=2,max_pairs=10,seed=42,offline_pages=self.raw,
                    retries=0,timeout=1.,retry_delay=0.,request_delay=0.)
        values.update(kwargs);return argparse.Namespace(**values)

    def test_offline_balanced_pairs_attribution_and_no_network(self):
        args=self.args()
        with patch('table_baselines.wiki_pairs.requests.Session',side_effect=AssertionError('offline must not connect')):
            meta=collect(args)
        rows=[json.loads(x) for x in (args.output_dir/'pairs.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows),4);self.assertEqual(sum(r['label'] for r in rows),2)
        self.assertTrue(all(r['source']['revision'] in (101,102) for r in rows))
        self.assertTrue(all(r['left']=='乙。' and r['right']=='丙。' for r in rows if r['label']==0))
        self.assertFalse(meta['task_call_data_used']);self.assertEqual(meta['pairs_sha256'],sha256(args.output_dir/'pairs.jsonl'))

    def test_network_failure_keeps_cache_and_resumes_cursor(self):
        args=self.args(offline_pages=None)
        session=MagicMock();session.get.side_effect=[response([page(1)],'cursor-1'),requests.ConnectionError('DNS failure')]
        with patch('table_baselines.wiki_pairs.requests.Session') as factory:
            factory.return_value.__enter__.return_value=session
            with self.assertRaisesRegex(ValueError,'Cached pages were kept'):collect(args)
        self.assertFalse((args.output_dir/'pairs.jsonl').exists())
        self.assertEqual(len((args.output_dir/'pages.jsonl').read_text(encoding='utf-8').splitlines()),1)
        resumed=MagicMock();resumed.get.return_value=response([page(2)])
        with patch('table_baselines.wiki_pairs.requests.Session') as factory:
            factory.return_value.__enter__.return_value=resumed;meta=collect(args)
        self.assertEqual(resumed.get.call_args.kwargs['params']['grncontinue'],'cursor-1')
        self.assertEqual(len(meta['pages']),2)

    def test_transient_failure_retries(self):
        args=self.args(offline_pages=None,retries=1)
        session=MagicMock();session.get.side_effect=[requests.Timeout('timeout'),response([page(1),page(2)])]
        with patch('table_baselines.wiki_pairs.requests.Session') as factory:
            factory.return_value.__enter__.return_value=session;collect(args)
        self.assertEqual(session.get.call_count,2)

    def test_completed_dataset_is_idempotent_and_tamper_is_rejected(self):
        args=self.args();meta=collect(args)
        with patch('table_baselines.wiki_pairs.requests.Session',side_effect=AssertionError('must reuse completed data')):
            self.assertEqual(collect(self.args(offline_pages=None)),meta)
        target=args.output_dir/'pairs.jsonl';target.write_text('tampered',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'modified data'):collect(args)
        self.assertEqual(target.read_text(),'tampered')

    def test_insufficient_offline_cache_cannot_publish(self):
        args=self.args(pages=3)
        with self.assertRaisesRegex(ValueError,'only 2'):collect(args)
        self.assertFalse((args.output_dir/'pairs.jsonl').exists())
        self.assertFalse((args.output_dir/'provenance.json').exists())

    def test_frozen_cache_reproduces_pair_hash(self):
        first=collect(self.args())
        second=collect(self.args(output_dir=self.root/'second',offline_pages=self.root/'out/pages.jsonl'))
        self.assertEqual(first['pairs_sha256'],second['pairs_sha256'])

    def test_snapshot_origin_is_preserved(self):
        rows=[page(1),page(2)]
        for row in rows:row['origin']=dict(dataset='wikimedia/wikipedia',snapshot='20231101.zh',repository_revision='frozen-test-version')
        self.raw.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
        meta=collect(self.args())
        self.assertEqual(meta['source'],'Chinese Wikipedia public dataset snapshot')
        self.assertTrue(all(p['corpus']['snapshot']=='20231101.zh' for p in meta['pages']))


if __name__=='__main__':unittest.main()
