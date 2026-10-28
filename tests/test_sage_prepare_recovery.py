import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from Selection_Retrieval.SAGE.feedback_json import parse_feedback
from table_baselines.prepare_progress import validate_resume, preparation_lock


class FeedbackTests(unittest.TestCase):
    def test_json_and_fences(self):
        for text in ['{"score":8,"adjustment":-1}', '```json\n{"score":8,"adjustment":-1}\n```']:
            self.assertEqual(parse_feedback(text), {'score':8,'adjustment':-1})

    def test_invalid_formats_and_ranges(self):
        for text in ['{"score":8, adjustment:-1}', '{"score":8,"adjustment":-1,}',
                     '{"score":11,"adjustment":1}', '{"score":true,"adjustment":1}',
                     '{"score":8,"adjustment":0}', '{"score":8,"adjustment":1.5}',
                     '{"score":"8","adjustment":1}', '{"score":8}', 'no JSON',
                     '{"score":8,"adjustment":1}{"score":2,"adjustment":-1}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_feedback(text)

    def test_retry_and_exhaustion(self):
        from Selection_Retrieval.SAGE.feedback import LocalFeedback
        obj = LocalFeedback.__new__(LocalFeedback)
        with patch.object(obj, 'generate', side_effect=['answer','{"score":8, bad}', '{"score":8,"adjustment":1}']) as gen:
            self.assertEqual(obj('query','context'), {'score':8,'adjustment':1})
            self.assertEqual(gen.call_count,3)
        with patch.object(obj,'generate',side_effect=['answer','bad','bad','bad']) as gen:
            with self.assertRaisesRegex(ValueError,'3 attempts'):
                obj('query','context')
            self.assertEqual(gen.call_count,4)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name)/'prepared'; self.output.mkdir()
        self.config = {'csv_sha256':'abc','classifier_max_length':8,'seed':42}
        (self.output/'prepare_config.json').write_text(json.dumps(self.config))
        self.partial=self.output/'selected.jsonl.partial'
        self.row={'id':'a','label':1,'split':'train','token_ids':[1,2,3]}
        self.partial.write_text(json.dumps(self.row)+'\n')
        self.expected=[('train',SimpleNamespace(id='a',label=1)),('val',SimpleNamespace(id='b',label=0))]

    def test_valid_prefix_unchanged(self):
        before=self.partial.read_bytes()
        self.assertEqual(validate_resume(self.output,dict(self.config,resume=True),self.expected),1)
        self.assertEqual(before,self.partial.read_bytes())

    def test_mismatched_source_or_parameters(self):
        for key in ['csv_sha256','seed','classifier_max_length']:
            with self.assertRaisesRegex(ValueError,'mismatch'):
                validate_resume(self.output,dict(self.config,**{key:'changed'}),self.expected)

    def test_wrong_or_duplicate_rows(self):
        for key,value in [('id','wrong'),('label',0),('split','test'),('token_ids',[]),('token_ids',[False])]:
            self.partial.write_text(json.dumps(dict(self.row,**{key:value}))+'\n')
            with self.assertRaises(ValueError): validate_resume(self.output,self.config,self.expected)
        self.partial.write_text((json.dumps(self.row)+'\n')*2)
        with self.assertRaises(ValueError): validate_resume(self.output,self.config,self.expected)

    def test_torn_tail_and_completed_output_preserved(self):
        self.partial.write_bytes(b'{"id":')
        with self.assertRaises(ValueError): validate_resume(self.output,self.config,self.expected)
        self.assertEqual(self.partial.read_bytes(), b'{"id":')
        (self.output/'selected.jsonl').write_text('existing')
        with self.assertRaisesRegex(ValueError,'complete'): validate_resume(self.output,self.config,self.expected)

    def test_lock_rejects_concurrent_preparation(self):
        with preparation_lock(self.output):
            with self.assertRaises(ValueError):
                with preparation_lock(self.output): pass
        with preparation_lock(self.output): pass

    def test_interrupted_pipeline_resumes_exact_prefix(self):
        from dataclasses import dataclass
        from table_baselines.cli import prepare
        @dataclass
        class Selected:
            token_ids: list
            details: dict
        args=SimpleNamespace(method='recent',query='',seed=42,classifier_model='tiny',classifier_max_length=8,
                             output_dir=self.output/'pipeline',csv='fixture',max_length=8,resume=False,smoke_test=True)
        rows=[SimpleNamespace(id=str(i),label=i%2,raw_text='text'+str(i)) for i in range(3)]
        tokenizer=SimpleNamespace(save_pretrained=lambda p:None)
        selector=SimpleNamespace(select=lambda text:Selected([1,2],{}))
        with patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer), \
             patch('transformers.AutoConfig.from_pretrained'), patch('table_baselines.runtime.validate_length'), \
             patch('table_baselines.runtime.sha256',return_value='hash'), \
             patch('table_baselines.runtime.tokenizer_fingerprint',return_value='tok'), \
             patch('table_baselines.runtime.split_records',return_value={'train':rows}), \
             patch('table_baselines.budget.TokenBudget'), \
             patch('table_baselines.cli.recent_input',side_effect=lambda text,*_: (text,{})), \
             patch('table_baselines.selectors.make_selector',return_value=selector):
            with patch.object(selector,'select',side_effect=[Selected([1,2],{}),ValueError('bad feedback')]):
                with self.assertRaises(ValueError): prepare(args)
            prefix=(args.output_dir/'selected.jsonl.partial').read_bytes()
            args.resume=True
            with patch.object(selector,'select',return_value=Selected([1,2],{})) as select:
                prepare(args)
                self.assertEqual(select.call_count,2)
            complete=(args.output_dir/'selected.jsonl').read_bytes()
            self.assertTrue(complete.startswith(prefix))
            self.assertEqual([json.loads(x)['id'] for x in complete.splitlines()],['0','1','2'])
            self.assertFalse((args.output_dir/'prepare_failure.json').exists())


if __name__=='__main__': unittest.main(verbosity=2)
