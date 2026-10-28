"""Default routes and failure guards, using tiny local models on CPU."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
from table_baselines.stable_training import train, warmup, diagnostics, collapse_reason, head_features, head_logits, validate_prepared
from table_baselines.runtime import sha256
from table_baselines.input_budget import LENGTH_SCHEMA


class DefaultStableTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        vocab=self.root/'vocab.txt';vocab.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','甲','乙','丙']))
        self.tok=BertTokenizerFast(vocab_file=str(vocab))
        self.base=self.root/'base';self.tok.save_pretrained(self.base)
        cfg=BertConfig(vocab_size=len(self.tok),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
            intermediate_size=32,max_position_embeddings=64,num_labels=2,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.)
        torch.manual_seed(17);self.model=BertForSequenceClassification(cfg).eval();self.model.save_pretrained(self.base)
        self.splits={s:[dict(id=f'{s}{i}',label=i//2,split=s,token_ids=[2,5+i%3,3]) for i in range(4)] for s in ('train','val','test')}
        self.csv=self.root/'calls.csv'
        with self.csv.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id','merged_text','label']);w.writeheader()
            for rows in self.splits.values():
                for row in rows:w.writerow(dict(id=row['id'],merged_text='甲乙',label='信访风险件' if row['label'] else '正常'))
        self.prep=self.root/'prepared';self.prep.mkdir();self.tok.save_pretrained(self.prep/'tokenizer')
        (self.prep/'selected.jsonl').write_text('\n'.join(json.dumps(r) for rows in self.splits.values() for r in rows))
        meta=dict(method='recent',length_schema=LENGTH_SCHEMA,csv=str(self.csv),csv_sha256=sha256(self.csv),
            smoke_test=True,max_length=64,classifier_max_length=32,split_ids={s:[r['id'] for r in rs] for s,rs in self.splits.items()})
        (self.prep/'prepare_config.json').write_text(json.dumps(meta))

    def tearDown(self):self.temp.cleanup()

    def args(self,out):
        return SimpleNamespace(prepared=self.prep,output_dir=out,classifier_model=str(self.base),epochs=1,head_epochs=2,
            head_lr=.001,encoder_lr=2e-6,finetune_head_lr=2e-5,effective_batch_size=64,train_token_budget=128,
            seed=42,num_threads=2,device='cpu',amp='none',gradient_checkpointing=False,batch_size=2)

    def test_bert_head_cache_matches_normal_forward(self):
        b=self.tok.pad([dict(input_ids=[2,5,3],attention_mask=[1,1,1])],return_tensors='pt')
        with torch.no_grad():
            torch.testing.assert_close(head_logits(self.model,head_features(self.model,b)),self.model(**b).logits)

    def test_all_collapsed_warmup_refuses_checkpoint(self):
        out=self.root/'failed';out.mkdir()
        with patch('table_baselines.stable_training.head_features',side_effect=lambda m,b:torch.zeros(len(b['input_ids']),16)):
            with self.assertRaisesRegex(ValueError,'refusing to publish'):
                warmup(copy.deepcopy(self.model),self.splits,self.tok,out,'cpu',epochs=2)
        self.assertTrue((out/'failure.json').exists())
        self.assertFalse((out/'classifier').exists())

    def test_collapsed_epoch_cannot_replace_healthier_checkpoint(self):
        # Constant AP=.5 would beat this deliberately poor but nonconstant AP.
        healthy=np.array([.9,.8,.1,.2]);constant=np.full(4,.3)
        out=self.root/'guarded'
        from table_baselines.runtime import save_model
        with patch('table_baselines.stable_training.warmup',return_value=(.4,1.)), \
             patch('table_baselines.stable_training.train_window',return_value=(1.,1.)), \
             patch('table_baselines.stable_training.probabilities',side_effect=[healthy,constant,healthy,healthy]), \
             patch('table_baselines.stable_training.save_model',wraps=save_model) as saved:
            train(self.args(out))
        self.assertEqual(saved.call_count,1)
        config=json.loads((out/'run_config.json').read_text())
        self.assertEqual(config['best_stage'],'head_warmup')
        self.assertTrue(config['stopped_for_collapse'])

    def test_near_constant_and_nan_detected(self):
        self.assertIsNotNone(collapse_reason(diagnostics(self.splits['val'],np.array([.1,.1,.1,.10000001]))))
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            diagnostics(self.splits['val'],np.array([.1,np.nan,.2,.3]))

    def test_all_six_method_entries_use_stable_defaults(self):
        from table_baselines.entry import run
        cases={'Recent_Truncation':'recent','Dense_TopK_BGE_M3':'dense','SAGE':'sage','CARROT':'carrot','LongLLMLingua':'longllmlingua','RECOMP_Extractive':'recomp'}
        for folder,method in cases.items():
            meta=json.loads((self.prep/'prepare_config.json').read_text());meta['method']=method
            (self.prep/'prepare_config.json').write_text(json.dumps(meta))
            with patch('table_baselines.stable_training.train') as trainer:
                run(method,'train',self.root/'Selection_Retrieval'/folder,
                    ['--prepared',str(self.prep),'--classifier-model',str(self.base),'--device','cpu'])
            args=trainer.call_args.args[0]
            self.assertEqual(args.encoder_lr,2e-6);self.assertEqual(args.head_epochs,8)
            self.assertEqual(args.effective_batch_size,64);self.assertEqual(args.output_dir.name,'model_final')

    def test_cog_entry_uses_stable_joint_defaults(self):
        from table_baselines.entry import run
        with patch('table_baselines.stable_cogltx.train') as trainer:
            run('cogltx','train-cogltx',self.root/'Selection_Retrieval'/'CogLTX',
                ['--classifier-model',str(self.base),'--encoder',str(self.base),'--device','cpu'])
        args=trainer.call_args.args[0]
        self.assertEqual(args.effective_batch_size,64)
        self.assertEqual(args.cache_dir.name,'model_final_blocks')

    def test_old_high_lr_command_fails_clearly(self):
        from table_baselines.cli import main
        with self.assertRaises(SystemExit) as exc:
            main(['train','--prepared',str(self.prep),'--output-dir',str(self.root/'bad'),'--lr','2e-5'])
        self.assertEqual(exc.exception.code,2)
        self.assertFalse((self.root/'bad').exists())

    def test_changed_csv_is_rejected_before_training(self):
        meta=json.loads((self.prep/'prepare_config.json').read_text())
        self.csv.write_bytes(self.csv.read_bytes()+b'\n')
        with self.assertRaisesRegex(ValueError,'Data source changed'):
            validate_prepared(self.prep,meta,[r for rs in self.splits.values() for r in rs])

    def test_changed_labels_and_overlapping_splits_are_rejected(self):
        meta=json.loads((self.prep/'prepare_config.json').read_text())
        rows=copy.deepcopy([r for rs in self.splits.values() for r in rs])
        rows[0]['label']=1
        with self.assertRaisesRegex(ValueError,'labels differ'):
            validate_prepared(self.prep,meta,rows)
        rows[0]['label']=0;rows[-1]['id']=rows[0]['id']
        with self.assertRaisesRegex(ValueError,'duplicate IDs'):
            validate_prepared(self.prep,meta,rows)

    def test_default_model_ignores_incomplete_runs(self):
        from table_baselines.entry import stable_model_default
        folder=self.root/'Selection_Retrieval'/'method';artifacts=folder/'artifacts'
        source=self.root/'merged_最终版.xlsx';source.write_bytes(b'current source fingerprint fixture')
        config=dict(stable_training=True,preparation={'csv_sha256':sha256(source)})
        stable=artifacts/'model_20261022_stable';stable.mkdir(parents=True)
        (stable/'run_config.json').write_text(json.dumps(config))
        (stable/'completion.json').write_text(json.dumps(dict(status='complete')))
        incomplete=artifacts/'model_final';incomplete.mkdir()
        (incomplete/'run_config.json').write_text(json.dumps(config))
        self.assertEqual(stable_model_default(folder),stable)
        (incomplete/'completion.json').write_text(json.dumps(dict(status='complete')))
        self.assertEqual(stable_model_default(folder),incomplete)
        # Old-source weights must not be auto-selected after a data replacement.
        config['preparation']['csv_sha256']='old source'
        (incomplete/'run_config.json').write_text(json.dumps(config))
        self.assertEqual(stable_model_default(folder),stable)


if __name__=='__main__':unittest.main()
