"""CPU integration tests using explicitly synthetic tiny models; no paper metrics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('TIKTOKEN_CACHE_DIR',str(Path(__file__).resolve().parents[1]/'.cache'/'tiktoken'))
import torch
from transformers import BertConfig, BertModel, BertTokenizerFast, BertForSequenceClassification

from table_baselines.budget import TokenBudget
from table_baselines.carrot import search
from table_baselines.sage import gradient_select, feedback_select
from table_baselines.runtime import tokenizer_fingerprint, save_model

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='table_tests_')
        cls.base=Path(cls.temp.name)
        cls.fixture=cls.base/'tiny'
        cls.fixture.mkdir()
        vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]']+list('甲乙丙丁你好电话风险正常投诉没有问题再见。！？：')
        (cls.fixture/'vocab.txt').write_text('\n'.join(dict.fromkeys(vocab)),encoding='utf-8')
        cls.tok=BertTokenizerFast(vocab_file=str(cls.fixture/'vocab.txt'))
        cls.tok.save_pretrained(cls.fixture)
        torch.manual_seed(7)
        # Stable head warmup assumes a pretrained encoder. A wholly random tiny
        # encoder can give genuinely near-identical CLS vectors and should be
        # rejected by the collapse guard. Learn this synthetic fixture first;
        # keep the production guard unchanged and save only its encoder.
        encoder_fixture=BertForSequenceClassification(BertConfig(vocab_size=len(cls.tok),hidden_size=24,num_hidden_layers=1,
                            num_attention_heads=2,intermediate_size=48,max_position_embeddings=128,
                            hidden_dropout_prob=0.,attention_probs_dropout_prob=0.))
        examples=cls.tok(['甲：你好。乙：正常。甲：再见。','甲：你好。乙：电话问题。甲：投诉风险。乙：再见。'],padding=True,return_tensors='pt')
        optimizer=torch.optim.AdamW(encoder_fixture.parameters(),lr=.003)
        for _ in range(50):
            optimizer.zero_grad();loss=encoder_fixture(**examples,labels=torch.tensor([0,1])).loss
            loss.backward();optimizer.step()
        encoder_fixture.bert.save_pretrained(cls.fixture)
        cls.csv=cls.base/'calls.csv'
        import csv
        with cls.csv.open('w',encoding='utf-8',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['id','merged_text','label'])
            for i in range(60):
                writer.writerow([str(i),'甲：你好。\n乙：电话问题。\n甲：投诉风险。\n乙：再见。' if i%2 else '甲：你好。\n乙：正常。\n甲：再见。','信访风险件' if i%2 else '正常'])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def cli(self,*args,code=0,contains=None):
        result=subprocess.run([sys.executable,str(ROOT/'run_table.py'),*map(str,args)],capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,code,result.stdout+'\n'+result.stderr)
        if contains:
            self.assertIn(contains,result.stdout+result.stderr)
        return result

    def test_recent_keeps_latest_tokens_and_special_budget(self):
        budget=TokenBudget(self.tok,6)
        selected=budget.recent('甲乙丙丁你好')
        self.assertEqual(selected.token_ids,[self.tok.cls_token_id]+self.tok.encode('丙丁你好',add_special_tokens=False)+[self.tok.sep_token_id])

    def test_save_transposed_pretrained_weight(self):
        model=BertModel.from_pretrained(self.fixture)
        weight=model.encoder.layer[0].attention.self.query.weight
        weight.data=weight.data.t()
        self.assertFalse(weight.is_contiguous())
        target=self.base/'transposed_checkpoint'
        save_model(model,target)
        loaded=BertModel.from_pretrained(target,local_files_only=True)
        torch.testing.assert_close(weight,loaded.encoder.layer[0].attention.self.query.weight)

    def test_vendored_longllmlingua_qwen_cache_api(self):
        from transformers import Qwen2Config,Qwen2ForCausalLM
        from table_baselines.selectors import LongLingua
        path=self.base/'tiny_qwen'
        self.tok.save_pretrained(path)
        model=Qwen2ForCausalLM(Qwen2Config(vocab_size=len(self.tok),hidden_size=24,
            intermediate_size=48,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,
            max_position_embeddings=2048,pad_token_id=self.tok.pad_token_id,
            bos_token_id=self.tok.cls_token_id,eos_token_id=self.tok.sep_token_id))
        model.save_pretrained(path)
        selector=LongLingua(TokenBudget(self.tok,12),str(path),'cpu','电话风险？')
        result=selector.select('甲：你好。\n乙：投诉风险。\n甲：电话问题。\n乙：再见。')
        self.assertLessEqual(len(result.token_ids),12)
        self.assertFalse(result.details['question_appended'])
        self.assertGreater(result.details['native_compressed_tokens'],0)

    def test_dense_keeps_late_relevant_turn_and_never_receives_label(self):
        from table_baselines.selectors import Dense
        class FakeEncoder:
            def scores(self,query,docs):
                return [10.0 if '风险' in x else 0.0 for x in docs]
        selector=Dense(TokenBudget(self.tok,12),FakeEncoder(),'风险',1)
        result=selector.select('甲：你好。\n乙：正常。\n甲：风险。')
        self.assertEqual(result.selected_indices,[2])
        self.assertLessEqual(len(result.token_ids),12)

    def test_neural_retrieval_cli_and_metadata(self):
        from transformers import BertForSequenceClassification
        reranker=self.base/'tiny_reranker'
        self.tok.save_pretrained(reranker)
        config=BertConfig.from_pretrained(self.fixture)
        config.num_labels=1
        BertForSequenceClassification(config).save_pretrained(reranker)
        for method in ('dense','recomp','carrot'):
            output=self.base/f'{method}_inputs'
            self.cli('prepare','--method',method,'--csv',self.csv,'--classifier-model',self.fixture,
                '--dense-model',self.fixture,'--recomp-model',self.fixture,'--reranker-model',reranker,
                '--retrieval-max-length','64','--max-length','8192','--classifier-max-length','16','--top-k','3','--mcts-iterations','4',
                '--smoke-test','--output-dir',output)
            rows=[json.loads(line) for line in (output/'selected.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),12)
            self.assertTrue(all(len(row['token_ids'])<=16 for row in rows))
            meta=json.loads((output/'prepare_config.json').read_text())
            self.assertFalse(meta['selection_uses_gold_labels'])
            self.assertIn('model_sources',meta)

    def test_model_folder_entrypoints_from_other_working_directory(self):
        folder=ROOT/'Selection_Retrieval/Recent_Truncation'
        prepared=self.base/'folder_prepared'
        model=self.base/'folder_model'
        def run(script,*args,code=0,contains=None):
            result=subprocess.run([sys.executable,str(script),*map(str,args)],cwd=self.base,
                                  capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,code,result.stdout+'\n'+result.stderr)
            if contains:self.assertIn(contains,result.stdout+result.stderr)
        run(folder/'prepare.py','--csv',self.csv,'--classifier-model',self.fixture,
            '--max_length','8192','--classifier_max_length','16','--smoke-test','--output-dir',prepared)
        run(folder/'train.py','--prepared',prepared,'--classifier-model',self.fixture,
            '--output-dir',model,'--effective-batch-size','64')
        run(folder/'infer.py','--prepared',prepared,'--model-dir',model)
        self.assertEqual((model/'test_predictions.csv').read_text(),(model/'infer_test_predictions.csv').read_text())
        # A model folder must not silently train on another method's selected data.
        run(ROOT/'Selection_Retrieval/Dense_TopK_BGE_M3/train.py','--prepared',prepared,
            code=1,contains='another method')
        run(folder/'prepare.py','--method','dense',code=1,contains='fixes the method')

    def test_each_active_model_has_independent_entry_help(self):
        paths=['Selection_Retrieval/Recent_Truncation','Selection_Retrieval/Dense_TopK_BGE_M3',
               'Selection_Retrieval/CogLTX','Selection_Retrieval/SAGE','Selection_Retrieval/CARROT',
               'Context_Compression/LongLLMLingua','Context_Compression/RECOMP_Extractive']
        for path in paths:
            self.assertTrue((ROOT/path/'README.md').is_file())
            result=subprocess.run([sys.executable,str(ROOT/path/'train.py'),'--help'],
                                  cwd=self.base,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_sage_literal_formula_and_feedback(self):
        self.assertEqual(gradient_select([.9,.8,.2],1,2),[0,1])
        self.assertEqual(gradient_select([.9,.8,.2],1,.3),[0])
        calls=[]
        def feedback(query,context):
            calls.append(context)
            return {'score':9 if len(calls)==2 else 3,'adjustment':1}
        result=feedback_select('风险',['甲','乙','丙'],[.9,.8,.2],TokenBudget(self.tok,8),feedback,
                               min_k=1,gradient_divisor=.3)
        self.assertEqual(len(calls),2)
        self.assertEqual(result.selected_indices,[0,1])
        self.assertLessEqual(len(result.token_ids),8)

    def test_carrot_budget_is_hard_and_order_is_searched(self):
        class Scores:
            def scores(self,query,docs):
                return [1.0 if d=='乙\n甲' else .2 for d in docs]
        budget=TokenBudget(self.tok,4)
        indices,details=search('风险',['甲','乙','丙丁你好'],Scores(),budget,iterations=30)
        self.assertEqual(indices,[1,0])
        self.assertLessEqual(details['content_tokens'],budget.content_limit)
        self.assertEqual(search('风险',['甲乙丙'],Scores(),budget)[0],[])

    def test_train_save_offline_infer_and_length_validation(self):
        prepared=self.base/'recent'
        trained=self.base/'trained'
        self.cli('prepare','--method','recent','--csv',self.csv,'--classifier-model',self.fixture,
                 '--output-dir',prepared,'--max_length','32768','--classifier_max_length','24','--smoke-test','--seed','17',
                 '--train-ratio','.6','--val-ratio','.2','--test-ratio','.2')
        self.cli('train','--prepared',prepared,'--classifier-model',self.fixture,'--output-dir',trained,
                 '--effective-batch-size','64','--gradient-checkpointing')
        self.cli('infer','--prepared',prepared,'--model-dir',trained)
        self.assertEqual((trained/'test_predictions.csv').read_text(),(trained/'infer_test_predictions.csv').read_text())
        config=json.loads((trained/'run_config.json').read_text())
        self.assertEqual(config['max_length'],32768)
        self.assertEqual(config['classifier_max_length'],24)
        self.assertTrue(config['preparation']['smoke_test'])
        self.cli('prepare','--method','recent','--csv',self.csv,'--classifier-model',self.fixture,
                 '--output-dir',prepared,code=2,contains='exceeds encoder limit')
        self.cli('prepare','--method','recent','--csv',self.csv,'--classifier-model',self.fixture,
                 '--output-dir',self.base/'bad','--classifier-max-length','129',code=2,contains='exceeds encoder limit')
        self.cli('prepare','--method','recent','--max_length','0',code=2,contains='positive integer')
        self.cli('train','--prepared',prepared,'--output-dir',trained,code=2,contains='not empty')

    def test_cogltx_joint_training_and_label_free_ranking(self):
        from table_baselines.cogltx import CogSelector
        output=self.base/'cog'
        self.cli('train-cogltx','--csv',self.csv,'--encoder',self.fixture,'--classifier-model',self.fixture,'--selector-max-length','64','--output-dir',output,
                 '--max-length','512','--classifier-max-length','24','--block-size','5','--smoke-test','--gradient-checkpointing')
        self.cli('infer-cogltx','--csv',self.csv,'--model-dir',output)
        self.assertEqual((output/'test_predictions.csv').read_text(),(output/'infer_test_predictions.csv').read_text())
        selector=CogSelector(output,TokenBudget(self.tok,24))
        result=selector.select('甲乙丙丁你好。'*30)
        self.assertLessEqual(len(result.token_ids),24)
        self.assertTrue(result.details['block_ranking'])
        self.assertFalse(result.details['memory_replay'])

    def test_cogltx_different_backbones_end_to_end(self):
        from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel
        official=ROOT/'models/BAAI--bge-m3'
        if not official.exists():
            self.skipTest('BGE tokenizer unavailable')
        final=self.base/'tiny_long_classifier'
        tokenizer=AutoTokenizer.from_pretrained(official,local_files_only=True)
        tokenizer.save_pretrained(final)
        XLMRobertaModel(XLMRobertaConfig(vocab_size=len(tokenizer),hidden_size=16,
            num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,
            max_position_embeddings=8194)).save_pretrained(final)
        output=self.base/'cog_dual'
        self.cli('train-cogltx','--csv',self.csv,'--encoder',self.fixture,
            '--classifier-model',final,'--selector-max-length','16',
            '--max-length','32768','--classifier-max-length','64','--block-size','5','--smoke-test','--output-dir',output)
        self.cli('infer-cogltx','--csv',self.csv,'--model-dir',output)
        self.assertEqual((output/'test_predictions.csv').read_text(),(output/'infer_test_predictions.csv').read_text())
        self.assertNotEqual(tokenizer_fingerprint(self.tok),tokenizer_fingerprint(tokenizer))
        self.assertTrue((output/'selector_tokenizer/tokenizer_config.json').is_file())

    def test_all_methods_truncate_complete_samples_before_selector(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from table_baselines.cli import prepare
        from table_baselines.input_budget import LENGTH_SCHEMA
        seen=[]
        class Capture:
            def select(inner,text):
                seen.append(text)
                self.assertLessEqual(len(self.tok.encode(text,add_special_tokens=False)),8)
                return TokenBudget(self.tok,24).recent(text)
        for method in ['recent','dense','cogltx','sage','carrot','recomp','longllmlingua']:
            out=self.base/('cap_'+method)
            args=SimpleNamespace(method=method,classifier_model=str(self.fixture),max_length=8,
                classifier_max_length=24,csv=self.csv,seed=42,train_ratio=.7,val_ratio=.15,
                test_ratio=.15,smoke_test=True,output_dir=out,query='风险',
                segmenter_checkpoint=Path('fake'),selector_checkpoint=Path('fake'))
            with patch('table_baselines.selectors.make_selector',return_value=Capture()):
                prepare(args)
            meta=json.loads((out/'prepare_config.json').read_text())
            self.assertEqual(meta['length_schema'],LENGTH_SCHEMA)
            rows=[json.loads(line) for line in (out/'selected.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),12)
            self.assertEqual(len({row['id'] for row in rows}),12)
            self.assertTrue(all(row['details']['retained_input_tokens']<=8 for row in rows))
            self.assertTrue(all(row['details']['input_truncated'] for row in rows))
        self.assertEqual(len(seen),84)

    def test_segmenter_training_and_loading(self):
        from table_baselines.segmenter import Segmenter
        pairs=self.base/'pairs.jsonl'
        pairs.write_text('\n'.join(json.dumps(p,ensure_ascii=False) for p in [
            {'left':'你好。','right':'电话。','label':1},{'left':'正常。','right':'再见。','label':0},
            {'left':'投诉。','right':'风险。','label':1},{'left':'甲。','right':'乙。','label':0}]),encoding='utf-8')
        output=self.base/'segmenter'
        self.cli('train-segmenter','--pairs',pairs,'--encoder',self.fixture,'--output-dir',output,
                 '--epochs','1','--max-length','24')
        model=Segmenter(output)
        self.assertTrue(model.split('你好。电话。再见。'))


if __name__=='__main__':
    unittest.main(verbosity=2)
