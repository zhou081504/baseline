"""Regression checks for cache proof, precision, batching and deletion masks."""
import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification

from table_baselines.refresh_prepared import restored_hash
from table_baselines.runtime import binary_logits
from table_baselines.stable_training import pad, batches, train_window
from table_baselines.stable_cogltx import block_scores, select_rows, erasure_losses, deletion_targets
from Selection_Retrieval.CogLTX.model import rank_blocks


class StableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp=tempfile.TemporaryDirectory()
        path=Path(cls.tmp.name)/'vocab.txt'
        path.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','甲','乙','丙','丁']))
        cls.tok=BertTokenizerFast(vocab_file=str(path))
        cfg=BertConfig(vocab_size=len(cls.tok),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
            intermediate_size=32,max_position_embeddings=64,hidden_dropout_prob=0.,attention_probs_dropout_prob=0.,num_labels=1)
        torch.manual_seed(7);cls.judge=BertForTokenClassification(cfg).eval()
        cfg2=BertConfig(**{**cfg.to_dict(),'num_labels':2,'id2label':{0:'N',1:'P'},'label2id':{'N':0,'P':1}})
        cls.reasoner=BertForSequenceClassification(cfg2).eval()

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_cache_proof_accepts_only_label_changes(self):
        rows=[dict(id='a',merged_text='甲\n乙',label='正常'),dict(id='b',merged_text='丙',label='正常')]
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
        expected=hashlib.sha256(b'\xef\xbb\xbf'+stream.getvalue().encode()).hexdigest()
        rows[1]['label']='信访风险件'
        self.assertEqual(restored_hash(rows,list(rows[0]),{'a':0,'b':0}),expected)
        rows[1]['merged_text']='丁'
        self.assertNotEqual(restored_hash(rows,list(rows[0]),{'a':0,'b':0}),expected)

    def test_margin_uses_float32(self):
        logits=torch.tensor([[.5,.50390625]],dtype=torch.bfloat16)
        result=binary_logits(lambda **kwargs:SimpleNamespace(logits=logits),{})
        self.assertEqual(result.dtype,torch.float32)

    def test_batched_block_scores_match_sequential(self):
        blocks=[[5,6,3],[7,3],[8,5,6,3]]
        _,sequential=rank_blocks(self.judge,blocks,blocks,self.tok,8,'cpu')
        rows=[dict(token_ids=[2]+b+[3]) for b in blocks]
        actual=block_scores(self.judge,rows,self.tok,'cpu',token_budget=64)
        np.testing.assert_allclose(actual,sequential,atol=1e-7)
        config=dict(classifier_max_length=8,top_k=None)
        items=[dict(id='a',label=1,blocks=blocks,final_blocks=blocks)]
        selected=select_rows(self.judge,items,self.tok,self.tok,config,'cpu')
        expected,_=rank_blocks(self.judge,blocks,blocks,self.tok,8,'cpu')
        self.assertEqual(selected[0]['selected_indices'],expected)
        # Fitting all blocks must avoid the judge entirely, regardless of labels.
        with patch('table_baselines.stable_cogltx.block_scores',side_effect=AssertionError('unneeded scoring')):
            selected=select_rows(self.judge,items,self.tok,self.tok,dict(classifier_max_length=32,top_k=None),'cpu')
        self.assertEqual(selected[0]['selected_indices'],[0,1,2])

    def test_erasure_batch_matches_individual_masks(self):
        rows=[dict(token_ids=[2,5,6,3],label=0),dict(token_ids=[2,7,8,5,3],label=1,attention_mask=[1,0,0,1,1])]
        batched=erasure_losses(self.reasoner,rows,self.tok,'cpu',64)
        serial=np.concatenate([erasure_losses(self.reasoner,[r],self.tok,'cpu',64) for r in rows])
        np.testing.assert_allclose(batched,serial,atol=1e-7)

    def test_fast_saved_model_inference_matches_legacy(self):
        from Selection_Retrieval.CogLTX.model import predict_records
        records=[SimpleNamespace(id=str(i),raw_text='fixture',label=i%2) for i in range(3)]
        config=dict(classifier_max_length=32,top_k=None)
        with patch('Selection_Retrieval.CogLTX.model.sample_blocks',return_value=([[5,6,3],[7,3]],[[5,6,3],[7,3]],{})):
            gold,slow,_=predict_records(self.reasoner,self.judge,records,self.tok,self.tok,config,'cpu')
            actual,fast,_=predict_records(self.reasoner,self.judge,records,self.tok,self.tok,dict(config,stable_training=True),'cpu')
        np.testing.assert_array_equal(actual,gold)
        np.testing.assert_allclose(fast,slow,atol=1e-7)

    def test_accumulation_weights_partial_microbatches(self):
        import copy
        a=copy.deepcopy(self.reasoner);b=copy.deepcopy(self.reasoner)
        rows=[dict(token_ids=[2,5,6,3],label=i%2) for i in range(5)]
        oa=torch.optim.SGD(a.parameters(),lr=.01);ob=torch.optim.SGD(b.parameters(),lr=.01)
        loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.))
        train_window(a,rows,self.tok,oa,loss,'cpu',token_budget=8)
        train_window(b,rows,self.tok,ob,loss,'cpu',token_budget=64)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=1e-7,rtol=1e-6)


if __name__=='__main__':unittest.main()
