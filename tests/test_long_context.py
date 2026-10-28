"""Attention correctness and different-tokenizer CogLTX regression tests."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import XLMRobertaConfig, XLMRobertaForSequenceClassification, AutoTokenizer
from table_baselines.long_context import enable_sdpa, load_classifier
from table_baselines.runtime import save_model, validate_length
from table_baselines.input_budget import recent_input
from Selection_Retrieval.CogLTX.model import dual_blocks, selector_windows, pack_blocks, fit, rank_blocks


class LongContextTests(unittest.TestCase):
    def test_masked_attention_outputs_gradients_and_reload(self):
        torch.manual_seed(19)
        config = XLMRobertaConfig(vocab_size=100, hidden_size=24, num_hidden_layers=2,
            num_attention_heads=2, intermediate_size=48, max_position_embeddings=8194,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0., num_labels=2)
        eager = XLMRobertaForSequenceClassification(config)
        sdpa = enable_sdpa(copy.deepcopy(eager))
        ids = torch.randint(3, 100, (2, 35))
        mask = torch.ones_like(ids)
        mask[0, 8:13] = 0  # CogLTX 删除块产生的非末尾 mask。
        mask[1, 29:] = 0
        a = eager(input_ids=ids, attention_mask=mask).logits
        b = sdpa(input_ids=ids, attention_mask=mask).logits
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        a.square().sum().backward()
        b.square().sum().backward()
        for (_, x), (_, y) in zip(eager.named_parameters(), sdpa.named_parameters()):
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            save_model(sdpa, directory)
            restored = load_classifier(directory, local_files_only=True).eval()
            torch.testing.assert_close(b, restored(input_ids=ids, attention_mask=mask).logits)

    def test_position_offset_boundary(self):
        class Tokenizer:
            model_max_length = 10**30
            def num_special_tokens_to_add(self, pair=False):
                return 2
        config = XLMRobertaConfig(max_position_embeddings=8194, pad_token_id=1)
        validate_length(Tokenizer(), config, 8192)
        with self.assertRaisesRegex(ValueError, 'encoder limit 8192'):
            validate_length(Tokenizer(), config, 8193)

    def test_cogltx_real_different_tokenizers_and_budgets(self):
        root = Path(__file__).resolve().parents[1]
        final_path = root/'models/BAAI--bge-m3'
        internal_path = Path('/data/models/chinese-bert-wwm-ext')
        if not final_path.exists() or not internal_path.exists():
            self.skipTest('Real local tokenizer files unavailable')
        internal = AutoTokenizer.from_pretrained(internal_path, local_files_only=True)
        final = AutoTokenizer.from_pretrained(final_path, local_files_only=True)
        text = '用户：我反复投诉，没有解决这个问题。\n客服：请说明您的情况。\n' * 400
        selected, classified = dual_blocks(text, internal, final, 64, 512, 4096)
        self.assertEqual(len(selected), len(classified))
        indices = fit(classified, range(len(classified)), 4094)
        batch, spans = pack_blocks(classified, indices, final, 'cpu')
        self.assertGreater(batch['input_ids'].shape[1], 512)
        self.assertLessEqual(batch['input_ids'].shape[1], 4096)
        windows = list(selector_windows(selected, indices, 512))
        self.assertGreater(len(windows), 1)
        self.assertEqual([i for w in windows for i in w], indices)
        for window in windows:
            inputs, _ = pack_blocks(selected, window, internal, 'cpu')
            self.assertLessEqual(inputs['input_ids'].shape[1], 512)
        self.assertEqual([i for i, _, _ in spans], indices)


    def test_real_tokenizer_recent_cap_preserves_original_suffix(self):
        source=Path(__file__).resolve().parents[1]/'models/BAAI--bge-m3'
        if not source.exists():self.skipTest('Local BGE tokenizer unavailable')
        tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
        text='甲：以前的内容。\n'*100+'乙：最后的投诉😕问题。\n客服：收到。'
        for cap in [1,2,8,17,32768]:
            retained,meta=recent_input(text,tokenizer,cap)
            self.assertTrue(text.endswith(retained))
            self.assertLessEqual(len(tokenizer.encode(retained,add_special_tokens=False)),cap)
            if cap==32768:self.assertEqual(retained,text)
            if cap==17:self.assertIn('\n',retained)

    def test_judge_ranking_selects_high_scores_and_restores_order(self):
        from types import SimpleNamespace
        class Tokenizer:
            cls_token_id=0
            sep_token_id=2
        class Judge:
            def eval(self):return self
            def __call__(self,input_ids,attention_mask):
                logits=input_ids.float().unsqueeze(-1)/10
                return SimpleNamespace(logits=logits)
        blocks=[[10,2],[30,2],[20,2]]
        selected,scores=rank_blocks(Judge(),blocks,blocks,Tokenizer(),6,'cpu')
        self.assertEqual(selected,[1,2])
        self.assertGreater(scores[1],scores[2])
        batch,_=pack_blocks(blocks,selected,Tokenizer(),'cpu')
        self.assertEqual(batch['input_ids'].shape[1],6)


if __name__ == '__main__':
    unittest.main()
