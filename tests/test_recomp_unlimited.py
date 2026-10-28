"""Unlimited original context must not disable downstream budget enforcement."""
import unittest
from pathlib import Path
from unittest.mock import patch
from transformers import AutoTokenizer
from table_baselines import cli
from table_baselines.input_budget import recent_input
from table_baselines.budget import TokenBudget
from Context_Compression.RECOMP_Extractive.selector import Recomp

class Encoder:
    def scores(self, query, documents):
        return list(range(len(documents)))

class UnlimitedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            Path(__file__).resolve().parents[1]/'models/BAAI--bge-m3', local_files_only=True)

    def test_unlimited_preserves_original_and_final_budget(self):
        text = '开头的重要信息。\n' + '这是一段很长的通话内容。'*1000
        retained, info = recent_input(text, self.tokenizer, 0)
        self.assertEqual(retained, text)
        self.assertGreater(info['retained_input_tokens'], 4096)
        self.assertEqual(info['input_truncation'], 'none')
        self.assertFalse(info['input_truncated'])
        self.assertEqual(info['original_input_tokens'], info['retained_input_tokens'])
        selection = Recomp(TokenBudget(self.tokenizer, 64), Encoder(), '风险', None).select(retained)
        self.assertLessEqual(len(selection.token_ids), 64)
        self.assertEqual(selection.selected_indices, sorted(selection.selected_indices))

    def test_positive_still_keeps_suffix(self):
        text = '开头。\n' + '中间通话。'*100 + '末尾。'
        retained, info = recent_input(text, self.tokenizer, 32)
        self.assertTrue(text.endswith(retained))
        self.assertLessEqual(info['retained_input_tokens'], 32)
        self.assertTrue(info['input_truncated'])
        self.assertEqual(info['input_truncation'], 'recent')
        with self.assertRaises(ValueError):
            recent_input(text, self.tokenizer, -1)

    def test_cli_zero_aliases_and_default(self):
        for flag in ('--max_length', '--max-length'):
            with patch('table_baselines.models.resolve_models'), patch.object(cli, 'prepare') as prepare:
                cli.main(['prepare','--method','recomp','--output-dir','/tmp/not_created',flag,'0'])
                self.assertEqual(prepare.call_args.args[0].max_length, 0)
        with patch('table_baselines.models.resolve_models'), patch.object(cli, 'prepare') as prepare:
            cli.main(['prepare','--method','recomp','--output-dir','/tmp/not_created'])
            self.assertEqual(prepare.call_args.args[0].max_length, 4096)
        for method, value in [('dense', '0'), ('recomp', '-1')]:
            with patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                cli.main(['prepare','--method',method,'--output-dir','/tmp/not_created','--max_length',value])
            self.assertEqual(error.exception.code, 2)

if __name__ == '__main__':
    unittest.main()
