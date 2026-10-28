import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from table_baselines.segmenter import train_segmenter


class SegmenterDatasetChecks(unittest.TestCase):
    def test_missing_pairs_explains_collection_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=SimpleNamespace(seed=42,output_dir=root/'model',pairs=root/'missing.jsonl')
            with self.assertRaisesRegex(ValueError,'Finish collect_wikipedia.py successfully first'):
                train_segmenter(args)
            self.assertFalse(args.output_dir.exists())

    def test_changed_pairs_are_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pairs=root/'pairs.jsonl'
            pairs.write_text('\n'.join(json.dumps(dict(left='甲',right='乙',label=y)) for y in (0,1)),encoding='utf-8')
            (root/'provenance.json').write_text(json.dumps(dict(pairs_sha256='wrong')))
            args=SimpleNamespace(seed=42,output_dir=root/'model',pairs=pairs)
            with self.assertRaisesRegex(ValueError,'checksum differs'):
                train_segmenter(args)
            self.assertFalse(args.output_dir.exists())


if __name__=='__main__':unittest.main()
