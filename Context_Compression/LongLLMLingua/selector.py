"""Adapter for the vendored original Microsoft compressor."""
from pathlib import Path
import sys
from table_baselines.budget import Selection

class LongLingua:
    def __init__(self, budget, model_name, device, query):
        import transformers
        if int(transformers.__version__.split('.')[0]) >= 5:
            raise ValueError('Vendored LongLLMLingua requires the isolated requirements-table.txt environment (transformers 4.44.2), not transformers 5')
        source = Path(__file__).resolve().parent/'upstream'
        sys.path.insert(0, str(source))
        try:
            from longllmlingua_baseline._vendor import PromptCompressor
            from longllmlingua_baseline.baseline import DEFAULTS
        finally:
            sys.path.remove(str(source))
        # 直接使用上游压缩机制，LLMLingua-2 token 分类路线不在本方法内。
        self.compressor = PromptCompressor(model_name=model_name, device_map=device,
                                            model_config={'trust_remote_code':False}, use_llmlingua2=False)
        self.defaults = dict(DEFAULTS, concate_question=False)
        self.budget, self.query = budget, query

    def select(self, text):
        chunks = [line.strip() for line in text.splitlines() if line.strip()] or [text]
        if not text.strip():
            return Selection(self.budget.finish([]), details={'empty_text':True})
        # Native compression token counts use a different tokenizer. Tighten the
        # native target until the downstream budget is met; record any final cap.
        # 生成模型与最终分类器的分词器不同：逐步收紧原生压缩目标，再验证最终预算。
        target = self.budget.content_limit
        for attempt in range(3):
            result = self.compressor.compress_prompt(context=chunks, question=self.query,
                       target_token=target, **self.defaults)
            ids = self.budget.encode(result['compressed_prompt'])
            if len(ids) <= self.budget.content_limit:
                break
            target = max(1, int(target*self.budget.content_limit/len(ids)*0.9))
        # 如仍需最终裁剪，明确写入结果元数据，避免将其隐瞒为原生压缩输出。
        capped = len(ids) > self.budget.content_limit
        return Selection(self.budget.finish(ids[:self.budget.content_limit]), details={
            'native_target_token':target, 'compression_attempts':attempt+1,
            'downstream_cap_applied':capped, 'native_compressed_tokens':result.get('compressed_tokens'),
            'compressed_content_tokens_before_cap':len(ids), 'question_appended':False})
