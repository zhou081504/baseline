"""Dedicated LongLLMLingua interface over the unmodified upstream core."""
DEFAULTS = dict(
    condition_in_question="after_condition",
    reorder_context="sort",
    dynamic_context_compression_ratio=0.4,
    condition_compare=True,
    context_budget="+100",
    rank_method="longllmlingua",
    use_context_level_filter=True,
    use_sentence_level_filter=False,
    use_token_level_filter=True,
    concate_question=True,
)


class LongLLMLingua:
    def __init__(self, model_name="NousResearch/Llama-2-7b-hf", device="cuda"):
        from ._vendor import PromptCompressor
        self.model_name = model_name
        self.device = device
        self.compressor = PromptCompressor(
            model_name=model_name, device_map=device,
            model_config={"trust_remote_code": False},
            use_llmlingua2=False, use_slingua=False,
        )

    def compress(self, context, question, instruction="", rate=0.5, target_token=-1):
        if isinstance(context, str):
            context = [context]
        if not isinstance(context, list) or not context or not all(
            isinstance(x, str) and x.strip() for x in context
        ):
            raise ValueError("context must be a nonempty string or list of nonempty strings")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("LongLLMLingua requires a nonempty question")
        if not isinstance(instruction, str):
            raise ValueError("instruction must be a string")
        if not 0 < rate <= 1:
            raise ValueError("rate must be in (0, 1]")
        if target_token != -1 and target_token <= 0:
            raise ValueError("target_token must be -1 or positive")
        return self.compressor.compress_prompt(
            context=context, question=question, instruction=instruction,
            rate=rate, target_token=target_token, **DEFAULTS,
        )
