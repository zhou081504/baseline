"""Self-contained, non-learning Recent Truncation implementation.

Retain the final content tokens of the complete call. Reserve classifier special
tokens first. No relevance labels, learned selector, retrieval or LLM is used.
"""
from table_baselines.budget import Selection


class Recent:
    def __init__(self,budget):
        self.budget=budget

    def select(self,text):
        # 先编码完整通话，不先按字符或前若干轮截断，才能真正保留末尾内容。
        ids=self.budget.encode(text)
        # content_limit 已扣除 分类器特殊 token；最终长度不会超过 max_length。
        retained=ids[-self.budget.content_limit:]
        return Selection(self.budget.finish(retained),details={
            'original_content_tokens':len(ids),'kept_content_tokens':len(retained)})
