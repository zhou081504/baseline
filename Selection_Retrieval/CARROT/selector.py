"""Retrieval followed by budget-constrained MCTS."""
from Selection_Retrieval.Dense_TopK_BGE_M3.selector import Dense,rank

class Carrot(Dense):
    def __init__(self, budget, encoder, reranker, query, top_k, iterations, exploration, penalty):
        super().__init__(budget, encoder, query, top_k)
        self.reranker, self.iterations, self.exploration, self.penalty = reranker, iterations, exploration, penalty

    def select(self, text):
        from .model import search
        chunks = self.chunks(text)
        candidates = rank(self.encoder.scores(self.query, chunks), self.top_k)
        texts = [chunks[i] for i in candidates]
        # 在检索候选中搜索组合与顺序；只使用相关性打分，不读取真实风险标签。
        indices, details = search(self.query, texts, self.reranker, self.budget,
                                  iterations=self.iterations, exploration=self.exploration, penalty=self.penalty)
        selected = self.budget.pack(texts, indices, chronological=False)
        selected.selected_indices = [candidates[i] for i in selected.selected_indices]
        selected.details.update(details)
        # Preserve the sequence optimized by MCTS, not chronological order.
        return selected
