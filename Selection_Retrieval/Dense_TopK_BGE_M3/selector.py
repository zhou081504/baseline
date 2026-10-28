"""BGE-M3 turn retrieval with a downstream token budget."""
def rank(scores, k):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]

class Dense:
    def __init__(self, budget, encoder, query, top_k):
        self.budget, self.encoder, self.query, self.top_k = budget, encoder, query, top_k

    def chunks(self, text):
        # CSV 中每个非空换行视作一个轮次，保留说话人和原始文本。
        return [line.strip() for line in text.splitlines() if line.strip()] or [text]

    def select(self, text):
        chunks = self.chunks(text)
        # 先给整通电话的所有轮次打分；函数只接收文本，不接收风险标签。
        scores = self.encoder.scores(self.query, chunks)
        # TopK 是候选条数，最终仍要按 BERT token 预算装入，并恢复原始顺序。
        selected = self.budget.pack(chunks, rank(scores, self.top_k))
        selected.scores = scores
        return selected
