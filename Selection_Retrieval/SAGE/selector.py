"""Learned segmentation, retrieval, gradient selection and feedback."""
from Selection_Retrieval.Dense_TopK_BGE_M3.selector import Dense,rank

class Sage(Dense):
    def __init__(self, budget, encoder, reranker, segmenter, feedback, args):
        super().__init__(budget, encoder, args.query, args.top_k)
        self.reranker, self.segmenter, self.feedback, self.args = reranker, segmenter, feedback, args

    def select(self, text):
        from .model import feedback_select
        # 先学习式分段，再检索、重排和反馈筛选，最后交给公共 BERT 分类器。
        chunks = self.segmenter.split(text, self.args.segmentation_threshold, self.args.coarse_length)
        candidates = rank(self.encoder.scores(self.query, chunks), self.top_k)
        texts = [chunks[i] for i in candidates]
        scores = self.reranker.scores(self.query, texts)
        selected = feedback_select(self.query, texts, scores, self.budget, self.feedback,
                      min_k=self.args.min_k, gradient_divisor=self.args.gradient_divisor,
                      feedback_threshold=self.args.feedback_threshold)
        selected.selected_indices = [candidates[i] for i in selected.selected_indices]
        selected.details['segmented_chunks'] = len(chunks)
        return selected
