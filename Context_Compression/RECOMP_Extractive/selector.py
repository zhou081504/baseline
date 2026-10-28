"""Sentence extraction ranked by relevance and bounded by classifier tokens."""
from Selection_Retrieval.Dense_TopK_BGE_M3.selector import Dense, rank


class Recomp(Dense):
    def chunks(self, text):
        from Selection_Retrieval.SAGE.segmenter import sentences
        return sentences(text)

    def select(self, text):
        chunks = self.chunks(text)
        scores = self.encoder.scores(self.query, chunks)
        # None ranks every sentence. Explicit top_k retains the legacy cap.
        ranked = rank(scores, self.top_k)
        selected = self.budget.pack(chunks, ranked, chronological=True)
        selected.scores = scores
        selected.details.update(selection_policy='relevance_greedy_token_budget',
                                candidate_sentences=len(chunks), top_k=self.top_k)
        return selected
