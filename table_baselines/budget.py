"""Count context budgets with the downstream tokenizer, including special tokens."""
from dataclasses import dataclass, field


@dataclass
class Selection:
    token_ids: list[int]
    selected_indices: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class TokenBudget:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.content_limit = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        if self.content_limit < 1:
            raise ValueError('max_length must leave room for content and special tokens')

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def finish(self, ids):
        if len(ids) > self.content_limit:
            raise ValueError('Selected content exceeds the token budget')
        result = self.tokenizer.build_inputs_with_special_tokens(ids)
        if len(result) > self.max_length:
            raise ValueError('Tokenizer special-token budget mismatch')
        return result

    def recent(self,text):
        # Keep existing callers working; the actual baseline lives in its folder.
        from Selection_Retrieval.Recent_Truncation.selector import Recent
        return Recent(self).select(text)

    def pack(self, texts, ranked_indices, *, chronological=True):
        """Greedy whole-unit packing; if none fits, truncate only the best unit.

        Long selected units are skipped while a whole shorter unit can fit.
        The single-unit fallback is recorded explicitly for reproducibility.
        """
        encoded = {i: self.encode(texts[i]) for i in ranked_indices}
        kept, length = [], 0
        for i in ranked_indices:
            if encoded[i] and length + len(encoded[i]) <= self.content_limit:
                kept.append(i)
                length += len(encoded[i])
        truncated = False
        if not kept and ranked_indices:
            kept = [ranked_indices[0]]
            encoded[kept[0]] = encoded[kept[0]][:self.content_limit]
            truncated = True
        if chronological:
            kept.sort()
        ids = [token for i in kept for token in encoded[i]]
        return Selection(self.finish(ids), kept, details={'truncated_single_unit': truncated, 'kept_content_tokens': len(ids)})
