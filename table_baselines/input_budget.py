"""A complete CSV row remains one sample; truncate only its oldest text."""
import bisect

LENGTH_SCHEMA = 'input_and_classifier_v1'
DEFAULT_INPUT_MAX_LENGTH = 4096
DEFAULT_CLASSIFIER_MAX_LENGTH = 4096


def recent_input(text, tokenizer, max_length):
    """Keep the latest original-text suffix within a content-token budget.

    Offsets preserve speaker markers and newlines. Decoding a sliced token list
    would normalize whitespace and could destroy the downstream turn boundaries.
    The input limit is independent of any encoder's position-embedding limit.
    """
    if max_length < 0:
        raise ValueError('Input max_length must be nonnegative (0 disables truncation)')
    if max_length == 0:
        count = len(tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False))
        return text, {
            'input_max_length': 0, 'original_input_tokens': count,
            'retained_input_tokens': count, 'input_truncated': False,
            'input_truncation': 'none',
        }
    if not tokenizer.is_fast:
        raise ValueError('Input truncation requires a fast tokenizer with character offsets')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
    original = len(encoded['input_ids'])
    retained = original
    start = 0
    if original > max_length:
        offsets = encoded['offset_mapping'][-max_length:]
        candidates = sorted({a for a, b in offsets if b > a} | {len(text)})
        start = candidates[0]
        retained = len(tokenizer.encode(text[start:], add_special_tokens=False, verbose=False))
        # 一个Unicode字符可能对应多个token；切口重新编码后再次核对预算。
        # 只向后移动原文切口，不增补/改写内容，不切断字符。
        while retained > max_length:
            index = bisect.bisect_right(candidates, start)
            start = candidates[index] if index < len(candidates) else len(text)
            retained = len(tokenizer.encode(text[start:], add_special_tokens=False, verbose=False))
    return text[start:], {
        'input_max_length': max_length,
        'original_input_tokens': original,
        'retained_input_tokens': retained,
        'input_truncated': start > 0,
        'input_truncation': 'recent',
    }
