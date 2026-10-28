"""SAGE selection and feedback, following arXiv:2503.01713v3 §§V–VI.

The paper's equation uses score[i] > score[i-1]/g, but reports g=0.3.
With positive descending scores that setting stops immediately after min_k.
Expose g as configuration rather than silently 'correcting' the paper.
"""
import math


def gradient_select(scores, min_k, gradient_divisor):
    if min_k < 1 or not math.isfinite(gradient_divisor) or gradient_divisor <= 0:
        raise ValueError('min_k and gradient_divisor must be positive')
    if any(not math.isfinite(float(s)) or s < 0 for s in scores):
        raise ValueError('SAGE expects finite nonnegative reranker scores')
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    kept = order[:min_k]
    for i in range(min_k, len(order)):
        # 按用户确认保留论文的除法公式；g=0.3 的歧义详见本目录 README。
        if scores[order[i]] > scores[order[i-1]] / gradient_divisor:
            kept.append(order[i])
        else:
            break
    return kept


def feedback_select(query, chunks, scores, budget, feedback, *, min_k=7,
                    gradient_divisor, feedback_threshold=9, max_feedback=3):
    if max_feedback < 1 or not 1 <= feedback_threshold <= 10:
        raise ValueError('Invalid SAGE feedback settings')
    history = []
    current_k = min(max(1, min_k), max(1, len(chunks)))
    for _ in range(max_feedback):
        ranked = gradient_select(scores, current_k, gradient_divisor)
        selected = budget.pack(chunks, ranked)
        # Feedback sees exactly the retained context, never the gold risk label.
        # 反馈只能看到实际预算内保留的文本，不能看到未选内容或真实标签。
        context = budget.tokenizer.decode(selected.token_ids, skip_special_tokens=True)
        result = feedback(query, context)
        score, adjustment = float(result['score']), int(result['adjustment'])
        if not math.isfinite(score) or not 1 <= score <= 10 or adjustment not in (-1, 1):
            raise ValueError('SAGE feedback must have score in [1,10] and adjustment -1 or 1')
        history.append({'min_k': current_k, 'score': score, 'adjustment': adjustment,
                        'selected_indices': selected.selected_indices})
        if score >= feedback_threshold:
            break
        # 依据生成模型反馈增减最少候选数，不用验证/测试标签调整。
        new_k = min(len(chunks), max(1, current_k + adjustment))
        if new_k == current_k:
            break
        current_k = new_k
    selected.details.update(feedback=history, gradient_divisor=gradient_divisor,
                            gradient_rule='score_current > score_previous / g')
    selected.scores = list(scores)
    return selected
