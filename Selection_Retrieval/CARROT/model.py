"""CARROT MCTS adaptation with strict downstream-tokenizer budget enforcement.

Based on wang0702/CARROT core/mcts.py and core/node.py (MIT).
Uses a fixed, explicitly configured reranker; the optional configuration agent
is not trained on dummy data or silently substituted.
"""
import math
from dataclasses import dataclass, field


@dataclass
class Node:
    state: tuple = ()
    parent: object = None
    children: list = field(default_factory=list)
    visits: int = 0
    reward_sum: float = 0.0
    score: float | None = None
    cost: int = 0
    expanded: bool = False


def search(query, chunks, reranker, budget, *, iterations=30, exploration=1.41, penalty=1.0):
    if iterations < 1:
        raise ValueError('MCTS iterations must be positive')
    # 所有节点统一用最终分类器 tokenizer 计数，避免检索模型 token 数不一致。
    encoded = [budget.encode(text) for text in chunks]
    # All nodes are feasible; unlike the upstream fallback, over-budget nodes
    # must never enter the output when no feasible node exists.
    root = Node()
    explored = []
    def available(node):
        return not node.expanded or any(available(child) for child in node.children)
    for _ in range(iterations):
        if not available(root):
            break
        node = root
        # UCB 在已知高分与未充分探索的序列之间选择，并扣除长度成本。
        while node.expanded and node.children:
            node = max([child for child in node.children if available(child)], key=lambda child: (
                child.reward_sum / child.visits + exploration * math.sqrt(math.log(max(1, node.visits))/child.visits)
                - penalty * child.cost / budget.content_limit
            ) if child.visits else float('inf'))
        node.expanded = True
        for i in range(len(chunks)):
            cost = node.cost + len(encoded[i])
            # 严格预算：超长单元不会进入搜索树，也不会作为越界兜底结果。
            if i not in node.state and cost <= budget.content_limit and encoded[i]:
                node.children.append(Node(node.state + (i,), node, cost=cost))
        if not node.children:
            continue
        texts = ['\n'.join(chunks[i] for i in child.state) for child in node.children]
        scores = reranker.scores(query, texts)
        for child, score in zip(node.children, scores):
            child.score = float(score)
            explored.append(child)
            current = child
            while current is not None:
                current.visits += 1
                current.reward_sum += float(score)
                current = current.parent
    if not explored:
        return [], {'evaluated_combinations': 0, 'no_feasible_whole_chunk': True}
    # 返回整段序列打分最高的可行节点，保持 MCTS 搜索出的顺序。
    best = max(explored, key=lambda n: (n.score, n.visits, len(n.state)))
    return list(best.state), {'evaluated_combinations': len(explored), 'reranker_score': best.score,
                              'content_tokens': best.cost, 'fixed_configuration': True}
