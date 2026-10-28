"""Dispatch only; method implementations live in their category/model folders."""
from Selection_Retrieval.Recent_Truncation.selector import Recent
from Selection_Retrieval.Dense_TopK_BGE_M3.selector import Dense,rank
from Selection_Retrieval.CARROT.selector import Carrot
from Selection_Retrieval.SAGE.selector import Sage
from Context_Compression.RECOMP_Extractive.selector import Recomp
from Context_Compression.LongLLMLingua.selector import LongLingua

DEFAULT_QUERY = '从这通电话里找出与信访风险有关的话。'
DEFAULT_LM = 'Qwen/Qwen2.5-3B-Instruct'

def make_selector(args, budget):
    from .encoders import TextEncoder, Reranker
    if args.method == 'recent':
        return Recent(budget)
    if args.method == 'longllmlingua':
        return LongLingua(budget, args.lm_model, args.device, args.query)
    if args.method == 'recomp':
        # The Chinese adaptation keeps sentence extraction but uses BGE CLS/cosine.
        # Old argument namespaces retain the original official RECOMP behavior.
        style = getattr(args, 'recomp_encoder_style', 'official')
        encoder = TextEncoder(args.recomp_model, args.device, args.selector_batch_size,
                              args.retrieval_max_length,
                              pooling='cls' if style == 'bge' else 'mean',
                              normalize=(style == 'bge'))
        return Recomp(budget, encoder, args.query, args.top_k)
    if args.method == 'cogltx':
        from .cogltx import CogSelector
        return CogSelector(args.selector_checkpoint, budget, args.device, input_max_length=args.max_length)
    encoder = TextEncoder(args.dense_model, args.device, args.selector_batch_size,
                          args.retrieval_max_length, pooling='cls', normalize=True)
    if args.method == 'dense':
        return Dense(budget, encoder, args.query, args.top_k)
    reranker = Reranker(args.reranker_model, args.device, args.selector_batch_size, args.retrieval_max_length)
    if args.method == 'carrot':
        return Carrot(budget, encoder, reranker, args.query, args.top_k,
                      args.mcts_iterations, args.exploration, args.cost_penalty)
    if args.method == 'sage':
        from .segmenter import Segmenter
        from .llm import LocalFeedback
        segmenter = Segmenter(args.segmenter_checkpoint, args.device, args.selector_batch_size)
        feedback = LocalFeedback(args.lm_model, args.feedback_device)
        return Sage(budget, encoder, reranker, segmenter, feedback, args)
    raise ValueError(f'Unsupported method {args.method}')
