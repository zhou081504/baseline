"""8192-token classification backbone and memory-efficient exact attention.

The server pins transformers 4.44.2, whose XLM-R implementation does not expose
SDPA. This local adapter keeps pretrained parameter names and standard HF
checkpoints; it changes attention computation, not positions or token budgets.
"""
from contextlib import nullcontext

import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaSelfAttention

DEFAULT_CLASSIFIER = 'BAAI/bge-m3'


class SDPASelfAttention(XLMRobertaSelfAttention):
    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):
        # 普通双向编码器走 SDPA；不支持的高级选项保留原库语义。
        if (head_mask is not None or encoder_hidden_states is not None
                or past_key_value is not None or output_attentions
                or self.is_decoder or self.position_embedding_type != 'absolute'):
            return super().forward(hidden_states, attention_mask, head_mask,
                                   encoder_hidden_states, encoder_attention_mask,
                                   past_key_value, output_attentions)
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        # 保留 padding/删块的加性 mask；不能丢弃，否则 CogLTX 的删块监督会失效。
        mask = attention_mask.to(query.dtype) if attention_mask is not None else None
        context = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False)
        context = context.permute(0, 2, 1, 3).contiguous()
        return (context.view(*context.size()[:-2], self.all_head_size),)


def enable_sdpa(model):
    count = 0
    for module in model.modules():
        if type(module) is XLMRobertaSelfAttention:
            # 直接复用已加载的 Q/K/V 参数，避免重新初始化或复制大模型权重。
            module.__class__ = SDPASelfAttention
            count += 1
    if count:
        model.config.long_context_attention = 'torch_sdpa'
    return model


def load_classifier(path, *, num_labels=2, **kwargs):
    return enable_sdpa(AutoModelForSequenceClassification.from_pretrained(
        path, num_labels=num_labels, **kwargs))


def load_introspector(path, *, num_labels=1, **kwargs):
    return enable_sdpa(AutoModelForTokenClassification.from_pretrained(
        path, num_labels=num_labels, **kwargs))


def resolve_amp(value, device):
    if value == 'auto':
        value = ('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if device == 'cuda' else 'none'
    if value != 'none' and device != 'cuda':
        raise ValueError('Mixed precision requires --device cuda; use --amp none for CPU')
    if value == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16; use --amp fp16')
    return value


def autocast_context(device, amp):
    if amp == 'none':
        return nullcontext()
    return torch.autocast(device_type=device, dtype=torch.bfloat16 if amp == 'bf16' else torch.float16)
