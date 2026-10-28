"""Bounded-batch implementations of official dense and RECOMP scoring."""
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class TextEncoder:
    def __init__(self, model_name, device='cpu', batch_size=1, max_length=512, *, pooling='cls', normalize=True):
        if batch_size < 1 or max_length < 1:
            raise ValueError('batch_size and max_length must be positive')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.pooling, self.normalize = pooling, normalize
        # XLM-R reserves two position IDs. tokenizer.model_max_length is the
        # effective text limit, unlike config.max_position_embeddings.
        limit = min(getattr(self.model.config, 'max_position_embeddings', max_length),
                    self.tokenizer.model_max_length)
        if max_length > limit:
            raise ValueError(f'retrieval_max_length={max_length} exceeds model limit {limit}')

    @torch.inference_mode()
    def encode(self, texts):
        result = []
        for start in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[start:start+self.batch_size], return_tensors='pt', padding=True,
                                   truncation=True, max_length=self.max_length).to(self.device)
            hidden = self.model(**batch).last_hidden_state
            # Dense/BGE 使用 CLS；RECOMP 使用 attention_mask 加权的 mean，二者不能混用。
            if self.pooling == 'cls':
                pooled = hidden[:, 0]
            elif self.pooling == 'mean':
                mask = batch['attention_mask'].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                raise ValueError(f'Unsupported pooling: {self.pooling}')
            # BGE dense 向量做 L2 归一化；官方 RECOMP 关闭该选项保留未归一化点积。
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            result.append(pooled.float().cpu())
        return torch.cat(result) if result else torch.empty(0, self.model.config.hidden_size)

    def scores(self, query, documents):
        if not documents:
            return []
        q = self.encode([query])[0]
        return (self.encode(documents) @ q).tolist()


class Reranker:
    def __init__(self, model_name, device='cpu', batch_size=1, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        if self.model.config.num_labels != 1:
            raise ValueError('Reranker must expose one relevance logit, not a generic classification head')

    @torch.inference_mode()
    def scores(self, query, documents):
        result = []
        for start in range(0, len(documents), self.batch_size):
            docs = documents[start:start+self.batch_size]
            batch = self.tokenizer([query]*len(docs), docs, padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors='pt').to(self.device)
            result.extend(torch.sigmoid(self.model(**batch).logits.float()).flatten().cpu().tolist())
        return result
