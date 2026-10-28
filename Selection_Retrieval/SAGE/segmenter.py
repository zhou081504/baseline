"""Learned SAGE pair segmentation: [x1,x2,x1-x2,x1*x2], sigmoid, MSE.

The paper does not specify MLP widths; hidden size is configurable and recorded.
No untrained/random segmentation model is accepted by the selection pipeline.
"""
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from table_baselines.text import sentences


class PairSegmenter(nn.Module):
    def __init__(self, encoder, hidden_size=256):
        super().__init__()
        self.encoder = encoder
        size = encoder.config.hidden_size
        self.mlp = nn.Sequential(nn.Linear(size*4, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1), nn.Sigmoid())

    def embeddings(self, batch):
        hidden = self.encoder(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, left, right):
        x1, x2 = self.embeddings(left), self.embeddings(right)
        # 相邻句子的语义、差值与乘积共同预测“是否属于同一段落”。
        return self.mlp(torch.cat([x1, x2, x1-x2, x1*x2], dim=-1)).squeeze(-1)


class Segmenter:
    def __init__(self, checkpoint, device='cpu', batch_size=1):
        checkpoint = Path(checkpoint)
        self.config = json.loads((checkpoint/'segmenter_config.json').read_text(encoding='utf-8'))
        if not self.config.get('trained'):
            raise ValueError('SAGE segmenter checkpoint has not been trained')
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint/'encoder', local_files_only=True)
        self.model = PairSegmenter(AutoModel.from_pretrained(checkpoint/'encoder', local_files_only=True),
                                  self.config['hidden_size'])
        self.model.mlp.load_state_dict(torch.load(checkpoint/'mlp.pt', map_location='cpu', weights_only=True))
        self.model.to(device).eval()
        self.device, self.batch_size = device, batch_size

    @torch.inference_mode()
    def split(self, text, threshold=0.55, coarse_length=400):
        parts = sentences(text)
        scores = []
        for i in range(0, len(parts)-1, self.batch_size):
            left = parts[i:min(i+self.batch_size, len(parts)-1)]
            right = parts[i+1:i+1+len(left)]
            encode = lambda xs: self.tokenizer(xs, padding=True, truncation=True,
                                               max_length=self.config['max_length'], return_tensors='pt').to(self.device)
            scores.extend(self.model(encode(left), encode(right)).cpu().tolist())
        chunks, current = [], []
        for i, part in enumerate(parts):
            proposed = '\n'.join(current+[part])
            coarse_break = len(self.tokenizer.encode(proposed, add_special_tokens=False)) > coarse_length
            if current and (scores[i-1] < threshold or coarse_break):
                chunks.append('\n'.join(current))
                current = []
            current.append(part)
        if current:
            chunks.append('\n'.join(current))
        return chunks


def train_segmenter(args):
    """Input JSONL: {left: str, right: str, label: 0|1}; 1 means same paragraph."""
    from table_baselines.runtime import set_seed, save_model, sha256
    set_seed(args.seed)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty segmenter output directory')
    if not Path(args.pairs).is_file():
        raise ValueError('Sentence-pair data is missing. Finish collect_wikipedia.py successfully first, or copy its verified offline dataset; then run train_segmenter.py. Use && between collection and training commands.')
    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding='utf-8').splitlines() if line.strip()]
    provenance = Path(args.pairs).parent/'provenance.json'
    if provenance.is_file():
        record=json.loads(provenance.read_text(encoding='utf-8'))
        if record.get('pairs_sha256') and record['pairs_sha256'] != sha256(args.pairs):
            raise ValueError('Sentence-pair checksum differs from provenance.json; restore the verified dataset')
    if not pairs or any(not p.get('left') or not p.get('right') or p.get('label') not in (0,1) for p in pairs):
        raise ValueError('Provide nonempty sentence-pair JSONL with left, right, label=0|1')
    if {p['label'] for p in pairs} != {0,1}:
        raise ValueError('Segmentation training needs both same-paragraph and boundary examples')
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = PairSegmenter(AutoModel.from_pretrained(args.encoder), args.hidden_size).to(args.device)
    if args.max_length > min(tokenizer.model_max_length, model.encoder.config.max_position_embeddings):
        raise ValueError('Segmenter max_length exceeds encoder limit')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        order = torch.randperm(len(pairs), generator=generator).tolist()
        for start in range(0, len(order), args.batch_size):
            batch = [pairs[i] for i in order[start:start+args.batch_size]]
            encode = lambda key: tokenizer([p[key] for p in batch], padding=True, truncation=True,
                                           max_length=args.max_length, return_tensors='pt').to(args.device)
            target = torch.tensor([p['label'] for p in batch], dtype=torch.float, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            # 此处 label 来自公开百科的段落关系，不是通话风险二分类标签。
            loss = nn.functional.mse_loss(model(encode('left'), encode('right')), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()*len(batch)
        print(f'segmenter epoch={epoch+1} mse={total/len(pairs):.6f}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    save_model(model.encoder,output/'encoder')
    tokenizer.save_pretrained(output/'encoder')
    torch.save(model.mlp.state_dict(), output/'mlp.pt')
    (output/'segmenter_config.json').write_text(json.dumps({
        'trained':True, 'hidden_size':args.hidden_size, 'max_length':args.max_length,
        'encoder':args.encoder, 'pairs':str(Path(args.pairs).resolve()), 'pairs_sha256':sha256(args.pairs), 'seed':args.seed,
        'examples':len(pairs), 'epochs':args.epochs, 'batch_size':args.batch_size, 'lr':args.lr, 'loss':'MSE', 'pooling':'mean',
        'feature_order':['x1','x2','x1-x2','x1*x2']}, indent=2), encoding='utf-8')
