"""Explicit GPU length validation on synthetic input; never reports benchmark metrics."""
import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from table_baselines.long_context import load_classifier, autocast_context, SDPASelfAttention
from table_baselines.runtime import save_model, validate_length


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sequence-length',type=int,default=4096)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    output = ROOT/f'docs/validation/gpu_{args.sequence_length}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'synthetic_input': True, 'benchmark_metrics': False, 'status': 'running'}
    output.write_text(json.dumps(report))
    try:
        if torch.cuda.mem_get_info()[0] < 18 * 1024**3:
            raise RuntimeError('Less than 18 GiB free; defer GPU validation without touching other jobs')
        source = ROOT/'models/BAAI--bge-m3'
        tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
        model = load_classifier(source, local_files_only=True).to('cuda')
        validate_length(tok, model.config, args.sequence_length)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        layers = sum(isinstance(m, SDPASelfAttention) for m in model.modules())
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, foreach=False)
        ids = torch.tensor([tok.build_inputs_with_special_tokens([100] * (args.sequence_length-2))], device='cuda')
        assert ids.shape == (1, args.sequence_length)
        batch = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
        batch['attention_mask'][:, min(300,args.sequence_length//2):min(350,args.sequence_length)] = 0  # Also exercise CogLTX's erasure mask.
        target = torch.tensor([1], device='cuda')
        # Keep the actual internal CogLTX model and optimizer resident too.
        internal = AutoModelForTokenClassification.from_pretrained(
            '/data/models/chinese-bert-wwm-ext', num_labels=1, local_files_only=True).to('cuda')
        internal.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        iopt = torch.optim.AdamW(internal.parameters(), lr=2e-5, foreach=False)
        with autocast_context('cuda', 'bf16'):
            ilogits = internal(input_ids=ids[:, :512], attention_mask=torch.ones_like(ids[:, :512])).logits
            iloss = ilogits.float().square().mean()
        iloss.backward()
        iopt.step()
        iopt.zero_grad(set_to_none=True)
        del ilogits, iloss
        torch.cuda.reset_peak_memory_stats()
        losses = []
        for _ in range(2):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context('cuda', 'bf16'):
                loss = model(**batch, labels=target).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite full-length training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss))
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.inference_mode(), autocast_context('cuda', 'bf16'):
            before = model(**batch).logits.float().cpu()
        report.update(sequence_length=args.sequence_length, internal_sequence_length=512,
            sdpa_layers=layers, amp='bf16', training_steps=2,
            training_losses=losses, gradients_finite=bool(torch.isfinite(norm)),
            peak_allocated_gib=round(torch.cuda.max_memory_allocated()/1024**3, 3),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved()/1024**3, 3))
        with tempfile.TemporaryDirectory(prefix='classifier_reload_', dir=ROOT) as saved:
            save_model(model, saved)
            del model, optimizer, internal, iopt, loss
            gc.collect()
            torch.cuda.empty_cache()
            loaded = load_classifier(saved, local_files_only=True).to('cuda').eval()
            with torch.inference_mode(), autocast_context('cuda', 'bf16'):
                after = loaded(**batch).logits.float().cpu()
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            report['checkpoint_reload_equal'] = True
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['seconds'] = round(time.monotonic()-started, 2)
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
