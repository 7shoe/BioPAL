#!/usr/bin/env python3
"""Generic backbone smoke test: offline load from the cache at a pinned revision.

Checks: tokenizer+model load, forward of a padded 2-sequence batch (finite
hidden states, expected shape), one backward step through a tiny regression
head on mean-pooled hidden states (finite, non-zero backbone gradients), and
optionally one long-context forward. Records hidden size, max input length,
token granularity, parameter count and peak GPU memory. Prints one JSON line.

The helpers are importable (tests/ uses them): load_backbone, encode_batch,
forward_hidden, pick_hidden.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

DEFAULT_CACHE = "/homes/csiebenschuh/data/bio_sequences/_hf_lm_cache"
ALPHA = {"dna": "ACGT", "rna": "ACGU", "protein": "ACDEFGHIKLMNPQRSTVWY"}


def seqs(modality: str, lengths: tuple[int, int]) -> list[str]:
    rng = random.Random(17)
    return ["".join(rng.choice(ALPHA[modality]) for _ in range(n)) for n in lengths]


def load_backbone(repo: str, revision: str, cache: str = DEFAULT_CACHE,
                  trust_remote_code: bool = False):
    """Offline tokenizer + model from the cache. Returns (tok, model, base);
    base is the unit to forward: task heads (ForPreTraining, ForMaskedLM) and
    seq2seq decoders are unwrapped."""
    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    try:
        import multimolecule  # noqa: F401  (registers Rna* classes if installed)
    except ImportError:
        pass
    kw = dict(revision=revision, cache_dir=cache,
              trust_remote_code=trust_remote_code)
    try:
        tok = AutoTokenizer.from_pretrained(repo, **kw)
    except Exception:
        # e.g. prot_t5: sentencepiece file not convertible to a fast tokenizer
        tok = AutoTokenizer.from_pretrained(repo, use_fast=False, **kw)
    cfg = AutoConfig.from_pretrained(repo, **kw)
    amap = getattr(cfg, "auto_map", None) or {}
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    if trust_remote_code and amap and "AutoModel" not in amap \
            and "AutoModelForMaskedLM" in amap:
        # repos like NTv2 ship custom code only for task classes; bare AutoModel
        # would build the native architecture and mis-shape the checkpoint
        model = transformers.AutoModelForMaskedLM.from_pretrained(repo, **kw)
    elif arch.endswith("EncoderModel") and hasattr(transformers, arch):
        # encoder-only checkpoint of an encoder-decoder family (e.g. prot_t5 -enc)
        model = getattr(transformers, arch).from_pretrained(repo, **kw)
    else:
        model = AutoModel.from_pretrained(repo, **kw)
    if getattr(model.config, "is_encoder_decoder", False) \
            and hasattr(model, "get_encoder"):
        base = model.get_encoder()
    else:
        base = getattr(model, model.base_model_prefix, model)
    return tok, model, base


def encode_batch(tok, ss: list[str], dev: str, **tkw):
    """Tokenize to device tensors; returns (batch, spaced). Retries with
    space-separated residues when the vocab wants them (e.g. prot_t5)."""
    def enc(xs):
        b = tok(list(xs), return_tensors="pt", padding=True, **tkw)
        return {k: v.to(dev) for k, v in b.items()
                if k in ("input_ids", "attention_mask")}
    batch = enc(ss)
    if tok.unk_token_id is not None and \
            (batch["input_ids"] == tok.unk_token_id).float().mean() >= 0.3:
        return enc([" ".join(s) for s in ss]), True
    return batch, False


def pick_hidden(o):
    # prefer true hidden states over the LM logits that o[0] may hold
    h = getattr(o, "last_hidden_state", None)
    if h is None:
        h = getattr(o, "embedding", None)  # e.g. NTv3
    if h is None:
        hs = getattr(o, "hidden_states", None)
        h = hs[-1] if hs else o[0]
    return h


def forward_hidden(base, batch):
    try:
        out = base(**batch, output_hidden_states=True)
    except TypeError:
        out = base(**batch)
    return pick_hidden(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--modality", choices=list(ALPHA), required=True)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--lengths", type=int, nargs=2, default=(96, 160))
    ap.add_argument("--long", type=int, default=0,
                    help="also forward one sequence this long")
    args = ap.parse_args()

    import torch

    assert torch.cuda.is_available(), "CUDA unavailable"
    dev = "cuda:0"
    tok, model, base = load_backbone(args.repo, args.revision, args.cache,
                                     args.trust_remote_code)
    model = model.to(dev).train()

    batch, spaced = encode_batch(tok, seqs(args.modality, tuple(args.lengths)), dev)
    torch.cuda.reset_peak_memory_stats()

    hidden = forward_hidden(base, batch)
    assert hidden.dim() == 3 and hidden.shape[0] == 2, \
        f"unexpected shape {tuple(hidden.shape)}"
    assert torch.isfinite(hidden).all().item(), "non-finite hidden states"

    mask = batch.get("attention_mask")
    if mask is not None and mask.shape[1] != hidden.shape[1]:
        mask = None  # architecture changes sequence resolution (e.g. NTv3 U-Net)
    pooled = ((hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
              if mask is not None else hidden.mean(1))
    head = torch.nn.Linear(hidden.shape[-1], 1).to(dev)
    loss = torch.nn.functional.mse_loss(head(pooled).squeeze(-1),
                                        torch.tensor([0.25, 0.75], device=dev))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no backbone gradients"
    gmax = max(g.abs().max().item() for g in grads)
    assert all(torch.isfinite(g).all().item() for g in grads), "non-finite gradients"
    assert gmax > 0, "all-zero backbone gradients"

    facts = dict(
        repo=args.repo, revision=args.revision[:12], modality=args.modality,
        hidden_size=int(hidden.shape[-1]),
        n_params=sum(p.numel() for p in base.parameters()),
        tokens_for_lengths={str(n): int((batch["input_ids"][i] != tok.pad_token_id).sum())
                            if tok.pad_token_id is not None else int(batch["input_ids"].shape[1])
                            for i, n in enumerate(args.lengths)},
        tokenizer_model_max_length=int(min(tok.model_max_length, 10**9)),
        max_position_embeddings=getattr(model.config, "max_position_embeddings", None),
        loss=float(loss.item()), grad_abs_max=gmax, spaced_input=spaced,
    )

    if args.long:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            ls = seqs(args.modality, (args.long, args.long))[:1]
            lb, _ = encode_batch(tok, [" ".join(s) for s in ls] if spaced else ls,
                                 dev, truncation=False)
            lh = forward_hidden(base, lb)
            assert torch.isfinite(lh).all().item(), "non-finite long-context output"
            facts["long_context"] = dict(chars=args.long,
                                         tokens=int(lb["input_ids"].shape[1]))

    facts["peak_gpu_mem_mb"] = int(torch.cuda.max_memory_allocated() / 2**20)
    print("PASS " + json.dumps(facts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
