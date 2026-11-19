"""The two checks every env's test files run per model; see tests/README.md.
Thin per-env test files parametrize these over their model_specs.MODELS."""
from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest

STEPS, LR = 3, 5e-5


@contextmanager
def _prepared(name, models, registry, cache_dirs, device):
    import torch
    if name not in registry:
        pytest.skip(f"{name} not in models.yaml (run audit_models.py)")
    spec, entry = models[name], registry[name]
    if spec.get("block_triton"):
        # DNABERT-2's bundled flash_attn_triton crashes on modern triton and
        # only its ImportError path is guarded; a None entry makes
        # `import triton` raise ImportError -> clean fallback attention
        sys.modules["triton"] = None
    try:
        from smoke_backbone import load_backbone
        tok, model, base = load_backbone(
            entry["repo_id"], entry["revision"],
            cache_dirs[entry["caches"][0]],
            trust_remote_code=spec.get("trust", False))
        model = model.to(device)
        if device == "cpu":
            model = model.float()  # half-precision checkpoints can't run on cpu
        yield spec, tok, model, base
        del tok, model, base
        if device != "cpu":
            torch.cuda.empty_cache()
    finally:
        if spec.get("block_triton"):
            sys.modules.pop("triton", None)


def forward_check(name, models, registry, cache_dirs, sample, device):
    """Offline load from the pinned snapshot + forward synthetic sequences
    to finite hidden states."""
    import torch
    from smoke_backbone import encode_batch, forward_hidden
    with _prepared(name, models, registry, cache_dirs, device) as \
            (spec, tok, model, base):
        model.eval()
        ss = sample[spec["modality"]]
        batch, _ = encode_batch(tok, ss, device)
        with torch.no_grad():
            hidden = forward_hidden(base, batch)
        assert hidden.dim() == 3, f"unexpected shape {tuple(hidden.shape)}"
        assert hidden.shape[0] == len(ss), "batch dimension lost"
        assert hidden.shape[-1] > 16, "looks like vocab logits, not hidden states"
        assert torch.isfinite(hidden).all(), "non-finite hidden states"


def training_check(name, models, registry, cache_dirs, task_batch, device):
    """3 AdamW steps on a balanced 4-sample curated-task batch through a
    linear probe, fine-tuning the whole backbone: asserts finite losses,
    non-zero backbone gradients, parameters moving, and decreasing loss."""
    import torch
    from smoke_backbone import encode_batch, forward_hidden
    with _prepared(name, models, registry, cache_dirs, device) as \
            (spec, tok, model, base):
        torch.manual_seed(17)
        model.train()
        ss, ys, _source = task_batch(spec["modality"])
        batch, _ = encode_batch(tok, ss, device)
        labels = torch.tensor(ys, device=device)

        probe, opt, losses = None, None, []
        watched = next(p for p in base.parameters() if p.requires_grad)
        before = watched.detach().clone()

        for step in range(STEPS):
            hidden = forward_hidden(base, batch)
            mask = batch.get("attention_mask")
            if mask is not None and mask.shape[1] != hidden.shape[1]:
                mask = None  # architecture changes sequence resolution (NTv3)
            pooled = ((hidden * mask.unsqueeze(-1)).sum(1)
                      / mask.sum(1, keepdim=True)
                      if mask is not None else hidden.mean(1))
            if probe is None:
                probe = torch.nn.Linear(hidden.shape[-1], 2).to(device)
                opt = torch.optim.AdamW([*model.parameters(),
                                         *probe.parameters()], lr=LR)
            loss = torch.nn.functional.cross_entropy(probe(pooled), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                grads = [p.grad for p in base.parameters() if p.grad is not None]
                assert grads, "no backbone gradients"
                assert all(torch.isfinite(g).all() for g in grads), \
                    "non-finite backbone gradients"
                assert max(g.abs().max().item() for g in grads) > 0, \
                    "all-zero backbone gradients"
            opt.step()
            losses.append(loss.item())

        assert all(l == l and abs(l) != float("inf") for l in losses), \
            f"non-finite losses {losses}"
        assert not torch.equal(before, watched.detach()), \
            "backbone parameters did not change after optimizer steps"
        assert losses[-1] < losses[0], \
            f"loss did not decrease while overfitting a fixed batch: {losses}"
