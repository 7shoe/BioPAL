"""A toy trainer with the interface finetune.py gives biopal.outer.Outer: a small model that returns every layer's
output, a synthetic DNA task, and the same update / rows / training-loop shape. Runs in seconds on CPU."""
import argparse
import math
import types

import numpy as np
import opal
import pandas as pd
import torch
import torch.nn.functional as F

from biopal import outer, panel

FLAGS = dict(seed=0, bio_policy="off", check_every=10, ledger=None, outer="both", outer_decisions=2, outer_steps=8,
             outer_alpha=.2, outer_audit=None, expert_shadow="none", expert_script=None, expert_model="claude-haiku-5-5",
             expert_effort="low", expert_budget=None, expert_timeout=5., expert_backend="claude",
             expert_url="http://127.0.0.1:8000/v1", expert_revision=None, expert_note=None, expert_request=None)


class Backbone(torch.nn.Module):
    def __init__(self, width=16, blocks=4):
        super().__init__()
        self.emb = torch.nn.Embedding(4, width)
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.Tanh()) for _ in range(blocks))
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, ids):   # embeddings, then each block's output; the last one passes the final norm
        hs = [self.emb(ids)]
        for block in self.blocks:
            hs.append(hs[-1] + block(hs[-1]))
        return hs[:-1] + [self.norm(hs[-1])]


def data(n=1200, classification=True, seed=0):
    """Sequences whose label is carried by the number of G; GC-rich rows are noisier."""
    rng = np.random.default_rng(seed)
    seqs = ["".join(rng.choice(list("ACGT"), 24, p=[.45 - g / 2, g / 2, g / 2 + .1, .45 - g / 2]))
            for g in rng.uniform(.05, .35, n)]
    g = np.array([s.count("G") for s in seqs], float)
    signal = g + rng.normal(0, 1 + 2 * (g > np.median(g)), n)
    y = (signal > np.median(signal)).astype(int) if classification else (signal - signal.mean()) / signal.std()
    return pd.DataFrame(dict(sequence=seqs, y=y))


def make(tmp_path, backend="keep", classification=True, epochs=3, batch_size=16, decisions=2, n=1200,
         wrapped=True, **flags):
    """-> (args, t, bio). bio is None when `wrapped` is False (the plain arm)."""
    args = argparse.Namespace(**{**FLAGS, **flags})
    torch.manual_seed(args.seed)
    frame = data(n, classification, args.seed)
    roles = panel.split(frame, "y", classification, .3, args.seed)
    ids = torch.tensor([["ACGT".index(c) for c in s] for s in frame.sequence])
    y = torch.tensor(frame.y.to_numpy(), dtype=torch.long if classification else torch.float32)
    backbone, num_out = Backbone(), 2 if classification else 1
    net = torch.nn.ModuleDict(dict(backbone=backbone, head=torch.nn.Linear(16, num_out), readout=outer.Readout(16)))
    per_epoch = math.ceil(len(roles["F"]) / batch_size)
    total = epochs * per_epoch
    base = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(base, lambda s: 1 - s / (total + 1))
    optimizer = base
    if wrapped:
        at = outer.spaced(total, decisions, args.check_every)
        optimizer = opal.wrap(base, net, scheduler=scheduler, total_steps=total, goal="toy", backend=backend,
                              schedule=at, max_decisions=len(at), every=args.check_every, blocking=True,
                              summarize=False, run_dir=str(tmp_path / "opal"))

    def each(out, target):
        return (F.cross_entropy(out, target, reduction="none") if classification
                else F.mse_loss(out.squeeze(-1), target, reduction="none"))

    def predict(x, layers=None):
        hs, route = backbone(x), int(net["readout"].layer)
        out = net["head"](net["readout"](hs[route].mean(1)) if route >= 0 else hs[-1].mean(1))
        return (out, {k: hs[k].mean(1) for k in layers}) if layers else out

    def update(indices, weights=None):
        optimizer.zero_grad(set_to_none=True)
        rows_ = each(predict(ids[indices]), y[indices])
        loss = rows_.mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")
        (loss if weights is None else (rows_ * weights).mean()).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        if wrapped:
            optimizer.step(loss=loss)
        else:
            optimizer.step()
        scheduler.step()
        return loss

    @torch.no_grad()
    def rows(indices, layers=None):
        modes = [(m, m.training) for m in net.modules()]
        net.eval()
        out = predict(ids[indices], layers)
        out, hidden = out if layers else (out, {})
        for m, was in modes:
            m.training = was
        return each(out, y[indices]).double().numpy(), {k: v.double().numpy() for k, v in hidden.items()}

    t = types.SimpleNamespace(
        net=net, optimizer=optimizer, scheduler=scheduler, update=update, rows=rows, frame=frame, label="y",
        roles=roles, classification=classification, y=frame.y.to_numpy(), num_out=num_out, modality="dna",
        task="toy_task", model="toy_model", device="cpu", per_epoch=per_epoch, total=total, blocks=4,
        out=tmp_path, epochs=epochs, batch_size=batch_size, wrapped=wrapped)
    return args, t, (outer.Outer(args, t) if wrapped else None)


def train(args, t, bio, switch=None):
    """The training loop of finetune.py, without validation. -> the loss of every committed update.
    switch: {update: cohort} sets the cohort after that update without a gate (the reference for an adoption)."""
    order_rng, losses, step = np.random.default_rng(args.seed), [], 0
    t.net.train()
    for _ in range(t.epochs):
        order = t.roles["F"][order_rng.permutation(len(t.roles["F"]))]
        for start in range(0, len(order), t.batch_size):
            indices = order[start:start + t.batch_size]
            if bio:
                bio.before(step + 1)
            losses.append(t.update(indices, bio.weights(indices) if bio else None).item())
            step += 1
            if switch and step in switch:
                bio.set_cohort(switch[step])
            if bio and step in bio.gates:
                bio.gate(step, [order[s:s + t.batch_size] for s in range(start + t.batch_size, len(order), t.batch_size)])
    if t.wrapped:
        t.optimizer.close()
    return losses, (bio.close() if bio else None)
