"""BiOPAL's outer level around one training run.

OPAL owns the update rule and sees the plain training loss, the loss on the diagnostic set D and the block that
feeds the task head. This module shows OPAL those values before each of its decisions.
"""
import json
import math
import types

import numpy as np
import torch

from . import ledger, measure, panel

SAMPLE = 2048   # fixed diagnostic and fit rows measured at every checkpoint


def spaced(total, n, every, lo=.15, hi=.85):
    """n update numbers spread evenly over [lo, hi] of the run, each moved up to the next check (a multiple of
    `every`) but not into the last `every` updates; fewer when the run is too short to hold n distinct ones."""
    at = np.linspace(lo * total, hi * total, n) if n > 1 else [total / 2]
    last = (total - every) // every * every
    return sorted({u for u in (min(math.ceil(a / every) * every, last) for a in at) if u >= every})


@torch.no_grad()
def blocks(base, x, default):
    """Number of blocks if the backbone returns every layer's output in the shape of its default output and the
    last one equals it; otherwise None (then there are no layer probes)."""
    modes = [(m, m.training) for m in base.modules()]
    base.eval()
    try:
        hs = base(**x, output_hidden_states=True).hidden_states
        ok = all(h.shape == hs[-1].shape for h in hs) and torch.allclose(hs[-1], default(base, x), atol=1e-5)
    except Exception:
        ok = False
    for m, was in modes:
        m.training = was
    return len(hs) - 1 if ok else None


class Outer:
    """The outer level of one run. `t` holds the trainer's objects and functions (finetune.py builds it):
    net, optimizer, scheduler, update(indices, weights), rows(indices, layers), frame (the train rows), label,
    y (labels, or standardized targets), roles (F, D, A), classification, num_out, modality, task, model, device,
    per_epoch, total, blocks, out."""

    def __init__(self, args, t):
        self.args, self.t = args, t
        comp = panel.composition(t.frame["sequence"].tolist(), t.modality)
        masks, self.defs = panel.cohorts(t.frame, t.label, t.classification, t.modality, comp, t.roles["F"])
        rng = np.random.default_rng([args.seed, SAMPLE])

        def pick(rows):
            return np.sort(rng.choice(rows, min(SAMPLE, len(rows)), replace=False))

        self.s = types.SimpleNamespace(D=pick(t.roles["D"]), F=pick(t.roles["F"]), A=t.roles["A"], y=t.y,
                                       cohorts=masks, classification=t.classification, num_out=t.num_out)
        simple = measure.fit_linear(comp[t.roles["F"]], t.y[t.roles["F"]], t.classification, t.num_out)
        self.s.base = measure.linear_loss(simple, comp[self.s.D], t.y[self.s.D], t.classification)
        self.decisions = [int(u) for u in getattr(t.optimizer, "schedule", [])]   # as OPAL resolved them
        self.gates = []
        self.log = (t.out / "outer.jsonl").open("w", buffering=1)
        self.write(event="schedule", opal_decisions=self.decisions, gates=self.gates)
        print(f"[biopal] OPAL decisions at updates {self.decisions}", flush=True)

    def write(self, **record):
        self.log.write(json.dumps(record) + "\n")

    def block(self):
        """Index of the block that feeds the task head (the last one unless a readout switch was adopted)."""
        return self.t.blocks

    def weights(self, indices):
        """Per-row loss weights of the active configuration for one batch; None means the ordinary mean."""
        return None

    def before(self, update):
        """Called before every update. When an OPAL decision is due at this update, show OPAL the loss on D and
        the readout block (and, with --bio-policy fixed, the two whole-panel gaps)."""
        if update not in self.decisions:
            return
        loss_d = self.t.rows(self.s.D)[0]
        shown = dict(panel_loss=float(loss_d.mean()))
        if self.t.blocks:
            shown["readout_block"] = self.block()
        if self.args.bio_policy == "fixed":
            shown.update(baseline_gap=float((loss_d - self.s.base).mean()),
                         generalization_gap=float(loss_d.mean() - self.t.rows(self.s.F)[0].mean()))
        self.t.optimizer.log(shown, show_agent=True)
        self.write(event="shown_to_opal", update=update, **shown)

    def close(self):
        """After OPAL closed: charge its spend to the ledger; returns the summary for result.json."""
        worker = getattr(self.t.optimizer, "worker", None)
        cost = float(worker.cost) if worker is not None else 0.
        if cost:
            ledger.charge(self.args.ledger, "opal", cost, run=str(self.t.out))
        self.log.close()
        return dict(opal_decisions=self.decisions, opal_cost_usd=cost)
