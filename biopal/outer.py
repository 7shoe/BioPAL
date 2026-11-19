"""BiOPAL's outer level around one training run.

OPAL owns the update rule and sees the plain training loss, the loss on the diagnostic set D and the block that
feeds the task head. This module shows OPAL those values before each of its decisions.

At a gate the outer level measures the checkpoint, asks the expert for one change to the training configuration
(which cohort's rows weigh more in the loss, or which block feeds the task head), and tests it: from one snapshot, the next K updates are run
with the change and without it, through the trainer's own update function, and both end states are scored per
row on the private set A. The change is adopted only if the mean loss difference is below minus one standard
error. Training then continues from the snapshot under the winning configuration, so the trial itself leaves no
trace in the model, in OPAL's state, or in what OPAL's agent is told.
"""
import copy
import hashlib
import json
import math
import os
import types

import numpy as np
import torch

from . import expert, ledger, measure, panel

SAMPLE = 2048   # fixed diagnostic and fit rows measured at every checkpoint
KEEP = ("keep", None)


class Readout(torch.nn.Module):
    """Which hidden state feeds the task head (-1: the backbone's default output) and a fixed rescaling of it.
    Buffers, so every checkpoint and every trial snapshot carries the route with the weights."""

    def __init__(self, width):
        super().__init__()
        self.register_buffer("layer", torch.tensor(-1))
        self.register_buffer("mu", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))

    def forward(self, pooled):
        return (pooled - self.mu) * self.scale


def spaced(total, n, every, lo=.15, hi=.85):
    """n update numbers spread evenly over [lo, hi] of the run, each moved up to the next check (a multiple of
    `every`) but not into the last `every` updates; fewer when the run is too short to hold n distinct ones."""
    at = np.linspace(lo * total, hi * total, n) if n > 1 else [total / 2]
    last = (total - every) // every * every
    return sorted({u for u in (min(math.ceil(a / every) * every, last) for a in at) if u >= every})


def place(gates, decisions, horizon, per_epoch, total, every):
    """Move each gate up to the first check whose trial window, the `horizon` updates after it, lies inside one
    epoch and holds no scheduled OPAL decision. A decision at the gate's own update is fine: OPAL decides first.
    -> (placed, dropped)"""
    placed, dropped = [], []
    for gate in gates:
        u = gate
        while u + horizon <= total:
            epoch_end = math.ceil(u / per_epoch) * per_epoch
            if u % per_epoch and u + horizon <= epoch_end and u not in placed \
                    and not any(u < d <= u + horizon for d in decisions):
                placed.append(u)
                break
            u += every
        else:
            dropped.append(gate)
    return placed, dropped


def verdict(candidate, incumbent):
    """Paired comparison on the rows of A: mean loss difference, its standard error, and the adoption rule."""
    d = candidate - incumbent
    mean, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
    return dict(mean_d=mean, se=se, passes=bool(np.isfinite(d).all() and mean < -se))


def is_scored(trial):
    """A trial counts as scored when it has a verdict whose mean difference and standard error are both finite
    (one private row, for example, leaves the standard error undefined)."""
    return trial is not None and math.isfinite(trial["mean_d"]) and math.isfinite(trial["se"])


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
        self.cohort, self.W = None, None                # the active configuration: which cohort weighs more
        self.layers = []                                # blocks that are probed and can feed the head
        if t.blocks and "readout" in t.net:
            self.layers = sorted({t.blocks // 2, 3 * t.blocks // 4, t.blocks})
        self.frozen, self.idle_ok = None, None          # parameter digests since a readout switch; all checks passed
        self.history, self.adopted, self.trial_calls, self.rewound_cost = [], [], 0, 0.
        self.audit = sorted(args.outer_audit) if args.outer_audit is not None else None
        self.horizon = max([args.outer_steps] + (self.audit or []))
        self.roles = [] if args.bio_policy != "expert" else ["informed"] + ["masked"] * (args.expert_shadow == "masked")
        self.expert = None
        if self.roles:
            self.expert = expert.Scripted(args.expert_script) if args.expert_script else expert.Claude(
                args.expert_model, args.expert_effort, args.expert_budget, args.expert_timeout, args.ledger,
                t.out / "expert")
        self.names = {role: panel.names(self.defs, t.task, t.modality, t.model, masked=role == "masked")
                      for role in self.roles}
        self.gates, dropped = [], []
        if self.roles or self.audit:
            self.gates, dropped = place(spaced(t.total, args.outer_decisions, args.check_every), self.decisions,
                                        self.horizon, t.per_epoch, t.total, args.check_every)
        self.log = (t.out / "outer.jsonl").open("w", buffering=1)
        self.write(event="schedule", opal_decisions=self.decisions, gates=self.gates, dropped_gates=dropped,
                   trial_updates=self.horizon)
        print(f"[biopal] OPAL decisions at updates {self.decisions}; gates after updates {self.gates}"
              + (f" (no room for a {self.horizon}-update trial at {dropped})" if dropped else ""), flush=True)
        if self.gates and t.device != "cpu" and getattr(args, "det", "strict") != "strict":
            print("[biopal] without --det strict a trial may not repeat bit for bit on the GPU (seen on ESM2, whose "
                  "attention kernel is then not deterministic); the committed updates can drift from the scored branch",
                  flush=True)

    def write(self, **record):
        self.log.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ the training configuration
    def config(self):
        return dict(cohort=self.cohort, readout_block=self.block())

    def set_cohort(self, cohort):
        """Rows of the cohort weigh (1 - alpha) + alpha / share, all others 1 - alpha: the mean weight over the
        fit set stays 1. None restores the ordinary mean."""
        self.cohort, self.W = cohort, None
        if cohort is not None:
            mask, alpha = self.s.cohorts[cohort], self.args.outer_alpha
            w = (1 - alpha) + alpha * mask / mask[self.t.roles["F"]].mean()
            self.W = torch.tensor(w, dtype=torch.float32, device=self.t.device)

    def install(self, layer, m):
        """Readout switch: the linear probe fitted on `layer` at this checkpoint becomes the task head. The probe's
        standardization is stored with the model, rescaled so that the head's input keeps its overall size (the
        optimizer's statistics for the head then stay of the right order). Blocks above `layer` stay in the model;
        they are no longer part of the forward pass, so they get no gradient and do not move."""
        p, head, r = m["probes"][layer], self.t.net["head"], self.t.net["readout"]

        def tensor(x):
            return torch.as_tensor(np.asarray(x), dtype=head.weight.dtype, device=head.weight.device)

        with torch.no_grad():
            was = tensor(m["head_input"])
            size = float((r(was) if int(r.layer) >= 0 else was).pow(2).mean().sqrt())
            head.weight.copy_(tensor(p["coef"] / size))
            head.bias.copy_(tensor(p["intercept"]))
            r.mu.copy_(tensor(p["mean"]))
            r.scale.copy_(tensor(size / p["sd"]))
            r.layer.fill_(layer)

    def apply(self, action, m):
        kind, target = action
        if kind in ("set_cohort", "ordinary_weighting"):
            self.set_cohort(target)
        elif kind in ("set_readout", "refit_head"):
            self.install(target, m)

    def digest(self):
        return {n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                for n, p in self.t.net.named_parameters() if p.requires_grad}

    def idle_check(self, where):
        """After a readout switch: a parameter that got no gradient at the last update must not have moved since
        the previous check, whatever program OPAL runs."""
        if self.frozen is None:
            return
        now = self.digest()
        idle = [n for n, p in self.t.net.named_parameters() if p.requires_grad and p.grad is None]
        moved = [n for n in idle if now[n] != self.frozen[n]]
        self.idle_ok = not moved and self.idle_ok is not False
        self.write(event="bypassed", at=where, readout_block=self.block(), idle_tensors=len(idle), moved=moved[:10],
                   unchanged=not moved)
        self.frozen = now

    def menu(self, m):
        """The changes on offer at this gate. A cohort can be proposed if it was measured (enough rows)."""
        measured = {r["cohort"] for r in m["table"]}
        out = {}
        if self.args.outer in ("cohort", "both"):
            out["set_cohort"] = [c for c in self.s.cohorts if c in measured and c != self.cohort]
            out["ordinary_weighting"] = self.cohort is not None
        if self.args.outer in ("readout", "both") and self.layers:
            out["set_readout"] = [k for k in self.layers if k != self.block()]
        return out

    # ------------------------------------------------------------------ snapshot, branch, gate
    def snapshot(self):
        o, worker = self.t.optimizer, getattr(self.t.optimizer, "worker", None)
        return dict(net=copy.deepcopy(self.t.net.state_dict()), opt=copy.deepcopy(o.state_dict()),
                    sched=copy.deepcopy(self.t.scheduler.state_dict()), rng=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    cohort=self.cohort, program=getattr(o, "digests", None),
                    calls=worker.k if worker else 0, cost=worker.cost if worker else 0.)

    def restore(self, snap):
        """Back to the snapshot. Returns whether OPAL's program changed since it; agent calls and cost that the
        rewind would otherwise forget are added to the run's counters first."""
        
        o, worker = self.t.optimizer, getattr(self.t.optimizer, "worker", None)
        
        if worker:
            self.trial_calls += worker.k - snap["calls"]
            self.rewound_cost += worker.cost - snap["cost"]
        changed = getattr(o, "digests", None) != snap["program"]
        self.t.net.load_state_dict(snap["net"])
        o.load_state_dict(copy.deepcopy(snap["opt"]))   # OPAL keeps what it is given: hand it a copy
        o._resume_note = ""                             # private in OPAL 0.3.10: its agent is not told of the rewind
        self.t.scheduler.load_state_dict(snap["sched"])
        
        torch.set_rng_state(snap["rng"])
        
        if snap["cuda"] is not None:
            torch.cuda.set_rng_state_all(snap["cuda"])
        self.set_cohort(snap["cohort"])
        
        return changed

    def branch(self, snap, action, m, batches, horizons):
        """From the snapshot: apply the action, run the updates, score A at each horizon."""
        
        self.restore(snap)
        self.apply(action, m)
        
        out = dict(losses=[], A={})
        
        if action[0] in ("set_readout", "refit_head"):   # the installed head must give the probe's loss on every row
            out["install_error"] = float(np.abs(self.t.rows(self.s.D)[0] - m["probe_loss"][action[1]]).max())
        
        for i, indices in enumerate(batches[:max(horizons)], 1):
            out["losses"].append(self.t.update(indices, self.weights(indices)).item())
            
            if i in horizons:
                out["A"][i] = self.t.rows(self.s.A)[0]
                if not np.isfinite(out["A"][i]).all():
                    raise RuntimeError("Non-finite loss on the private set")
        return out

    def propose(self, role, gate, update, m, menu):
        """One proposal; a failed call or an unusable answer keeps the incumbent and is not retried. Each expert
        gets the same checkpoint and its own earlier proposals, in the names it reads."""
        
        names, t = self.names[role], self.t
        
        text = expert.brief(dict(
            task=t.task, modality=t.modality, model=t.model, blocks=t.blocks or "an unknown number of",
            objective="classification" if t.classification else "regression", defs=self.defs, update=update,
            total=t.total, config=self.config(), table=m["table"], menu=menu, k=self.args.outer_steps,
            alpha=self.args.outer_alpha, history=[h for h in self.history if h["role"] == role]),
            names, masked=role == "masked")
        
        (t.out / "expert").mkdir(exist_ok=True)
        (t.out / "expert" / f"gate{gate + 1}_{role}_brief.txt").write_text(text)
        
        scripted = getattr(self.expert, "scripted", False)
        shown = {names[c]: c for c in menu.get("set_cohort", [])}
        shown.update({f"block_{k}": k for k in menu.get("set_readout", [])})
        if scripted:                                    # offline plans name real cohorts and integer blocks
            shown.update({target: target for target in list(shown.values())})

        try:
            answer = self.expert.propose(role, gate, text)
            action, target = expert.check(answer, menu, shown)
            # the record holds the answer's schema fields and the trainer's own; nothing else a reply carries
            said = {k: answer[k] for k in expert.SCHEMA["properties"] if k in answer}
            force = dict(force=True) if scripted and answer.get("force") is True else {}
            return dict(said, action=action, target=target, ok=True, **force)
        except Exception as e:
            return dict(action="keep", target=None, ok=False, error=f"{type(e).__name__}: {e}")

    def gate(self, update, batches):
        """Called after `update` when it is a gate; `batches` are the row indices of the updates that follow in
        this epoch. Leaves the trainer at the same update, under the configuration that won."""
        
        a, o, i = self.args, self.t.optimizer, self.gates.index(update)
        self.idle_check(f"gate {i}")
        
        m = measure.checkpoint(self.t.rows, self.s, self.layers, self.block())
        menu = self.menu(m)
        self.write(event="table", gate=i, update=update, config=self.config(), table=m["table"],
                   panel_loss=m["panel_loss"], fit_loss=m["fit_loss"], menu=menu)
        proposals = {role: self.propose(role, i, update, m, menu) for role in self.roles}
        self.write(event="proposals", gate=i, update=update, **proposals)
        self.log.flush()
        
        os.fsync(self.log.fileno())                     # every proposal is on disk before A is read
        by = {}                                         # each distinct change gets one branch
        
        for role, p in proposals.items():
            if p["action"] != "keep":
                by.setdefault((p["action"], p["target"]), []).append(role)
        if self.audit:
            options = [("set_cohort", c) for c in menu.get("set_cohort", [])]
            options += [("set_readout", k) for k in menu.get("set_readout", [])]
            options += [("refit_head", self.block())] * ("set_readout" in menu)   # the same block, head refitted
            for action in options + [("incumbent_repeat", None)]:
                by.setdefault(action, []).append("audit")
        horizons = sorted(set(self.audit or []) | {a.outer_steps})
        results, adopt, changed, force = {}, None, False, False

        if by:
            snap, worker = self.snapshot(), getattr(o, "worker", None)
            cap = getattr(o, "max_decisions", None)
            if worker:
                o.max_decisions = worker.k              # no agent call can be asked inside a branch
            try:
                scored = {}
                for action in by:
                    try:
                        scored[action] = self.branch(snap, action, m, batches, horizons)
                        if getattr(o, "digests", None) != snap["program"]:
                            raise RuntimeError("OPAL changed its program inside the branch")
                    except RuntimeError as e:           # a failed candidate falls back to the incumbent
                        scored[action] = dict(failed=str(e))
                try:
                    incumbent = self.branch(snap, KEEP, m, batches, horizons)   # last, so a failure ends the run
                except RuntimeError as e:
                    self.write(event="incumbent_failed", gate=i, update=update, error=str(e))
                    raise RuntimeError(f"BiOPAL gate {i}: the incumbent branch failed ({e}); run failed") from e
            finally:
                changed = self.restore(snap)            # also when the run fails: counters and ledger stay right
                if worker:
                    o.max_decisions = cap
            for action, roles in by.items():
                r = results[action] = dict(action=action[0], target=action[1], by=roles)
                if "failed" in scored[action]:
                    r["failed"] = scored[action]["failed"]
                else:
                    r.update({k: v for k, v in scored[action].items() if k == "install_error"})
                    r.update(losses=scored[action]["losses"],
                             horizons={h: verdict(scored[action]["A"][h], incumbent["A"][h]) for h in horizons})
            informed = proposals.get("informed", {})
            won = results.get((informed.get("action"), informed.get("target")), {})
            trial = won.get("horizons", {}).get(a.outer_steps)
            force = informed.get("force") is True       # propose() sets it, and only for a scripted expert
            if informed.get("action", "keep") != "keep" and not changed and is_scored(trial) and (
                    force or trial["passes"]):
                adopt = (informed["action"], informed["target"])
                self.apply(adopt, m)
                if adopt[0] == "set_readout":
                    self.frozen = self.digest()
                self.adopted.append(dict(gate=i, update=update, action=adopt[0], target=adopt[1]))
            self.write(event="trial", gate=i, update=update, trial_updates=a.outer_steps,
                       incumbent=dict(losses=incumbent["losses"],
                                      loss_A={h: float(incumbent["A"][h].mean()) for h in horizons}),
                       branches=[dict(r, horizons={str(h): v for h, v in r.get("horizons", {}).items()})
                                 for r in results.values()])
        
        for role, p in proposals.items():               # what each expert is told later about its own proposal
            r = results.get((p["action"], p["target"]), {})
            outcome = ("kept" if p["action"] == "keep" and p["ok"] else "failed" if not p["ok"] or "failed" in r
                       else "passed" if r["horizons"][a.outer_steps]["passes"] else "did not pass")
            p.update(outcome=outcome, trial=r.get("horizons", {}).get(a.outer_steps))
            if p["trial"] and p.get("predicted_sign"):   # did the expert call the direction of its own trial?
                d, se = p["trial"]["mean_d"], p["trial"]["se"]
                p["sign_correct"] = p["predicted_sign"] == ("lower" if d < -se else "higher" if d > se else "no_change")
            self.history.append(dict(gate=i, update=update, role=role, action=p["action"], target=p["target"],
                                     outcome=outcome, config_after=self.config()))
        self.write(event="decision", gate=i, update=update, adopted=adopt is not None,
                   action=adopt[0] if adopt else "keep", target=adopt[1] if adopt else None,
                   forced=bool(adopt and force), config_after=self.config(),
                   proposals={role: {k: p.get(k) for k in ("action", "target", "ok", "error", "outcome", "trial",
                                                           "predicted_sign", "sign_correct")}
                              for role, p in proposals.items()},
                   opal_program_changed=changed, opal_calls_in_trial=self.trial_calls,
                   loss_aware_updater=bool(getattr(o, "_step_ctx", False)), next_update=update + 1)
        print(f"[biopal] gate {i} after update {update}: " + (f"adopted {adopt[0]} {adopt[1]}" if adopt else "kept")
              + f"; configuration {self.config()}", flush=True)

    def block(self):
        """Index of the block that feeds the task head (the last one unless a readout switch was adopted)."""
        route = int(self.t.net["readout"].layer) if "readout" in self.t.net else -1
        
        return route if route >= 0 else self.t.blocks

    def weights(self, indices):
        """Per-row loss weights of the active configuration for one batch; None means the ordinary mean."""
        
        return None if self.W is None else self.W[torch.as_tensor(np.asarray(indices), device=self.W.device)]

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
        self.idle_check("end of training")
        cost = (float(worker.cost) if worker is not None else 0.) + self.rewound_cost
        if cost:
            ledger.charge(self.args.ledger, "opal", cost, run=str(self.t.out))
        self.log.close()
        
        return dict(opal_decisions=self.decisions, opal_cost_usd=cost, gates=self.gates, adopted=self.adopted,
                    final_config=self.config(), opal_calls_in_trials=self.trial_calls,
                    expert_calls=getattr(self.expert, "calls", 0), expert_cost_usd=getattr(self.expert, "cost", 0.),
                    bypassed_unchanged=self.idle_ok)
