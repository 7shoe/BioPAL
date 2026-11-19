"""BiOPAL's biological expert proposes one training action per gate

At each checkpoint, the expert reviews the task, cohort definitions, diagnostic
gaps, current config and own proposal history. Then choses action.

Action space: {set_cohort, ordinary_weighting, set_readout, keep}.

The informed expert sees the original names and biological background. The
masked expert sees the same measurements with opaque IDs and no background.
Both see their own past trial outcomes. Neither sees raw sequences, labels,
private evaluation rows, trial scores, or OPAL's update programs.

Claude makes one `claude -p` call per proposal using the configured model,
effort, JSON schema, and remaining budget. Tools, session persistence, and
retries are disabled. Scripted reads proposals from a JSON file for tests
and offline runs.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from . import ledger, panel

SYSTEM = (
    "You are the outer level of a two-level fine-tuning procedure for a sequence model. An optimizer agent decides "
    "how the parameters are updated; you decide what is trained. At a checkpoint you read measurements of the "
    "model and propose one change to the training configuration, or keep it. Your proposal is tested: the next "
    "updates are trained twice from this checkpoint, with and without your change, and compared on private "
    "held-out rows. Propose a change only when the measurements give a specific reason to expect that it lowers "
    "the private loss. Answer with the JSON object only.")

ACTIONS = ["set_cohort", "ordinary_weighting", "set_readout", "keep"]

SIGNS = ["lower", "higher", "no_change"]

SCHEMA = dict(
    type="object", additionalProperties=False,
    required=["layer_evidence", "evidence_rows", "action", "target", "predicted_sign", "hypothesis"],
    properties=dict(layer_evidence=dict(type="string"),
                    evidence_rows=dict(type="array", items=dict(type="string"), maxItems=2),
                    action=dict(type="string", enum=ACTIONS), target=dict(type="string"),
                    predicted_sign=dict(type="string", enum=SIGNS), hypothesis=dict(type="string")))


def describe(config, name):
    weights = f"cohort {name(config['cohort'])} upweighted" if config["cohort"] else "ordinary (every row equal)"
    return f"loss weights {weights}; the task head reads block_{config['readout_block']}"


def brief(c, names, masked):
    """The text one expert reads. `c` is the gate's context (see Outer.context); `names` maps every real name to
    the one this expert sees. The two experts' briefs differ in those names and in the background line only."""
    name = lambda x: names.get(x, x)   # noqa: E731
    if masked:
        background = "The names of the task, the model, the features and the cohorts are replaced by opaque ids."
    else:
        features = dict.fromkeys(f for f, _ in c["defs"].values())
        background = " ".join([panel.ABOUT.get(c["task"], "No description of the task is on file.")]
                              + [f"{f}: {panel.ABOUT[f]}." for f in features])
    lines = [
        "TASK",
        f"{name(c['task'])} ({c['objective']}) on {name(c['modality'])} sequences; model {name(c['model'])} with "
        f"{c['blocks']} blocks.",
        f"Background: {background}",
        "",
        f"CHECKPOINT after update {c['update']} of {c['total']}",
        f"Configuration: {describe(c['config'], name)}.",
        "",
        "COHORTS (groups of rows)"]
    lines += [f"  {name(k)}: {text.format(f=name(f))}" for k, (f, text) in c["defs"].items()]
    lines += [
        "",
        "MEASUREMENTS (each row is a loss difference with its 95% interval, on n diagnostic rows of the cohort)",
        "  baseline: model loss minus the loss of a simple model on sequence composition. Positive: the simple "
        "model is better on this cohort. Negative: the model is better.",
        "  generalization: loss on held-out diagnostic rows minus loss on training rows. Positive: the model does "
        "better on rows it trained on.",
        "  layer block_k: loss of a linear probe on the block that feeds the head now, minus the loss of the same "
        "probe on block_k. Positive: block_k reads out better than the block in use. Negative: block_k is worse.",
        "id | measurement | cohort | n | gap [95% interval]"]
    for r in c["table"]:
        kind = r["measurement"] + (f" block_{r['layer']}" if r["layer"] is not None else "")
        lines.append(f"{r['id']} | {kind} | {name(r['cohort'])} | {r['n']} | {r['gap']:+.4f} "
                     f"[{r['lo']:+.4f}, {r['hi']:+.4f}]")
    lines += ["", "EARLIER GATES (your own proposals; \"passed\" means the change beat the unchanged configuration "
              "in its trial; the configuration in force afterwards is listed separately)"]
    for h in c["history"]:
        target = f" block_{h['target']}" if h["action"] == "set_readout" else f" {name(h['target'])}" if h["target"] else ""
        lines.append(f"  gate {h['gate'] + 1} after update {h['update']}: you proposed {h['action']}{target}; "
                     f"trial: {h['outcome']}; configuration afterwards: {describe(h['config_after'], name)}")
    lines += ["  none"] * (not c["history"])
    a, menu = c["alpha"], c["menu"]
    lines += ["", "ACTIONS: choose exactly one"]
    if menu.get("set_cohort"):
        lines.append(f"  set_cohort <cohort>: its rows weigh {1 - a:g} + {a:g} / (the cohort's share of the training "
                     f"rows) in the loss, all other rows {1 - a:g}. On offer: " + ", ".join(map(name, menu["set_cohort"])))
    if menu.get("ordinary_weighting"):
        lines.append("  ordinary_weighting: every row weighs the same again")
    if menu.get("set_readout"):
        lines.append("  set_readout <block>: the task head is refitted to that block and reads it from now on. "
                     "On offer: " + ", ".join(f"block_{k}" for k in menu["set_readout"]))
    lines += [
        "  keep: change nothing",
        f"The change is trained for {c['k']} updates from this checkpoint and compared with the unchanged "
        "configuration on private held-out rows. It is adopted only if the private loss falls by more than one "
        "standard error.",
        "",
        "Answer with: layer_evidence (the block whose rows you rely on, or \"none\"), evidence_rows (up to two row "
        "ids), action, target (the cohort or block; \"\" for keep and ordinary_weighting), predicted_sign (the "
        "effect of your action on the private loss: \"lower\", \"higher\" or \"no_change\"), hypothesis (one "
        "sentence)."]
    return "\n".join(lines)


def validate(answer):
    """The answer if it follows SCHEMA, the contract sent to the CLI: every field, no other field, the declared
    types and allowed values. ValueError naming what is wrong otherwise."""
    if not isinstance(answer, dict):
        raise ValueError(f"answer does not follow the schema: not an object: {str(answer)[:200]}")
    fields = SCHEMA["properties"]
    wrong = [f"missing field {k!r}" for k in SCHEMA["required"] if k not in answer]
    wrong += [f"unexpected field {k!r}" for k in answer if k not in fields]
    for k, spec in fields.items():
        if k not in answer:
            continue
        v = answer[k]
        if spec["type"] == "string":
            if not isinstance(v, str):
                wrong.append(f"{k!r} is not a string")
            elif "enum" in spec and v not in spec["enum"]:
                wrong.append(f"{k!r} is {v[:40]!r}, not one of {spec['enum']}")
        elif spec["type"] == "array" and spec["items"] == dict(type="string"):
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                wrong.append(f"{k!r} is not a list of strings")
            elif len(v) > spec.get("maxItems", len(v)):
                wrong.append(f"{k!r} has more than {spec['maxItems']} items")
        else:
            raise TypeError(f"validate() cannot check field {k!r} of SCHEMA")
    if wrong:
        raise ValueError("answer does not follow the schema: " + "; ".join(wrong))
    return answer


def extract(reply):
    """The expert's answer from the CLI's JSON reply. `structured_output` is used whenever the reply has that key;
    only a reply without it is read from the result text, which must then be plain JSON. Both must follow SCHEMA."""
    if "structured_output" in reply:
        return validate(reply["structured_output"])
    try:
        answer = json.loads(reply.get("result"))
    except (TypeError, ValueError):
        raise ValueError("answer does not follow the schema: no structured_output and the result text is not "
                         f"JSON: {str(reply.get('result'))[:200]}")
    return validate(answer)


def check(answer, menu, shown=None):
    """(action, target) if the answer is well formed and on offer at this gate; ValueError otherwise. `shown` maps
    every name this expert may answer with (masked cohort ids, block_k) to the real target; a target is accepted
    only under one of those names."""
    if not isinstance(answer, dict) or answer.get("action") not in ACTIONS \
            or answer.get("predicted_sign", "no_change") not in SIGNS:
        raise ValueError(f"answer does not follow the schema: {str(answer)[:200]}")
    action, target = answer["action"], answer.get("target")
    target = (shown or {}).get(target) if type(target) in (str, int) else None
    if action == "keep" or (action == "ordinary_weighting" and menu.get("ordinary_weighting")):
        return action, None
    if action in ("set_cohort", "set_readout") and target in menu.get(action, []):
        return action, target
    raise ValueError(f"unavailable action: {action} {answer.get('target')!r}")


class Scripted:
    """Answers read from a JSON file {"informed": [...], "masked": [...]}: entry i answers gate i, a missing entry
    keeps. An entry needs only action and target; targets are real names (cohorts) and integers (blocks). An entry
    may carry "force": true to adopt its change without the trial margin, once its trial has been scored (tests
    only; no other expert can force)."""
    cost = calls = 0
    scripted = True

    def __init__(self, path):
        self.plan = json.loads(Path(path).read_text())

    def propose(self, role, gate, text):
        plan = self.plan.get(role, [])
        return dict(plan[gate]) if gate < len(plan) else dict(action="keep", target="")


class Claude:
    """Live expert. `budget` is this run's cap, or with a ledger the ceiling on what all runs sharing it have
    spent on the expert. Once the budget cannot cover another call, or most of a call's cost was billed to another
    model than the one asked for, the expert is off for the rest of the run and every later gate keeps the
    incumbent. (A small helper call of the CLI on another model is tolerated.) A reply that does not follow SCHEMA
    is an error like any other: the gate keeps the incumbent and the call is not repeated."""
    scripted = False

    def __init__(self, model, effort, budget, timeout, ledger_path, folder):
        self.model, self.effort, self.budget, self.timeout = model, effort, budget, timeout
        self.ledger, self.folder, self.cost, self.calls, self.off = ledger_path, Path(folder), 0., 0, None
        self.folder.mkdir(parents=True, exist_ok=True)

    def left(self):
        return self.budget - (ledger.spent(self.ledger, "expert") if self.ledger else self.cost)

    def propose(self, role, gate, text):
        left = self.left()
        if self.off is None and left < max(.01, self.cost / max(self.calls, 1)):
            self.off = f"expert budget used up (${left:.4f} of ${self.budget} left)"
        if self.off:
            raise RuntimeError(self.off)
        cmd = [os.environ.get("BIOPAL_CLAUDE", "claude"), "-p", text, "--model", self.model, "--effort", self.effort,
               "--system-prompt", SYSTEM, "--tools", "", "--json-schema", json.dumps(SCHEMA),
               "--output-format", "json", "--max-budget-usd", f"{left:.4f}", "--no-session-persistence",
               "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
        self.calls += 1
        with tempfile.TemporaryDirectory() as neutral:   # no project files or memory in reach
            done = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=neutral,
                                  stdin=subprocess.DEVNULL)
        (self.folder / f"gate{gate + 1}_{role}_reply.json").write_text(done.stdout or done.stderr)
        try:
            reply = json.loads(done.stdout)
        except ValueError:
            raise RuntimeError(f"no JSON from the CLI (exit {done.returncode}): {(done.stderr or done.stdout)[:200]}")
        cost = float(reply.get("total_cost_usd") or 0.)
        self.cost += cost
        ledger.charge(self.ledger, "expert", cost, run=str(self.folder.parent), gate=gate + 1, expert=role)
        usage = reply.get("modelUsage") or {}
        paid = {asked: sum(float(v.get("costUSD") or 0.) for m, v in usage.items() if m.startswith(self.model) == asked)
                for asked in (True, False)}
        other = sorted(m for m in usage if not m.startswith(self.model))
        if other and paid[False] >= paid[True]:
            self.off = f"billed to {other}, not {self.model}: the expert is off for the rest of the run"
            raise RuntimeError(self.off)
        if reply.get("is_error"):
            raise RuntimeError(f"the CLI reported an error: {str(reply.get('result'))[:200]}")
        return extract(reply)
