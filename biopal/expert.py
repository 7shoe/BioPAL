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
and offline runs. Local sends one request per proposal to an open-weights
model served on this machine and may add the task owner's private note to
the informed brief; nothing it reads is ever sent to another backend.
"""

import hashlib
import ipaddress
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
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


NOTE_OPEN = "TASK NOTE (from the owner of this task; not part of the measurements)"
NOTE_CLOSE = "END OF TASK NOTE"
WITHHELD = "[withheld: the brief carried a private note; the reply is in the run's expert folder]"


def load_note(path, task=None):
    """A task owner's note for the informed local expert, from a text file: header lines, a blank line, the note.

        task: nt_tasks_official_splice_sites_donors
        label: simulated institutional knowledge
        visibility: private

        Positives that ... (free text)

    `task` binds the note: with a task given, a note written for another one is refused. `visibility: public`
    allows the text and the expert's free-text answers in logs and summaries; a private note (the default) leaves
    this process only inside the request to the local model, and logs carry its hash."""
    head, _, text = Path(path).read_text().partition("\n\n")
    fields, text = {}, text.strip()
    for line in head.splitlines():
        key, colon, value = line.partition(":")
        if not colon or key.strip() not in ("task", "label", "visibility") or key.strip() in fields:
            raise ValueError(f"{path}: a note starts with the header lines task, label and visibility (each "
                             f"'key: value', once), then a blank line; cannot read {line[:60]!r}")
        fields[key.strip()] = value.strip()
    if not fields.get("task") or not text or fields.get("visibility", "private") not in ("private", "public"):
        raise ValueError(f"{path}: a note needs a 'task: <name>' header, a visibility of private or public if one "
                         "is given, and text after a blank line")
    if task is not None and fields["task"] != task:
        raise ValueError(f"{path} is a note for task {fields['task']!r}, not for {task!r}")
    return dict(task=fields["task"], label=fields.get("label", ""), public=fields.get("visibility") == "public",
                text=text, sha256=hashlib.sha256(text.encode()).hexdigest(), chars=len(text))


def note_record(note):
    """What logs and summaries say about a note: its text only if it is public."""
    return None if note is None else {k: v for k, v in note.items() if k != "text" or note["public"]}


def with_note(text, note):
    """The brief with the note placed after the background line. Everything else stays as it is, byte for byte:
    measurements, menu and their order are the same with and without the note."""
    lines = text.split("\n")
    at = next(i for i, line in enumerate(lines) if line.startswith("Background: ")) + 1
    return "\n".join(lines[:at] + ["", NOTE_OPEN] + note["text"].splitlines() + [NOTE_CLOSE] + lines[at:])


def shown(names, menu, real=False):
    """{name an expert may answer with: real target} for one gate's menu: the cohort names this expert reads and
    block_k. `real` also accepts the real names themselves (scripted plans name real cohorts and integer blocks)."""
    out = {names.get(c, c): c for c in menu.get("set_cohort", [])}
    out.update({f"block_{k}": k for k in menu.get("set_readout", [])})
    if real:
        out.update({target: target for target in list(out.values())})
    return out


def failure(error):
    """Why a proposal failed, as one word. Unlike the error text it cannot quote the reply, so it is what a log
    keeps when the brief carried a private note."""
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) or "timed out" in str(error):
        return "timeout"
    if isinstance(error, ValueError):
        return "off_menu" if str(error).startswith("unavailable action") else "malformed"
    return "transport" if isinstance(error, OSError) else "backend"


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


LIMIT = 1 << 20   # bytes read from one reply of a local server


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "the local expert's server answered with a redirect; "
                                     "redirects are not followed", headers, fp)


def loopback(url):
    """The address if it is plain http on a loopback IP literal (127.0.0.0/8 or ::1), else ValueError. A host name,
    also `localhost`, is refused: what it resolves to is not this module's to decide."""
    u = urllib.parse.urlsplit(url)
    try:
        ok = u.scheme == "http" and ipaddress.ip_address(u.hostname or "").is_loopback and u.port is not None
    except ValueError:
        ok = False
    if not ok or u.username or u.password or u.query or u.fragment:
        raise ValueError(f"the local expert is reached at http://<loopback IP>:<port>/..., for example "
                         f"http://127.0.0.1:8000/v1 (no host names, no https, no credentials): {url!r}")
    return url.rstrip("/")


def listeners(port):
    """User ids of the processes that listen on this TCP port (Linux; None where /proc does not tell)."""
    found = set()
    try:
        for table in ("/proc/net/tcp", "/proc/net/tcp6"):
            for row in Path(table).read_text().splitlines()[1:]:
                f = row.split()
                if f[3] == "0A" and int(f[1].rsplit(":", 1)[1], 16) == port:   # 0A: listening
                    found.add(int(f[7]))
    except (OSError, IndexError, ValueError):
        return None
    return found


class Local:
    """An open-weights model behind an OpenAI-compatible chat-completions endpoint on this machine (vLLM and
    others). One request per proposal: a timeout, an HTTP error or a reply that does not follow SCHEMA is an error
    like any other, the gate keeps the incumbent, and nothing is asked again or asked elsewhere. There is no dollar
    budget and no ledger entry.

    The brief can hold private measurements and a private note, so the client stays on the host: the address must
    be a loopback IP literal, proxies from the environment are not used, a redirect is an error, and on a shared
    machine the server's port must belong to this user (or root). `request` adds or replaces fields of the request
    body, e.g. {"chat_template_kwargs": {"enable_thinking": false}} for Qwen3 on vLLM; with the defaults below it
    is recorded as the decoding settings."""
    scripted = False
    cost = 0.

    def __init__(self, url, model, revision, timeout, folder, seed=0, request=None):
        self.url, self.model, self.revision, self.timeout = loopback(url), model, revision, timeout
        self.folder, self.calls, self.latency = Path(folder), 0, []
        if {"model", "messages"} & set(request or {}):
            raise ValueError("the extra request fields cannot replace model or messages")
        self.decoding = {**dict(temperature=0, top_p=1, seed=seed, max_tokens=512), **(request or {})}
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.served = self.describe()

    def fetch(self, url, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
        with self.opener.open(request, timeout=self.timeout) as reply:
            return reply.read(LIMIT)

    def describe(self):
        """What is being served, asked once before training starts: a wrong address, another user's server or a
        different model fails here and not at the first gate."""
        port, mine = urllib.parse.urlsplit(self.url).port, {os.getuid(), 0}
        owners = listeners(port)
        if owners is not None and not owners <= mine:
            raise RuntimeError(f"port {port} is served by another user of this machine (uid "
                               f"{sorted(owners - mine)}); the brief is not sent there")
        try:
            cards = json.loads(self.fetch(self.url + "/models"))["data"]
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"no model list from {self.url}/models ({type(e).__name__}: {e}); is the server up?")
        card = next((c for c in cards if c.get("id") == self.model), None)
        if card is None:
            raise RuntimeError(f"{self.url} does not serve {self.model!r}; it serves {[c.get('id') for c in cards]}")
        root = str(card.get("root") or "")
        pinned = re.findall(r"\b[0-9a-f]{40}\b", root)   # a snapshot folder of the Hugging Face cache names its commit
        if pinned and self.revision not in pinned:
            raise RuntimeError(f"the server loaded {root}, not revision {self.revision}")
        try:   # vLLM reports its version next to the API root; other servers may not
            version = json.loads(self.fetch(f"http://{urllib.parse.urlsplit(self.url).netloc}/version")).get("version")
        except (OSError, ValueError, AttributeError):
            version = None
        return dict(url=self.url, model=self.model, revision=self.revision, revision_in_served_path=bool(pinned),
                    served_root=root or None, max_model_len=card.get("max_model_len"), owned_by=card.get("owned_by"),
                    serving_version=version, port_owner_checked=owners is not None)

    def propose(self, role, gate, text):
        body = dict(model=self.model, messages=[dict(role="system", content=SYSTEM), dict(role="user", content=text)],
                    response_format=dict(type="json_schema",
                                         json_schema=dict(name="proposal", schema=SCHEMA, strict=True)))
        body.update(self.decoding)
        saved = self.folder / f"gate{gate + 1}_{role}_reply.json"
        self.calls += 1
        tick = time.perf_counter()
        try:
            raw = self.fetch(self.url + "/chat/completions", body)
        except urllib.error.HTTPError as e:
            saved.write_bytes(e.read(LIMIT))
            raise
        finally:
            self.latency.append(time.perf_counter() - tick)
        saved.write_bytes(raw)
        try:
            answer = json.loads(json.loads(raw)["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError("answer does not follow the schema: the reply's message content is not plain JSON: "
                             + raw[:200].decode("utf-8", "replace"))
        return validate(answer)

    def record(self):
        """Model, revision, serving version, decoding settings, calls and latency, for the run's summary."""
        t = self.latency
        return dict(self.served, backend="local", decoding=self.decoding, calls=self.calls,
                    latency_seconds=dict(mean=sum(t) / len(t), max=max(t), each=[round(x, 3) for x in t]) if t else None)
