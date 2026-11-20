#!/usr/bin/env python3
"""Runner of the BiOPAL campaign. The protocol is README.md in this folder; section numbers below refer to it.

  campaign.py manifest                      write runs.tsv, the list of all runs
  campaign.py check [--no-live]             readiness checks of stage 0 that train nothing
  campaign.py run --stage 2 [--deadline H]  start the runs of a stage that are not done, wait for them
  campaign.py run --stage 2 --dry-run       print every command of the stage, start nothing
  campaign.py run --catchup --deadline H    runs of stages 1 to 5 that are still missing, most important first
  campaign.py status                        rewrite and print status/status.tsv
  campaign.py verdict                       after stage 0: which kinds of run and which pairs work
  campaign.py fresh-ready                   put the outputs of an earlier stage 0 aside
  campaign.py cleanup --yes                 delete the kept checkpoints (after the user signed off)

Several runners may work at once (one per stage script): they share the GPUs, the budget and the status files
through status/lock. A run is done when its result file exists, so every command can be repeated.
Standard library only.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FT = REPO / "finetune"
OUT = FT / "runs" / "campaign"          # all outputs; not in git
LOGS = OUT / "_logs"
STATUS = HERE / "status"
ENVS = Path(os.environ.get("CONDA_ENVS")
            or Path(os.environ.get("CONDA_EXE", "~/miniconda3/bin/conda")).expanduser().parents[1] / "envs")

PAIRS = {  # short name: task, model, conda env, objective, minutes of one plain fit, free GPU memory needed (GB)
    "ctcf": ("fish_gue_ctcf", "dnabert2_117m", "biopal-v2", "classification", 11, 12),
    "h3k27ac": ("nt_tasks_official_h3k27ac", "ntv2_50m", "biopal-v2", "classification", 8, 12),
    "splice": ("rna_dst_splice_site_acceptor", "splicebert", "bio-rna", "classification", 8, 12),
    "mrl": ("rna_dst_mean_ribosome_load", "rnafm", "bio-rna", "regression", 7, 12),
    "melt": ("cds_melting_point", "esm2_t12_35m", "biopal-v2", "regression", 14, 30),
    "gfp": ("proteingym_gfp_fluorescence", "esm2_t12_35m", "biopal-v2", "regression", 20, 12),
}
FACTOR = dict(plain=1, matched=1, opal=1.15, biopal=1.75, biopal_qwen=1.75, hpo_study=6, hpo=1, audit=13)
SETTINGS = {  # name: OPAL's model and effort, the expert's model and effort, cap of OPAL's agent and of the expert (USD)
    "default": ("claude-opus-5-5", "medium", "claude-opus-5-5", "high", 2.00, 1.00),
    "sonnet": ("claude-sonnet-5-5", "high", "claude-sonnet-5-5", "high", 2.00, 1.00),
    "opus_high": ("claude-opus-5-5", "high", "claude-opus-5-5", "high", 4.00, 1.00),
    "tiny": ("claude-haiku-5-5", "low", "claude-haiku-5-5", "low", 0.10, 0.10),   # readiness runs only
}
CAPS = dict(pilot=20., seed42=30., seeds43_44=60., replays=5., agent_ablation=20.)   # section 4
TOTAL_CAP = 135.                        # of 150: the rest covers what the caps of single runs cannot stop exactly
DISK_FLOOR_GB, DISK_STOP_GB = 100, 40   # section 6
MAX_GPU_JOBS, LANES = 8, 4              # section 5
MIN_LIMIT_MIN = 45                      # a run is stopped after 3 times its expected minutes, never before these
QWEN, QWEN_REV = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
QWEN_URL = f"http://127.0.0.1:{os.environ.get('QWEN_PORT', '8000')}/v1"
QWEN_REQUEST = '{"chat_template_kwargs": {"enable_thinking": false}}'
PANEL = ["--panel-frac", "0.15", "--panel-until", "0.6"]
FIELDS = ["id", "stage", "pair", "name", "seed", "setting", "kind", "env", "need_gb", "minutes", "paid", "runs",
          "needs", "soft"]
TINY_GROUPS = ("ready", "pilot_")       # tier 1 of section 6, with the kinds hpo_study and audit


# ---------------------------------------------------------------------------------------------- the list of runs
def manifest():
    rows = []

    def add(stage, group, pair, name, seed=42, setting="default", needs=(), soft=()):
        kind = {"hpo_study": "hpo", "audit": "audit", "probe": "train"}.get(name, "replay" if name.startswith("replay")
                                                                              else "train")
        tiny = setting == "tiny"
        if pair in PAIRS:
            _, _, env, _, minutes, need = PAIRS[pair]
        else:                           # a replay over several pairs: no GPU of its own
            env, minutes, need = "biopal-v2", 15, 0
        if kind == "replay":
            minutes, need = 15, 0
        elif tiny or name == "probe":
            minutes = 5
        else:
            minutes = round(minutes * FACTOR[name])
        paid = {"opal": "agent", "opal_cap": "agent", "biopal": "agent+expert", "biopal_qwen": "agent"}.get(name, "")
        if kind == "replay" and name != "replay_local":
            paid = "hosted"
        rows.append(dict(id=f"{group}/{pair}/{name}_s{seed}", stage=stage, pair=pair, name=name, seed=seed,
                         setting=setting, kind=kind, env=env, need_gb=need, minutes=minutes, paid=paid,
                         runs=6 if name == "hpo_study" and not tiny else 1, needs=",".join(needs), soft=",".join(soft)))

    # stage 0: every pair loads and trains for a few updates; one tiny run of every kind on the cheapest pair
    for pair in PAIRS:
        add("0", "ready", pair, "probe", setting="tiny")
    for name in ("plain", "matched", "opal", "biopal", "biopal_qwen", "opal_cap", "hpo_study", "audit"):
        add("0", "ready", "mrl", name, setting="tiny")
    add("0", "ready", "mrl", "hpo", setting="tiny", needs=["ready/mrl/hpo_study_s42"])
    add("0", "ready", "mrl", "replay", setting="tiny", needs=["ready/mrl/audit_s42"])
    # stage 1: seed 42, what costs no Claude call
    for pair in PAIRS:
        for name in ("hpo_study", "plain", "matched"):
            add("1", "main", pair, name)
    # stage 1b: pilot
    for setting, pairs in (("default", ("h3k27ac", "mrl")), ("sonnet", ("h3k27ac", "mrl")), ("opus_high", ("mrl",))):
        for pair in pairs:
            for name in ("opal", "biopal"):
                add("1b", f"pilot_{setting}", pair, name, setting=setting)
    # stage 2: seed 42, the agent arms and the HPO winners
    for pair in PAIRS:
        for name in ("biopal", "opal", "biopal_qwen"):
            add("2", "main", pair, name)
        add("2", "main", pair, "hpo", needs=[f"main/{pair}/hpo_study_s42"])
    # stage 3: seeds 43 and 44; an arm that failed twice on a pair at seed 42 is not run again on that pair
    for seed in (43, 44):
        for pair in PAIRS:
            for name in ("plain", "matched", "biopal", "opal", "biopal_qwen"):
                add("3", "main", pair, name, seed, soft=[f"main/{pair}/{name}_s42"])
            add("3", "main", pair, "hpo", seed, needs=[f"main/{pair}/hpo_study_s42"], soft=[f"main/{pair}/hpo_s42"])
    # stage 4: audits on one pair per modality, then replays of other proposers on their saved gates
    audits = [f"ablation_expert/{pair}/audit_s42" for pair in ("h3k27ac", "mrl", "melt")]
    for pair in ("h3k27ac", "mrl", "melt"):
        add("4", "ablation_expert", pair, "audit")
    add("4", "ablation_expert", "all", "replay_qwen_opus", needs=audits)
    add("4", "ablation_expert", "all", "replay_sonnet", needs=audits)
    # stage 5: the same Claude model for both agents, Sonnet at high effort
    for seed in (42, 43, 44):
        for pair in ("h3k27ac", "mrl", "melt"):
            for name in ("biopal", "opal"):
                add("5", "ablation_agent", pair, name, seed, setting="sonnet")
    return rows


def read_manifest():
    path = HERE / "runs.tsv"
    if not path.exists():
        sys.exit("runs.tsv is missing: run `campaign.py manifest` first")
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["seed"], r["need_gb"], r["minutes"], r["runs"] = int(r["seed"]), float(r["need_gb"]), float(r["minutes"]), int(r["runs"])
        r["needs"] = [x for x in r["needs"].split(",") if x]
        r["soft"] = [x for x in r["soft"].split(",") if x]
    return rows


def group_of(run_id):
    """The budget line of section 4 a run counts against, from its id (also for ledgers put aside)."""
    group, _, rest = run_id.partition("/")
    if group == "ready" or group.startswith("pilot_"):
        return "pilot"
    if group == "ablation_expert":
        return "replays"
    if group == "ablation_agent":
        return "agent_ablation"
    return "seed42" if rest.endswith("_s42") else "seeds43_44"


def tier1(row):
    """Checkpoints nobody needs once the run has ended (section 6)."""
    return row["id"].startswith(TINY_GROUPS) or row["kind"] in ("hpo", "audit")


# ------------------------------------------------------------------------------------------------ one run's command
def outdir(run_id):
    return OUT / run_id


def ledger_of(run_id):
    return STATUS / "ledgers" / (run_id.replace("/", "__") + ".jsonl")


def result_file(row):
    return outdir(row["id"]) / ("hpo_result.json" if row["kind"] == "hpo" else "result.json")


def done(row):
    f = result_file(row)
    if not f.exists():
        return False
    if row["kind"] == "hpo":
        try:
            return "best" in json.loads(f.read_text())
        except ValueError:
            return False
    return True


def caps_of(row):
    """(cap of OPAL's agent, cap of the expert or of the hosted replay model), 0 where none is called."""
    s = SETTINGS[row["setting"]]
    if row["name"] == "opal_cap":       # readiness: what does the trainer do when a cap is reached?
        return .01, 0.
    if row["kind"] == "replay":
        return 0., {"replay": .05, "replay_qwen_opus": 3., "replay_sonnet": 2.}.get(row["name"], 0.)
    return (s[4] if "agent" in row["paid"] else 0.), (s[5] if "expert" in row["paid"] else 0.)


def build(row, dry=False):
    """-> (argv, working directory, extra environment) of one run."""
    py = str(ENVS / row["env"] / "bin" / "python")
    out, tiny, name = outdir(row["id"]), row["setting"] == "tiny", row["name"]
    opal_model, opal_effort, expert_model, expert_effort = SETTINGS[row["setting"]][:4]
    agent_cap, expert_cap = caps_of(row)
    env = {}
    if row["kind"] == "replay":
        runs = [str(outdir(d)) for d in row["needs"] if (outdir(d) / "result.json").exists() or dry]
        argv = [py, "-m", "biopal.replay", *runs, "--out", str(out)]
        if name in ("replay", "replay_qwen_opus"):
            tasks = sorted({PAIRS[d.split("/")[1]][0] for d in row["needs"]})
            argv += ["--local-model", QWEN, "--local-revision", QWEN_REV, "--local-url", QWEN_URL, "--local-request",
                     QWEN_REQUEST, "--note", *[str(HERE / "notes" / f"{t}.txt") for t in tasks]]
        hosted = dict(replay=("claude-haiku-5-5", "low"), replay_qwen_opus=("claude-opus-5-5", "high"),
                      replay_sonnet=("claude-sonnet-5-5", "high"))[name]
        argv += ["--hosted-model", hosted[0], "--hosted-effort", hosted[1], "--hosted-budget", f"{expert_cap:g}",
                 "--ledger", str(ledger_of(row["id"])), "--exportable"]   # the six tasks are public data (section 3)
        return argv, REPO, env
    task, model, _, objective, _, _ = PAIRS[row["pair"]]
    size = ["--limit", "6000", "--epochs", "1"] if tiny else ["--epochs", "2"]
    common = ["--model", model, "--task", task, "--objective", objective, "--seed", str(row["seed"]), "--out", str(out)]
    shared = [*size, "--batch-size", "16", "--det", "strict"]
    if row["kind"] == "hpo":
        return [py, "hpo.py", *common, "--mode", "BO", "--n_trials", "1" if tiny else "6", *shared], FT, env
    argv = [py, "finetune.py", *common, *shared]
    few = ["--decisions", "2"] if tiny else []
    gates = ["--outer-decisions", "1", "--outer-steps", "32"] if tiny else []
    paid = ["--budget", f"{agent_cap:g}", "--ledger", str(ledger_of(row["id"]))]
    if "agent" in row["paid"]:
        env.update(OPAL_MODEL=opal_model, OPAL_EFFORT=opal_effort)
    if name == "probe":
        argv = [py, "finetune.py", *common, "--limit", "2000", "--epochs", "1", "--max-steps", "30", "--batch-size",
                "16", "--det", "strict", "--mode", "plain"]
    elif name == "plain":
        argv += ["--mode", "plain"]
    elif name == "matched":
        argv += ["--mode", "plain", *PANEL]
    elif name in ("opal", "opal_cap"):
        argv += ["--mode", "biopal", "--bio-policy", "off", *PANEL, *few, *paid]
    elif name == "biopal":
        argv += ["--mode", "biopal", "--bio-policy", "expert", "--outer", "both", *PANEL, *few, *gates,
                 "--expert-model", expert_model, "--expert-effort", expert_effort, "--expert-budget", f"{expert_cap:g}", *paid]
    elif name == "biopal_qwen":
        argv += ["--mode", "biopal", "--bio-policy", "expert", "--outer", "both", *PANEL, *few, *gates,
                 "--expert-backend", "local", "--expert-model", QWEN, "--expert-revision", QWEN_REV, "--expert-url",
                 QWEN_URL, "--expert-request", QWEN_REQUEST, "--expert-note", str(HERE / "notes" / f"{task}.txt"), *paid]
    elif name == "hpo":                 # the winner of the study, trained once more and scored on test
        study = outdir(row["needs"][0]) / "hpo_result.json"
        best = json.loads(study.read_text())["best"]["params"] if study.exists() else dict(
            lr="<lr>", weight_decay="<weight decay>", warmup_frac="<warm-up>")
        argv += ["--mode", "plain", "--lr", repr(best["lr"]), "--weight-decay", repr(best["weight_decay"]),
                 "--warmup-frac", repr(best["warmup_frac"])]
    elif name == "audit":
        argv += ["--mode", "keep", *PANEL, "--bio-policy", "expert", "--expert-script", str(HERE / "keep.json"),
                 "--expert-shadow", "masked", "--outer", "both", *(gates or ["--outer-steps", "64"]),
                 "--outer-audit", *(["16", "32"] if tiny else ["16", "32", "64", "128"]), "--no-test"]
    else:
        raise ValueError(f"no command for {row['id']}")
    return argv, FT, env


def shown(argv, cwd, env, card):
    quote = lambda s: "'" + s + "'" if any(c in s for c in ' {}"<>') else s   # noqa: E731
    head = " ".join(f"{k}={v}" for k, v in {"CUDA_VISIBLE_DEVICES": card, **env}.items())
    return f"(cd {cwd} && {head} {' '.join(quote(str(a)) for a in argv)})"


# --------------------------------------------------------------------------------------- state shared by all runners
@contextmanager
def locked():
    STATUS.mkdir(parents=True, exist_ok=True)
    with open(STATUS / "lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def alive(pid):
    try:
        os.kill(int(pid), 0)
        with open(f"/proc/{int(pid)}/stat") as f:      # a zombie is not alive
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, ValueError, IndexError):
        return False


def note(text, file="events.log"):
    STATUS.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}"
    with open(STATUS / file, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def attempts():
    """{run id: its attempts so far}, without those put aside by `fresh-ready`."""
    out, path = defaultdict(list), STATUS / "attempts.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if "reset" in r:
                for k in [k for k in out if k.startswith(r["reset"])]:
                    del out[k]
            else:
                out[r["id"]].append(r)
    return out


def failed(run_id, att):
    return sum(a["outcome"] != "ok" for a in att.get(run_id, []))


def running_entries():
    out = []
    for f in sorted((STATUS / "running").glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except ValueError:
            pass
    return out


def runner_stages():
    """Stages some live runner is working on."""
    out = set()
    for f in (STATUS / "runners").glob("*.json"):
        try:
            r = json.loads(f.read_text())
        except ValueError:
            continue
        if alive(r["pid"]):
            out.update(r["stages"])
        else:
            f.unlink(missing_ok=True)
    return out


def ledger_sum(path):
    total = 0.
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                total += float(json.loads(line)["usd"])
    return total


def spend(att, entries):
    """USD per budget line, counted as section 4 says: what the ledgers hold, the full cap of OPAL's agent for
    every failed attempt of a run (the agent books its spend only when a run ends), and what a run that is
    going on may still spend under its caps."""
    per = defaultdict(float)
    for f in (STATUS / "ledgers").rglob("*.jsonl"):
        per[group_of(f.stem.replace("__", "/"))] += ledger_sum(f)
    for run_id, rows in att.items():
        per[group_of(run_id)] += sum(a.get("agent_cap", 0.) for a in rows if a["outcome"] != "ok")
    for e in entries:
        per[group_of(e["id"])] += max(e.get("cap", 0.) - ledger_sum(ledger_of(e["id"])), 0.)
    return per


def free_gb():
    return shutil.disk_usage(REPO).free / 1e9


def gpus():
    """{card: free memory in GB}"""
    p = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60)
    out = {}
    for line in p.stdout.strip().splitlines():
        i, total, used = (x.strip() for x in line.split(","))
        out[int(i)] = (float(total) - float(used)) / 1024
    return out


def pick(need, entries, avoid=None):
    """The card for a job that needs `need` GB free, or None (section 5). A job started in the last three minutes
    has not taken its memory yet, so its need is subtracted."""
    ours = Counter(e["card"] for e in entries if e["card"] is not None)
    if sum(ours.values()) >= MAX_GPU_JOBS:
        return None
    free, now = gpus(), time.time()
    for e in entries:
        if e["card"] is not None and now - e["start"] < 180:
            free[e["card"]] = free.get(e["card"], 0) - e["need"]
    room = [c for c in free if free[c] >= need + 2]
    pool = [c for c in room if ours[c] == 0] or ([c for c in room if ours[c] == 1] if sum(ours.values()) < LANES else [])
    if avoid is not None and len(pool) > 1:
        pool = [c for c in pool if c != avoid]
    return max(pool, key=lambda c: free[c]) if pool else None


def server_up():
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(QWEN_URL + "/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def drop_checkpoints(folder):
    n = 0
    for f in Path(folder).rglob("best.pt"):
        f.unlink()
        n += 1
    return n


# --------------------------------------------------------------------------------------------------- the status table
def state_of(row, att, entries):
    if done(row):
        return "done"
    if any(e["id"] == row["id"] for e in entries):
        return "running"
    return "failed" if failed(row["id"], att) >= 2 else "pending"


def write_status(rows, att, entries, reasons):
    per = spend(att, entries)
    lines = ["\t".join(["id", "stage", "state", "detail", "card", "minutes", "usd"])]
    count = defaultdict(Counter)
    for r in rows:
        s = state_of(r, att, entries)
        e = next((e for e in entries if e["id"] == r["id"]), None)
        ok = next((a for a in att.get(r["id"], []) if a["outcome"] == "ok"), None)
        detail = reasons.get(r["id"], "") if s == "pending" else ""
        if s in ("pending", "failed") and failed(r["id"], att):
            detail = (detail + "; " if detail else "") + f"{failed(r['id'], att)} failed attempt(s): " + \
                att[r["id"]][-1].get("why", "")
        usd = ledger_sum(ledger_of(r["id"]))
        lines.append("\t".join([r["id"], r["stage"], s, detail, "" if e is None or e["card"] is None else str(e["card"]),
                                f"{ok['seconds'] / 60:.1f}" if ok else "", f"{usd:.3f}" if usd else ""]))
        count[r["stage"]][s] += r["runs"]
    (STATUS / "status.tsv").write_text("\n".join(lines) + "\n")
    summary = [f"{time.strftime('%Y-%m-%d %H:%M:%S')}  free disk {free_gb():.0f} GB"]
    for stage in sorted(count):
        summary.append(f"stage {stage:2s}: " + ", ".join(f"{n} {s}" for s, n in sorted(count[stage].items())))
    summary.append("Claude USD (ledgers, failed attempts at full cap, reservations of running jobs): "
                   + ", ".join(f"{g} {per[g]:.2f} of {CAPS[g]:g}" for g in CAPS) + f"; total {sum(per.values()):.2f} of {TOTAL_CAP:g}")
    (STATUS / "summary.txt").write_text("\n".join(summary) + "\n")
    return summary


def load_reasons():
    try:
        return json.loads((STATUS / "reasons.json").read_text())
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------------------------------------------------ running
FINAL = ("skipped", "deferred", "budget cap", "server", "claude")       # reasons a runner does not wait for


def finish(row, entry, code, why):
    """Book the end of an attempt. Under the lock."""
    ok = code == 0 and done(row)
    out = outdir(row["id"])
    if ok:
        if tier1(row):
            drop_checkpoints(out)
    else:
        why = why or ("no result file" if code == 0 else f"exit code {code}")
        log = Path(entry["log"])
        if log.exists() and "out of memory" in log.read_text(errors="replace")[-20000:].lower():
            why += " (out of GPU memory)"
        if out.exists() and row["kind"] != "hpo":      # an HPO study continues in its folder; the trainer needs a new one
            drop_checkpoints(out)
            n = 1
            while out.with_name(out.name + f".failed{n}").exists():
                n += 1
            out.rename(out.with_name(out.name + f".failed{n}"))
        elif out.exists():
            drop_checkpoints(out)
        note(f"FAILED {row['id']} on card {entry['card']}: {why}; log {entry['log']}", "failures.log")
    with open(STATUS / "attempts.jsonl", "a") as f:
        f.write(json.dumps(dict(id=row["id"], outcome="ok" if ok else "failed", why=why or "", card=entry["card"],
                                start=entry["start"], seconds=round(time.time() - entry["start"]),
                                agent_cap=entry.get("agent_cap", 0.), log=entry["log"])) + "\n")
    (STATUS / "running" / (row["id"].replace("/", "__") + ".json")).unlink(missing_ok=True)
    if ok:
        note(f"done {row['id']} in {(time.time() - entry['start']) / 60:.1f} min")
    return ok


def why_not(row, by_id, att, entries, reasons, ready, stages_alive, per):
    """None if the run may start now, else the reason (one that starts with a word of FINAL will not change
    while this runner lives)."""
    if row["paid"] and not ready.get("claude"):
        return "claude: the readiness check of Claude did not pass"
    if row["stage"] != "0":             # what did not work at readiness size is not tried at full size
        if ready.get(f"ok_pair_{row['pair']}") is False:
            return "skipped: this pair did not load in the readiness stage"
        if ready.get("ok_replay" if row["kind"] == "replay" else f"ok_{row['name']}") is False:
            return "skipped: the readiness run of this kind of run failed"
    resolved = []
    for dep in row["needs"]:
        d = by_id[dep]
        s = state_of(d, att, entries)
        if s == "done":
            resolved.append(True)
        elif s == "failed" or reasons.get(dep, "").startswith(FINAL):
            resolved.append(False)
        elif s == "running" or d["stage"] in stages_alive:
            return f"waiting for {dep}"
        else:
            resolved.append(False)
    if row["needs"] and not (any(resolved) if row["kind"] == "replay" else all(resolved)):
        return "skipped: " + ", ".join(d for d, ok in zip(row["needs"], resolved) if not ok) + " has no result"
    for dep in row["soft"]:             # the same arm at seed 42: wait while it is undecided, skip if it failed twice
        d = by_id[dep]
        s = state_of(d, att, entries)
        if s == "failed":
            return f"skipped: {dep} failed twice"
        if s == "running" or (s == "pending" and d["stage"] in stages_alive and not reasons.get(dep, "").startswith(FINAL)):
            return f"waiting for {dep}"
    if row["stage"] == "3" and row["paid"] and row["seed"] == 44:       # whole seeds first (section 3)
        for other in by_id.values():
            if other["stage"] == "3" and other["paid"] and other["seed"] == 43 and not att.get(other["id"]) \
                    and state_of(other, att, entries) == "pending" and not reasons.get(other["id"], "").startswith(FINAL):
                return "after the paid runs of seed 43"
    if free_gb() < DISK_STOP_GB:
        return f"less than {DISK_STOP_GB} GB of disk free"
    if row["paid"]:
        cap = sum(caps_of(row))
        left = max(cap - ledger_sum(ledger_of(row["id"])), 0.)
        g = group_of(row["id"])
        if per[g] + left > CAPS[g] or sum(per.values()) + left > TOTAL_CAP:
            if any(e.get("cap") for e in entries):
                return "waiting for running paid jobs to book their spend"
            return f"budget cap reached ({g} {per[g]:.2f} of {CAPS[g]:g}, total {sum(per.values()):.2f} of {TOTAL_CAP:g})"
    return None


def start(row, card):
    argv, cwd, extra = build(row)
    n = 1
    LOGS.mkdir(parents=True, exist_ok=True)
    while (LOGS / f"{row['id'].replace('/', '__')}.attempt{n}.log").exists():
        n += 1
    log = LOGS / f"{row['id'].replace('/', '__')}.attempt{n}.log"
    env = dict(os.environ, **extra)
    env.update(CUDA_VISIBLE_DEVICES="" if card is None else str(card), CONDA_DEFAULT_ENV=row["env"],
               CONDA_PREFIX=str(ENVS / row["env"]),
               PATH=f"{ENVS / row['env'] / 'bin'}:{Path.home() / '.local' / 'bin'}:{os.environ.get('PATH', '')}")
    outdir(row["id"]).parent.mkdir(parents=True, exist_ok=True)
    ledger_of(row["id"]).parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w")
    fh.write(shown(argv, cwd, extra, "" if card is None else card) + "\n\n")
    fh.flush()
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    agent_cap, other_cap = caps_of(row)
    entry = dict(id=row["id"], pid=proc.pid, card=card, need=row["need_gb"], start=time.time(), runner=os.getpid(),
                 cap=agent_cap + other_cap if row["paid"] else 0., agent_cap=agent_cap if row["paid"] else 0.,
                 limit=max(3 * row["minutes"], MIN_LIMIT_MIN) * 60, log=str(log))
    (STATUS / "running").mkdir(parents=True, exist_ok=True)
    (STATUS / "running" / (row["id"].replace("/", "__") + ".json")).write_text(json.dumps(entry))
    note(f"start {row['id']} on card {card} (attempt {n}, limit {entry['limit'] / 60:.0f} min)")
    return proc, entry, fh


def stop(pid):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(int(pid), sig)
        except OSError:
            return
        for _ in range(30):
            if not alive(pid):
                return
            time.sleep(1)


def order(rows, catchup):
    if catchup:      # what is cut last comes first (section 8)
        def key(r):
            rank = {"2": 0, "1": 0, "3": 1, "4": 2, "5": 4}[r["stage"]]
            if r["stage"] == "3" and r["name"] == "biopal_qwen" and r["seed"] == 44:
                rank = 3
            if r["stage"] == "4" and r["pair"] == "melt":
                rank = 3.5
            return rank, r["seed"], -r["minutes"]
        return sorted(rows, key=key)
    return sorted(rows, key=lambda r: (r["kind"] != "hpo", r["seed"], -r["minutes"]))   # longest first: a study is a chain


def run(args):
    rows = read_manifest()
    by_id = {r["id"]: r for r in rows}
    stages = [s for s in ("1", "2", "3", "4", "5")] if args.catchup else args.stage
    mine = order([r for r in rows if r["stage"] in stages and (not args.only or args.only in r["id"])], args.catchup)
    if args.dry_run:
        for r in mine:
            mark = "done   " if done(r) else "       "
            print(f"{mark}[{r['id']}] needs {r['need_gb']:g} GB, about {r['minutes']:g} min"
                  + (f", Claude cap ${sum(caps_of(r)):g}" if r["paid"] else "") + "\n  " + shown(*build(r, dry=True), "<card>"))
        print(f"\n{len(mine)} commands, {sum(r['runs'] for r in mine)} runs "
              f"({sum(r['runs'] for r in mine if r['kind'] != 'replay')} without replays), "
              f"{sum(r['minutes'] for r in mine) / 60:.1f} GPU-hours expected, "
              f"Claude caps add up to ${sum(sum(caps_of(r)) for r in mine if r['paid']):.2f}")
        return
    t0 = float((STATUS / "t0").read_text()) if (STATUS / "t0").exists() else None
    deadline = t0 + args.deadline * 3600 if t0 and args.deadline else None
    try:
        ready = json.loads((STATUS / "readiness.json").read_text())
    except (OSError, ValueError):
        ready = {}
    (STATUS / "runners").mkdir(parents=True, exist_ok=True)
    me = STATUS / "runners" / f"{os.getpid()}.json"
    me.write_text(json.dumps(dict(pid=os.getpid(), stages=stages)))
    note(f"runner for stage(s) {', '.join(stages)}: {sum(not done(r) for r in mine)} of {len(mine)} commands to do"
         + (f", deadline {time.strftime('%Y-%m-%d %H:%M', time.localtime(deadline))}" if deadline else ", no deadline"))
    procs, server_tried, floor_said, last_print = {}, False, False, 0.
    try:
        while True:
            want_server = False
            with locked():
                att, entries, reasons = attempts(), [], load_reasons()
                for e in running_entries():      # jobs of a runner that is gone: book how they ended
                    if e["id"] in procs or alive(e["pid"]) or alive(e["runner"]):
                        entries.append(e)
                    elif e["id"] in by_id:
                        finish(by_id[e["id"]], e, 0 if done(by_id[e["id"]]) else 1, "its runner was gone when it ended")
                for run_id, (proc, entry, fh) in list(procs.items()):
                    code, why = proc.poll(), None
                    if code is None and time.time() - entry["start"] > entry["limit"]:
                        stop(proc.pid)
                        code, why = proc.wait(), f"stopped after the time limit of {entry['limit'] / 60:.0f} min"
                    if code is not None:
                        fh.close()
                        finish(by_id[run_id], entry, code, why)
                        del procs[run_id]
                        entries = [e for e in entries if e["id"] != run_id]
                att = attempts()
                if free_gb() < DISK_FLOOR_GB:    # the safety floor: no checkpoint of a finished run is kept
                    busy = {e["id"] for e in entries}
                    n = sum(drop_checkpoints(outdir(r["id"])) for r in rows if r["id"] not in busy and outdir(r["id"]).exists())
                    if n or not floor_said:
                        note(f"free disk under {DISK_FLOOR_GB} GB: deleted {n} kept checkpoints")
                        floor_said = True
                per, stages_alive = spend(att, entries), runner_stages()
                late = deadline is not None and time.time() > deadline
                for r in mine:
                    s = state_of(r, att, entries)
                    if s != "pending":
                        reasons.pop(r["id"], None)
                        continue
                    why = "deferred: the stage's deadline has passed" if late else \
                        why_not(r, by_id, att, entries, reasons, ready, stages_alive, per)
                    if why is None and "qwen" in r["name"] + r["id"] or why is None and r["name"] == "replay":
                        if not server_up():
                            why = "server: the local expert does not answer" if server_tried else "waiting for the local server"
                            want_server = not server_tried
                    card = None
                    if why is None and r["need_gb"] > 0:
                        last = att.get(r["id"], [{}])[-1].get("card")
                        card = pick(r["need_gb"], entries, avoid=last)
                        if card is None:
                            why = "no card with enough free memory"
                    if why is None:
                        proc, entry, fh = start(r, card)
                        procs[r["id"]] = (proc, entry, fh)
                        entries.append(entry)
                        per[group_of(r["id"])] += max(entry["cap"] - ledger_sum(ledger_of(r["id"])), 0.)
                        reasons.pop(r["id"], None)
                    else:
                        reasons[r["id"]] = why
                (STATUS / "reasons.json").write_text(json.dumps(reasons, indent=1))
                summary = write_status(rows, att, entries, reasons)
                open_ = [r for r in mine if state_of(r, att, entries) == "pending"]
                waiting = [r for r in open_ if not reasons.get(r["id"], "").startswith(FINAL)]
            if time.time() - last_print > 600:
                print("\n".join(summary), flush=True)
                last_print = time.time()
            if want_server:
                server_tried = True
                note("the local expert does not answer: starting it")
                subprocess.run([str(HERE / "qwen_server.sh"), "start"], timeout=1500)
                continue
            if not procs and not waiting:
                break
            time.sleep(args.poll)
    finally:
        me.unlink(missing_ok=True)
        for run_id, (proc, entry, fh) in procs.items():   # the runner is stopped: its jobs go on and are booked later
            note(f"runner stops while {run_id} is still going (pid {proc.pid})")
    with locked():
        att, entries, reasons = attempts(), running_entries(), load_reasons()
        print("\n".join(write_status(rows, att, entries, reasons)), flush=True)
    left = [r for r in mine if not done(r)]
    note(f"runner for stage(s) {', '.join(stages)} ends: {len(mine) - len(left)} of {len(mine)} commands have a result"
         + ("" if not left else "; without: " + ", ".join(f"{r['id']} ({reasons.get(r['id'], 'failed')})" for r in left[:40])))
    return 0


# ------------------------------------------------------------------------------------------------------------- checks
def note_hash(path):
    text = path.read_text().partition("\n\n")[2].strip()      # as biopal/expert.py hashes a note
    return hashlib.sha256(text.encode()).hexdigest()


def check(args):
    STATUS.mkdir(parents=True, exist_ok=True)
    result, lines = {}, []

    def say(key, ok, text):
        result[key] = bool(ok)
        lines.append(f"{'PASS' if ok else 'FAIL'}  {text}")
        print(lines[-1], flush=True)

    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    (STATUS / "repo.txt").write_text(f"commit {git('rev-parse', 'HEAD')}\n\n{git('status', '--short')}\n")
    say("repo", True, f"repository at commit {git('rev-parse', '--short', 'HEAD')}, "
        f"{len(git('status', '--short').splitlines())} changed or new files (status/repo.txt)")
    envs = all((ENVS / e / "bin" / "python").exists() for e in ("biopal-v2", "bio-rna", "agent"))
    opal = []
    for e in ("biopal-v2", "bio-rna"):
        p = subprocess.run([str(ENVS / e / "bin" / "python"), "-c", "import importlib.metadata as m; print(m.version('opal-optimizer'))"],
                           capture_output=True, text=True, timeout=300)
        opal.append(p.stdout.strip())
    say("envs", envs and opal == ["0.3.10", "0.3.10"], f"conda environments biopal-v2, bio-rna, agent; OPAL {opal}")
    hashes = dict(line.split("\t") for line in (HERE / "notes" / "hashes.tsv").read_text().splitlines() if line.strip())
    bad = [t for t, h in hashes.items() if not (HERE / "notes" / f"{t}.txt").exists() or note_hash(HERE / "notes" / f"{t}.txt") != h]
    say("notes", not bad and len(hashes) == 6, "the six notes match their recorded hashes" if not bad else f"notes changed or missing: {bad}")
    say("disk", free_gb() >= 150, f"{free_gb():.0f} GB of disk free (150 wanted)")
    free = gpus()
    small, large = sum(f >= 14 for f in free.values()), sum(f >= 32 for f in free.values())
    say("gpus", small >= LANES and large >= 1, f"{small} cards with room for a small job, {large} with room for the melting-point job; "
        + "free GB per card: " + " ".join(f"{c}:{f:.0f}" for c, f in sorted(free.items())))
    say("qwen", server_up(), f"local expert at {QWEN_URL} " + ("answers" if server_up() else "does not answer (./qwen_server.sh start)"))
    if args.no_live:
        say("claude", False, "Claude not checked (--no-live): paid runs stay off")
    else:
        env = dict(os.environ, PATH=f"{ENVS / 'biopal-v2' / 'bin'}:{Path.home() / '.local' / 'bin'}:{os.environ.get('PATH', '')}")
        p = subprocess.run([str(ENVS / "biopal-v2" / "bin" / "opal"), "doctor", "--live"], capture_output=True, text=True,
                           timeout=600, env=env, cwd=str(REPO))
        (STATUS / "opal_doctor.txt").write_text(p.stdout + p.stderr)
        say("claude", p.returncode == 0, f"`opal doctor --live` exit code {p.returncode} (status/opal_doctor.txt)")
        ledger = STATUS / "ledgers" / "ready__doctor.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as f:      # the check makes one small Claude call; OPAL calls it "a few cents"
            f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), role="opal", usd=.05, note="estimate")) + "\n")
    (STATUS / "readiness.json").write_text(json.dumps(result, indent=1))
    (STATUS / "readiness.txt").write_text("\n".join(lines) + "\n")
    return 0


def verdict(args):
    """After the runs of stage 0: which kinds of run and which pairs work. Later stages skip the others."""
    rows = [r for r in read_manifest() if r["stage"] == "0"]
    try:
        ready = json.loads((STATUS / "readiness.json").read_text())
    except (OSError, ValueError):
        ready = {}
    lines = []
    for r in rows:
        ok, usd = done(r), ledger_sum(ledger_of(r["id"]))
        extra = f", Claude ${usd:.3f}" if r["paid"] else ""
        if r["name"] == "probe":
            ready[f"ok_pair_{r['pair']}"] = ok
            if ok:
                extra += f", {json.loads(result_file(r).read_text())['seconds_per_update']:.3f} s per update"
        elif r["name"] == "opal_cap":   # for the record only: what a cap that is reached does to a run
            extra += f" of a cap of $0.01; the run {'finished' if ok else 'did not finish'}"
        else:
            ready["ok_replay" if r["kind"] == "replay" else f"ok_{r['name']}"] = ok
        lines.append(f"{'PASS' if ok or r['name'] == 'opal_cap' else 'FAIL'}  {r['id']}{extra}")
    (STATUS / "readiness.json").write_text(json.dumps(ready, indent=1))
    with open(STATUS / "readiness.txt", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    off = sorted(k[3:] for k, v in ready.items() if k.startswith("ok_") and v is False)
    print("everything that was tried works" if not off else "switched off for the later stages: " + ", ".join(off))
    return 0


def fresh_ready(args):
    """Put the outputs of an earlier stage 0 aside, so that a launch tests the machine as it is now. What they
    cost stays in the ledgers and counts."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    with locked():
        if any(e["id"].startswith("ready/") and alive(e["pid"]) for e in running_entries()):
            sys.exit("a readiness run is still going")
        if (OUT / "ready").exists():
            (OUT / "ready").rename(OUT / f"ready.{stamp}")
        with open(STATUS / "attempts.jsonl", "a") as f:
            f.write(json.dumps(dict(reset="ready/")) + "\n")
        for f in list((STATUS / "ledgers").glob("ready__*.jsonl")):
            (STATUS / "ledgers" / "earlier").mkdir(exist_ok=True)
            f.rename(STATUS / "ledgers" / "earlier" / f"{f.stem}__{stamp}.jsonl")
    print(f"earlier readiness outputs put aside ({stamp})")
    return 0


def status(args):
    rows = read_manifest()
    with locked():
        print("\n".join(write_status(rows, attempts(), [e for e in running_entries() if alive(e["pid"])], load_reasons())))
    print(f"table: {STATUS / 'status.tsv'}")
    return 0


def cleanup(args):
    if not args.yes:
        sys.exit("this deletes every kept checkpoint (best.pt) of the campaign; repeat with --yes")
    with locked():
        busy = {e["id"] for e in running_entries() if alive(e["pid"])}
        n = sum(drop_checkpoints(d) for d in OUT.glob("*/*/*") if d.is_dir() and "/".join(d.parts[-3:]) not in busy)
    note(f"sign-off cleanup: deleted {n} checkpoints")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("manifest")
    c = sub.add_parser("check")
    c.add_argument("--no-live", action="store_true", help="skip the one small Claude call; paid runs then stay off")
    r = sub.add_parser("run")
    r.add_argument("--stage", nargs="+", default=[], choices=["0", "1", "1b", "2", "3", "4", "5"])
    r.add_argument("--catchup", action="store_true")
    r.add_argument("--deadline", type=float, help="hours after the campaign's start (status/t0) at which nothing new starts")
    r.add_argument("--only", help="only runs whose id contains this text")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--poll", type=float, default=15., help="seconds between looks at the jobs")
    sub.add_parser("status")
    sub.add_parser("verdict")
    sub.add_parser("fresh-ready")
    k = sub.add_parser("cleanup")
    k.add_argument("--yes", action="store_true")
    args = p.parse_args()
    if args.command == "manifest":
        rows = manifest()
        with (HERE / "runs.tsv").open("w", newline="") as f:
            w = csv.DictWriter(f, FIELDS, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        per = Counter()
        for r in rows:
            per[r["stage"]] += r["runs"]
        print(f"runs.tsv: {len(rows)} rows; runs per stage: " + ", ".join(f"{s}: {n}" for s, n in sorted(per.items())))
        return 0
    if args.command == "run" and not (args.stage or args.catchup):
        p.error("run needs --stage or --catchup")
    return dict(check=check, run=run, status=status, cleanup=cleanup, verdict=verdict,
                **{"fresh-ready": fresh_ready})[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
