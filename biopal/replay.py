"""Replay the gates of saved runs with other proposers and compare their choices on the recorded trials.

    python -m biopal.replay RUN_DIR [RUN_DIR ...] --out DIR [--local-model NAME --local-revision SHA --note FILE]

A run made with --outer-audit scored every action on the menu at every gate, each from the same snapshot. Replay
asks other proposers what they would have done at those checkpoints. Each reads the brief the run saved, without
its earlier-gates section (that history is the run's own proposer's), and answers once per gate. Every answer is
written to DIR/proposals.jsonl before a trial score is read; then each choice is joined to the recorded loss
difference on the private rows. `keep`, chosen or forced by a failed call, scores 0: it is the incumbent.

Proposers
  always_keep           never changes anything
  uniform               the mean over every legal action of the gate, keep included (an expectation, no draw)
  worst_gap             the offered cohort with the largest baseline gap among those whose 95% interval lies above
                        0; keep when there is none
  local:masked          the local model on the masked brief
  local:informed        the local model on the informed brief (names and public background)
  local:informed+note   the same brief with the task owner's note (--note; its header binds it to one task)
  hosted:informed       Claude on the informed brief, the text local:informed reads (--hosted-model, --exportable)

A gate enters a comparison at a horizon only if every legal action has a score there, so all proposers are
compared on the same gates; what is missing or failed is listed. incumbent_repeat and refit_head are controls,
not choices: the first shows how far a repeat of the incumbent drifts, the second what refitting the head on
the block in use does, which a readout switch has to beat.

DIR/result.json is the shareable result: choices, joins and summaries, no reply text and no private note.
DIR/local/ holds the briefs and the raw replies. A reply can repeat a note: that folder stays on this machine.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import expert

KEEP = ("keep", None)


def records(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def checkpoints(path):
    """What a proposer may know of a run: the schedule and, per gate, the measurements and the menu."""
    path = Path(path)
    log, recipe = records(path / "outer.jsonl"), json.loads((path / "config.json").read_text())["recipe"]
    gates = [dict(gate=r["gate"], update=r["update"], table=r["table"], menu=r["menu"])
             for r in log if r["event"] == "table"]
    return dict(path=path, name=path.name, task=recipe["task"], model=recipe["model"], det=recipe.get("det"),
                schedule=next(r for r in log if r["event"] == "schedule"), gates=gates)


def trials(path):
    """{gate: trial record} of a run: the private scores. Read only once every answer is on disk."""
    return {r["gate"]: r for r in records(Path(path) / "outer.jsonl") if r["event"] == "trial"}


def legal(menu):
    """Every action a proposer may choose at a gate."""
    return ([KEEP] + [("set_cohort", c) for c in menu.get("set_cohort", [])]
            + [("ordinary_weighting", None)] * bool(menu.get("ordinary_weighting"))
            + [("set_readout", k) for k in menu.get("set_readout", [])])


def label(action):
    kind, target = action
    return kind + ("" if target is None else f" block_{target}" if kind in ("set_readout", "refit_head")
                   else f" {target}")


def without_history(text):
    """A saved brief without its earlier-gates section."""
    lines = text.split("\n")
    try:
        start = next(i for i, x in enumerate(lines) if x.startswith("EARLIER GATES"))
        end = lines.index("ACTIONS: choose exactly one")
    except (StopIteration, ValueError):
        raise ValueError("not a brief this version can replay: no earlier-gates or actions section")
    return "\n".join(lines[:start] + lines[end:])


def names_of(run, role, gate, text):
    """{real name: the name this role reads}. Taken from the run's log; a log from before names were recorded is
    matched to the brief through the row ids of the measurement table, and refused unless the brief shows every
    row of the log and every cohort under exactly one name of its own. Either way the brief must offer the menu's
    cohorts in the menu's order."""
    names = (run["schedule"].get("names") or {}).get(role)
    if names is None:
        real, seen, rows = {r["id"]: r["cohort"] for r in gate["table"]}, {}, set()
        for line in text.splitlines():
            f = line.split(" | ")
            if len(f) == 5 and f[0] in real:
                rows.add(f[0])
                seen.setdefault(real[f[0]], set()).add(f[2])
        names = {cohort: min(read) for cohort, read in seen.items()}
        if rows != set(real) or any(len(read) > 1 for read in seen.values()) \
                or len(set(names.values())) < len(names):
            raise ValueError(f"{run['name']} gate {gate['gate'] + 1}: the {role} brief does not show which of its "
                             "cohort names is which cohort of the log")
    offer = [x for x in text.splitlines() if x.startswith("  set_cohort <cohort>")]
    offered = offer[0].split("On offer: ")[1].split(", ") if offer else []
    if offered != [names.get(c, c) for c in gate["menu"].get("set_cohort", [])]:
        raise ValueError(f"{run['name']} gate {gate['gate'] + 1}: the {role} brief does not offer the cohorts of "
                         "the log's menu")
    return names


def worst_gap(table, menu):
    """Rule baseline: among the baseline-gap rows of offered cohorts whose interval lies above 0, the largest gap
    (ties: the cohort listed first on the menu); keep when no row qualifies."""
    offered = menu.get("set_cohort", [])
    rows = [r for r in table if r["measurement"] == "baseline" and r["cohort"] in offered and r["lo"] > 0]
    if not rows:
        return KEEP
    return "set_cohort", max(rows, key=lambda r: (r["gap"], -offered.index(r["cohort"])))["cohort"]


def ranks(x):
    """Ranks from 1; tied values share the mean of their ranks."""
    x = np.asarray(x, float)
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def spearman(x, y):
    """Rank correlation with ties at their mean rank. None where it is undefined: fewer than two pairs, or one
    side constant."""
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def percentile(value, scores):
    """Where a choice stands among the legal actions of its gate: 1 for the lowest loss difference, 0 for the
    highest, ties at the middle of their span. None for a menu of one."""
    if len(scores) < 2:
        return None
    worse, tied = sum(s > value for s in scores), sum(s == value for s in scores) - 1
    return (worse + tied / 2) / (len(scores) - 1)


def score(branches, action, h):
    """(mean loss difference on the private rows against keep, standard error) after h updates; keep is 0 by
    definition. None for a branch that is missing, failed or not finite."""
    if action == KEEP:
        return 0., 0.
    v = (branches.get(action) or {}).get("horizons", {}).get(str(h))
    return (v["mean_d"], v["se"]) if v and math.isfinite(v["mean_d"]) else None


def answers(run, gate, local, hosted, note, out):
    """One row per proposer for one gate: the action and target in the run's real names. ok is False for a call
    that failed; the gate then keeps the incumbent, as in training, and the row says why in one word."""
    base, n = dict(run=run["name"], gate=gate["gate"], update=gate["update"]), gate["gate"] + 1
    yield dict(base, proposer="always_keep", action="keep", target=None, ok=True)
    action, target = worst_gap(gate["table"], gate["menu"])
    yield dict(base, proposer="worst_gap", action=action, target=target, ok=True)
    asked = [("local:masked", local, "masked", None), ("local:informed", local, "informed", None)] if local else []
    asked += [("local:informed+note", local, "informed", note)] * bool(local and note)
    asked += [("hosted:informed", hosted, "informed", None)] * bool(hosted)   # never the masked brief, never a note
    folder = out / "local" / run["name"]
    folder.mkdir(parents=True, exist_ok=True)
    for proposer, client, role, add in asked:
        text = without_history((run["path"] / "expert" / f"gate{n}_{role}_brief.txt").read_text())
        shown = expert.shown(names_of(run, role, gate, text), gate["menu"])
        (folder / f"gate{n}_{role}_brief.txt").write_text(text)
        client.folder, name, tick = folder, proposer.replace(":", "_"), time.perf_counter()
        try:
            answer = client.propose(name, gate["gate"], expert.with_note(text, add) if add else text)
            action, target = expert.check(answer, gate["menu"], shown)
            row = dict(action=action, target=target, ok=True, predicted_sign=answer["predicted_sign"])
        except Exception as e:
            (folder / f"gate{n}_{name}_error.txt").write_text(f"{type(e).__name__}: {e}")
            row = dict(action="keep", target=None, ok=False, failure=expert.failure(e))
        yield dict(base, proposer=proposer, seconds=round(time.perf_counter() - tick, 3), **row)


def mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def join(runs, rows, recorded, primary=None):
    """Join every answer to the recorded trials. -> the result: per gate what each proposer chose and how that
    action scored at every horizon, coverage of the menu, the controls, and summaries over complete gates."""
    steps = {t["trial_updates"] for by_gate in recorded.values() for t in by_gate.values()}
    if not steps:
        raise ValueError("none of the runs recorded a trial: replay needs runs made with --outer-audit")
    if primary is None:
        if len(steps) > 1:
            raise ValueError(f"the runs tested proposals after {sorted(steps)} updates; name one with --horizon")
        primary = steps.pop()
    gates, coverage, horizons = [], [], set()
    for run in runs:
        for g in run["gates"]:
            trial = recorded[run["name"]].get(g["gate"], {})
            branches = {(b["action"], b["target"]): b for b in trial.get("branches", [])}
            actions, at = legal(g["menu"]), sorted({int(h) for b in branches.values() for h in b.get("horizons", {})})
            horizons |= set(at)
            out = dict(run=run["name"], task=run["task"], gate=g["gate"] + 1, update=g["update"],
                       legal=[label(a) for a in actions], complete={}, menu={}, uniform={}, controls={}, picks={})
            block = next((b["target"] for b in branches.values() if b["action"] == "refit_head"), None)
            for h in at:
                scores = {a: score(branches, a, h) for a in actions}
                failed = [label(a) for a, s in scores.items() if s is None and "failed" in branches.get(a, {})]
                missing = [label(a) for a, s in scores.items() if s is None and label(a) not in failed]
                done = out["complete"][str(h)] = not failed and not missing
                coverage.append(dict(run=run["name"], gate=g["gate"] + 1, update=g["update"], horizon=h,
                                     legal=len(actions), scored=len(actions) - len(failed) - len(missing),
                                     missing=missing, failed=failed, complete=done))
                out["menu"][str(h)] = {label(a): s and s[0] for a, s in scores.items()}
                refit, repeat = score(branches, ("refit_head", block), h), score(branches, ("incumbent_repeat", None), h)
                out["controls"][str(h)] = dict(incumbent_repeat=repeat and repeat[0], refit_head=refit and refit[0],
                                               refit_head_block=block)
                if done:
                    out["uniform"][str(h)] = mean([s[0] for s in scores.values()])
                for row in (r for r in rows if r["run"] == run["name"] and r["gate"] == g["gate"]):
                    choice = (row["action"], row["target"])
                    pick = out["picks"].setdefault(row["proposer"], dict(
                        action=row["action"], target=row["target"], choice=label(choice), ok=row["ok"],
                        failure=row.get("failure"), predicted_sign=row.get("predicted_sign"), at={}))
                    s = scores.get(choice)
                    pick["at"][str(h)] = None if s is None else dict(
                        mean_d=s[0], se=s[1],
                        percentile=percentile(s[0], [x[0] for x in scores.values()]) if done else None,
                        # a readout switch also refits the head: what it adds to a refit on the block in use
                        beyond_refit=s[0] - refit[0] if row["action"] == "set_readout" and refit else None)
            gates.append(out)
    proposers = list(dict.fromkeys(r["proposer"] for r in rows))
    pairs = [p for p in (("local:informed+note", "local:informed"), ("local:informed", "local:masked"),
                         ("hosted:informed", "local:informed")) if set(p) <= set(proposers)]
    by_horizon = {}
    for h in sorted(horizons):
        k, table, paired = str(h), {}, {}
        use = {run["name"]: [g for g in gates if g["run"] == run["name"] and g["complete"].get(k)] for run in runs}
        for p in proposers + ["uniform"]:
            by_run = {}
            for name, gs in use.items():
                if p == "uniform":
                    by_run[name] = dict(gates=len(gs), mean_d=mean([g["uniform"][k] for g in gs]))
                    continue
                picks = [g["picks"][p] for g in gs if p in g["picks"]]   # a proposer may sit out a run (no note)
                by_run[name] = dict(
                    gates=len(picks), mean_d=mean([x["at"][k]["mean_d"] for x in picks]),
                    changes=sum(x["action"] != "keep" for x in picks),
                    keeps=sum(x["action"] == "keep" and x["ok"] for x in picks),   # chosen, not forced by a failure
                    failures=sum(not x["ok"] for x in picks),
                    mean_percentile=mean([x["at"][k]["percentile"] for x in picks
                                          if x["at"][k]["percentile"] is not None]))
            table[p] = dict(by_run=by_run, by_task=by_task(runs, by_run, "mean_d"))
        for a, b in pairs:
            by_run = {}
            for name, gs in use.items():
                each = [dict(gate=g["gate"], first=g["picks"][a]["choice"], second=g["picks"][b]["choice"],
                             difference=g["picks"][a]["at"][k]["mean_d"] - g["picks"][b]["at"][k]["mean_d"])
                        for g in gs if a in g["picks"] and b in g["picks"]]
                by_run[name] = dict(gates=len(each), disagreements=sum(x["first"] != x["second"] for x in each),
                                    mean_difference=mean([x["difference"] for x in each]), per_gate=each)
            paired[f"{a} minus {b}"] = dict(by_run=by_run, by_task=by_task(runs, by_run, "mean_difference"))
        ranks_ = []   # does the menu keep its order between this horizon and the primary one?
        for g in gates:
            both = [(x, y) for x, y in zip(g["menu"].get(k, {}).values(), g["menu"].get(str(primary), {}).values())
                    if x is not None and y is not None][1:]   # without keep, which is 0 at every horizon
            ranks_.append(dict(run=g["run"], gate=g["gate"], actions=len(both),
                               spearman=spearman([x for x, _ in both], [y for _, y in both])))
        by_horizon[k] = dict(complete_gates={name: [g["gate"] for g in gs] for name, gs in use.items()},
                             proposers=table, paired=paired, menu_rank_correlation_with_primary=ranks_)
    return dict(primary_horizon=primary, horizons=sorted(horizons), coverage=coverage, gates=gates,
                by_horizon=by_horizon)


def by_task(runs, by_run, key):
    """Runs are the unit: gates of one run share a trajectory, so a task's value is the mean of its runs' means
    and no test over pooled gates is given."""
    out = {}
    for task in dict.fromkeys(r["task"] for r in runs):
        values = [by_run[r["name"]][key] for r in runs if r["task"] == task and by_run[r["name"]][key] is not None]
        out[task] = dict(runs=len(values), mean=mean(values))
    return out


def show(result):
    """The result as text: coverage, the proposers at the primary horizon, the paired comparisons, the example."""
    def number(x, width=8):
        return f"{'n/a':>{width}}" if x is None else f"{x:+{width}.4f}"

    k, w = str(result["primary_horizon"]), print
    for r in result["runs"]:
        w(f"{r['run']}: {r['model']} on {r['task']} (--det {r['det']}); {len(r['gates_placed'])} gates placed"
          f"{', none dropped' if not r['gates_dropped'] else ', dropped: ' + str(r['gates_dropped'])}; "
          f"{r['gates_with_trial']} with a trial; note: {'yes' if r['note'] else 'none'}")
    w(f"\nCoverage: legal actions with a score (primary horizon: {k} updates; a gate counts where all are scored)")
    for c in result["coverage"]:
        gaps = "".join(f"; {kind}: {', '.join(c[kind])}" for kind in ("missing", "failed") if c[kind])
        if c["horizon"] == result["primary_horizon"] or gaps:
            w(f"  {c['run']} gate {c['gate']} (update {c['update']}) at {c['horizon']}: {c['scored']}/{c['legal']}{gaps}")
    drift = [abs(g["controls"][h]["incumbent_repeat"]) for g in result["gates"] for h in g["controls"]
             if g["controls"][h]["incumbent_repeat"] is not None]
    w(f"Incumbent repeat: largest |loss difference| {max(drift):.2g} over {len(drift)} scores (0 = the trial "
      "repeats exactly)" if drift else "Incumbent repeat: not scored in these runs")
    w(f"\nLoss difference on the private rows against keep after {k} updates (negative: better than keep)")
    w(f"  {'proposer':22s}{'run':24s}{'gates':>6s}{'mean_d':>9s}{'changes':>8s}{'keeps':>6s}{'failed':>7s}{'pctile':>8s}")
    at = result["by_horizon"].get(k, dict(proposers={}, paired={}))
    for p, table in at["proposers"].items():
        for run, v in table["by_run"].items():
            w(f"  {p:22s}{run[:23]:24s}{v['gates']:6d} {number(v['mean_d'])}{v.get('changes', ''):>8}"
              f"{v.get('keeps', ''):>6}{v.get('failures', ''):>7}"
              + (f"{v['mean_percentile']:8.2f}" if v.get("mean_percentile") is not None else f"{'':8s}"))
    for name, pair in at["paired"].items():
        w(f"\nPaired, same gates: {name} (negative favors the first)")
        for run, v in pair["by_run"].items():
            w(f"  {run}: {v['gates']} gates, {v['disagreements']} with different choices, mean {number(v['mean_difference']).strip()}")
            for x in v["per_gate"]:
                w(f"    gate {x['gate']}: {x['first']} | {x['second']} | {number(x['difference']).strip()}")
        w("  across horizons (mean per run): " + "; ".join(
            f"{h}: " + ", ".join(number(v["mean_difference"]).strip() for v in result["by_horizon"][h]["paired"][name]["by_run"].values())
            for h in result["by_horizon"]))
    others = [h for h in result["by_horizon"] if h != k]
    if others:
        w(f"\nDoes the menu keep its order? Rank correlation of the actions' loss differences with those at {k} "
          "updates, per gate (n/a: undefined)")
        for h in others:
            w(f"  {h}: " + " ".join(number(x["spearman"], 6).strip() for x in result["by_horizon"][h][
                "menu_rank_correlation_with_primary"]))
    for p, v in result["proposers"].items():
        if v.get("calls") is not None:
            w(f"\n{p}: {v['calls']} calls, {v['failures']} failed" + (f", mean {v['seconds_mean']:.2f} s" if v.get(
                "seconds_mean") is not None else "") + (f", ${v['cost_usd']:.4f} of ${v['budget_usd']}" if "cost_usd" in v else ""))
    e = result.get("example")
    if e:
        w(f"\nExample, chosen before any score was read: {e['run']} gate {e['gate']} (update {e['update']})")
        note = result["notes"].get(e["task"])
        if note:
            w("  note" + (f" ({note['label']})" if note["label"] else "") + ": " + (
                note["text"] if note["public"] else f"private, not shown (sha256 {note['sha256'][:12]})"))
        for p, v in e["picks"].items():
            s = v["at"].get(k)
            w(f"  {p:20s} {v['choice']}{'' if v['ok'] else ' (call failed: ' + str(v['failure']) + ')'}: "
              + (f"{s['mean_d']:+.4f} (SE {s['se']:.4f})" if s else "not scored"))
        if e.get("tie") is not None:
            w("  " + ("the note changed nothing at this gate: the same choice, a difference of 0" if e["tie"]
                      else "the note changed the choice at this gate"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run directories of finetune.py made with --outer-audit")
    p.add_argument("--out", type=Path, required=True, help="new directory for the result")
    p.add_argument("--horizon", type=int, help="primary horizon in updates (default: the runs' --outer-steps)")
    p.add_argument("--example-gate", type=int, default=1,
                   help="gate of the first run to show in full; fixed here, before any score is read (default 1)")
    p.add_argument("--timeout", type=float, default=120., help="seconds per call")
    p.add_argument("--seed", type=int, default=0, help="sampling seed sent to the local server")
    g = p.add_argument_group("local model (OpenAI-compatible server on this machine)")
    g.add_argument("--local-model", help="name the server serves the model under")
    g.add_argument("--local-revision", help="revision (commit) of the served weights")
    g.add_argument("--local-url", default="http://127.0.0.1:8000/v1")
    g.add_argument("--local-request", type=json.loads, metavar="JSON", help="extra fields of the request body")
    g.add_argument("--note", type=Path, nargs="+", default=[],
                   help="note files; each is used only for runs of the task its header names")
    g = p.add_argument_group("hosted comparison (optional, costs money)")
    g.add_argument("--hosted-model", help="Claude model that answers on the informed brief")
    g.add_argument("--hosted-effort", default="low")
    g.add_argument("--hosted-budget", type=float, help="USD cap over all gates and runs of this replay")
    g.add_argument("--ledger", type=Path, help="spend file shared with other runs; the cap then covers its sum")
    g.add_argument("--exportable", action="store_true",
                   help="confirm that the informed briefs of these runs (names and measurements) may be sent to "
                        "the hosted model; masking names does not make private measurements exportable")
    args = p.parse_args(argv)
    if args.local_model and not args.local_revision or args.note and not args.local_model \
            or not isinstance(args.local_request or {}, dict):
        p.error("--local-model needs --local-revision, --note needs --local-model, --local-request a JSON object")
    if args.hosted_model and not ((args.hosted_budget or 0) > 0 and args.exportable):
        p.error("--hosted-model needs --hosted-budget and --exportable")
    runs = [checkpoints(path) for path in args.runs]
    if len({r["name"] for r in runs}) < len(runs):
        p.error("two run directories have the same name")
    notes = {}
    for path in args.note:
        note = expert.load_note(path)
        if note["task"] in notes or note["task"] not in {r["task"] for r in runs}:
            p.error(f"{path} is a note for task {note['task']!r}: a second note for it, or no run of that task")
        notes[note["task"]] = note
    roles = ["masked", "informed"] * bool(args.local_model) or ["informed"] * bool(args.hosted_model)
    lost = [str(r["path"] / "expert" / f"gate{g['gate'] + 1}_{role}_brief.txt") for r in runs for g in r["gates"]
            for role in roles if not (r["path"] / "expert" / f"gate{g['gate'] + 1}_{role}_brief.txt").exists()]
    if lost:
        p.error(f"no saved brief {lost[0]} ({len(lost)} missing): make the audit with --bio-policy expert "
                "--expert-script keep.json --expert-shadow masked, which saves both briefs at every gate")
    if not 1 <= args.example_gate <= len(runs[0]["gates"]):
        p.error(f"--example-gate: {runs[0]['name']} has gates 1 to {len(runs[0]['gates'])}")
    args.out.mkdir(parents=True, exist_ok=False)
    local = hosted = None
    if args.local_model:
        local = expert.Local(args.local_url, args.local_model, args.local_revision, args.timeout,
                             args.out / "local", args.seed, args.local_request)
    if args.hosted_model:   # one client for the whole replay: its budget is the cap over every gate of every run
        hosted = expert.Claude(args.hosted_model, args.hosted_effort, args.hosted_budget, args.timeout,
                               args.ledger, args.out / "local")
    rows = []
    with (args.out / "proposals.jsonl").open("w") as f:
        for run in runs:
            for gate in run["gates"]:
                for row in answers(run, gate, local, hosted, notes.get(run["task"]), args.out):
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())                            # every answer is on disk before a private score is read
    recorded = {run["name"]: trials(run["path"]) for run in runs}
    result = join(runs, rows, recorded, args.horizon)
    about = {}
    for proposer in dict.fromkeys(r["proposer"] for r in rows):
        mine = [r for r in rows if r["proposer"] == proposer]
        about[proposer] = dict(failures=sum(not r["ok"] for r in mine), failure_kinds=sorted(
            {r["failure"] for r in mine if not r["ok"]}))
        if proposer.startswith(("local:", "hosted:")):
            about[proposer].update(calls=len(mine), seconds_mean=mean([r["seconds"] for r in mine]),
                                   seconds_max=max(r["seconds"] for r in mine))
        if proposer.startswith("local:"):
            about[proposer].update({k: v for k, v in local.record().items() if k not in ("calls", "latency_seconds")})
        if proposer.startswith("hosted:"):
            about[proposer].update(backend="claude", model=args.hosted_model, effort=args.hosted_effort,
                                   budget_usd=args.hosted_budget, cost_usd=hosted.cost, stopped=hosted.off)
    first = next(g for g in result["gates"] if g["run"] == runs[0]["name"] and g["gate"] == args.example_gate)
    with_, without = first["picks"].get("local:informed+note"), first["picks"].get("local:informed")
    result = dict(
        tool="biopal.replay", created=time.strftime("%Y-%m-%d %H:%M:%S"),
        runs=[dict(run=r["name"], path=str(r["path"]), task=r["task"], model=r["model"], det=r["det"],
                   gates_placed=r["schedule"]["gates"], gates_dropped=r["schedule"]["dropped_gates"],
                   gates_with_trial=len(recorded[r["name"]]), note=expert.note_record(notes.get(r["task"])) and
                   notes[r["task"]]["sha256"]) for r in runs],
        notes={task: expert.note_record(note) for task, note in notes.items()}, proposers=about, **result,
        example=dict(first, tie=with_["choice"] == without["choice"] if with_ and without else None))
    (args.out / "result.json").write_text(json.dumps(result, indent=1))
    show(result)
    print(f"\nresult: {args.out / 'result.json'}; briefs and raw replies (keep on this machine): {args.out / 'local'}")
    return result


if __name__ == "__main__":
    main()
