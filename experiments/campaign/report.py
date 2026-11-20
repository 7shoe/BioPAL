#!/usr/bin/env python3
"""Stage 6 of the campaign: tables, plots, spend and the list of missing runs. Trains nothing and may be run at
any time; it reports whatever has a result. Writes status/results.tsv, status/report.md and, with matplotlib,
status/plots/<pair>_s<seed>.png. How results are read is fixed in README.md, section 9."""
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as c   # noqa: E402

ARMS = ["plain", "matched", "opal", "biopal", "biopal_qwen", "hpo"]
COMPARE = [("matched", "plain", "cost of the held-out rows"), ("opal", "matched", "OPAL's agent"),
           ("biopal", "opal", "the expert"), ("biopal_qwen", "biopal", "local against hosted expert"),
           ("hpo", "plain", "HPO"), ("biopal", "hpo", "BiOPAL against HPO")]


def lines_of(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def read(row):
    """What the tables need from one finished training run."""
    out, objective = c.outdir(row["id"]), c.PAIRS[row["pair"]][3]
    r = json.loads((out / "result.json").read_text())
    primary = "mcc" if objective == "classification" else "spearman"
    test, bio, recipe = r.get("test") or {}, r.get("biopal") or {}, r.get("recipe", {})
    val = {v["epoch"]: v["step"] for v in lines_of(out / "validation.jsonl")}
    release = recipe.get("panel_release")
    events = lines_of(out / "opal" / "events.jsonl")
    decisions = [e for e in lines_of(out / "outer.jsonl") if e.get("event") == "decision"]
    return dict(
        id=row["id"], stage=row["stage"], pair=row["pair"], name=row["name"], seed=row["seed"], setting=row["setting"],
        metric=primary, value=test.get(primary), test_loss=test.get("loss"),
        other="; ".join(f"{k} {v:.4f}" for k, v in test.items() if k not in (primary, "loss") and isinstance(v, float)),
        best_epoch=r.get("best_epoch"),
        picked_before_return=bool(release and val.get(r.get("best_epoch"), 10 ** 9) < release),
        weights=r.get("final_sha256", "")[:12],
        opal_decisions=len(bio.get("opal_decisions", [])) if bio else "",
        rule_changes=sum(e.get("kind") == "applied" for e in events) if events else "",
        gates=len(bio.get("gates", [])) if bio else "",
        proposals=sum(d.get("action") not in (None, "keep") for d in decisions) if decisions else "",
        adopted="; ".join(f"{a.get('action')} {a.get('target', '')}".strip() for a in bio.get("adopted", [])),
        n_adopted=len(bio.get("adopted", [])) if bio else "",
        usd=round(c.ledger_sum(c.ledger_of(row["id"])), 4) or "",
        minutes=round(r.get("training_and_validation_wall_seconds", 0) / 60, 1))


def fmt(x, digits=4):
    return "" if x is None or x == "" else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def plots(got, md):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        md.append("No plots: this Python has no matplotlib (set CAMPAIGN_PYTHON to one that has it).")
        return
    folder = c.STATUS / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    made = []
    for pair in c.PAIRS:
        for seed in (42, 43, 44):
            runs = {a: got.get(("main", pair, a, seed)) for a in ARMS}
            if not any(runs.values()):
                continue
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for arm, r in runs.items():
                if r is None:
                    continue
                out = c.outdir(r["id"])
                loss = [x["loss"] for x in lines_of(out / "train.jsonl")]
                w = 50
                smooth = [sum(loss[max(0, i - w + 1):i + 1]) / len(loss[max(0, i - w + 1):i + 1]) for i in range(len(loss))]
                line, = ax.plot(range(1, len(smooth) + 1), smooth, label=arm, linewidth=1.2)
                res = json.loads((out / "result.json").read_text())
                bio = res.get("biopal") or {}
                for u in bio.get("opal_decisions", []):       # OPAL's decisions: ticks at the bottom
                    ax.axvline(u, 0, .04, color=line.get_color(), linewidth=1)
                for a in bio.get("adopted", []):              # adopted proposals of the expert: ticks at the top
                    ax.axvline(a["update"], .94, 1, color=line.get_color(), linewidth=2)
                release = res.get("recipe", {}).get("panel_release")
                if release and arm == "matched":
                    ax.axvline(release, color="gray", linestyle=":", linewidth=1)
            ax.set(xlabel="update", ylabel="training loss (mean of the last 50 updates)", yscale="log",
                   title=f"{pair} ({c.PAIRS[pair][0]}, {c.PAIRS[pair][1]}), seed {seed}")
            ax.legend(fontsize=8)
            fig.text(.01, .01, "dotted: held-out rows return; bottom ticks: OPAL decisions; top ticks: adopted expert proposals",
                     fontsize=7)
            fig.tight_layout()
            fig.savefig(folder / f"{pair}_s{seed}.png", dpi=130)
            plt.close(fig)
            made.append(f"{pair}_s{seed}.png")
    md.append(f"{len(made)} trajectory plots in `status/plots/`." if made else "No finished main runs to plot yet.")


def main():
    rows = c.read_manifest()
    with c.locked():
        att, entries, reasons = c.attempts(), [e for e in c.running_entries() if c.alive(e["pid"])], c.load_reasons()
        summary = c.write_status(rows, att, entries, reasons)
        per = c.spend(att, entries)
    results, got = [], {}
    for r in rows:
        if r["kind"] == "train" and r["name"] != "probe" and r["stage"] != "0" and c.done(r):
            try:
                x = read(r)
            except (OSError, ValueError, KeyError) as e:
                print(f"cannot read {r['id']}: {e}")
                continue
            results.append(x)
            got[(r["id"].split("/")[0], r["pair"], r["name"], r["seed"])] = x
    keys = list(results[0]) if results else []
    (c.STATUS / "results.tsv").write_text("\n".join(["\t".join(keys)] + ["\t".join(fmt(x[k], 6) for k in keys) for x in results]) + "\n")

    md = [f"# BiOPAL campaign report, {time.strftime('%Y-%m-%d %H:%M')}", "",
          "Every number is a test metric of the checkpoint with the lowest validation loss. MCC for classification, "
          "Spearman correlation for regression; higher is better. The agent has no shell in any run (README.md, section 2).",
          "", "## State", "", "```", *summary, "```", ""]

    md += ["## Main comparison (stages 1 to 3)", ""]
    for pair, (task, model, _, objective, _, _) in c.PAIRS.items():
        body = []
        for arm in ARMS:
            v = [got.get(("main", pair, arm, s), {}).get("value") for s in (42, 43, 44)]
            have = [x for x in v if x is not None]
            body.append([arm, *[fmt(x) for x in v], fmt(statistics.mean(have)) if have else "",
                         fmt(statistics.stdev(have)) if len(have) > 1 else ""])
        if any(any(b[1:4]) for b in body):
            md += [f"### {pair}: {task}, {model} ({'MCC' if objective == 'classification' else 'Spearman'})", "",
                   *table(["arm", "seed 42", "seed 43", "seed 44", "mean", "SD"], body), ""]

    md += ["## Paired differences, same pair and seed", "",
           "\"better\" or \"worse\" only when the difference has the same sign on all three seeds; otherwise "
           "\"not distinguishable\". Fewer than three seeds: no verdict.", ""]
    body = []
    for a, b, what in COMPARE:
        for pair in c.PAIRS:
            d = [got[("main", pair, a, s)]["value"] - got[("main", pair, b, s)]["value"] for s in (42, 43, 44)
                 if ("main", pair, a, s) in got and ("main", pair, b, s) in got
                 and None not in (got[("main", pair, a, s)]["value"], got[("main", pair, b, s)]["value"])]
            if not d:
                continue
            verdict = "" if len(d) < 3 else "better" if min(d) > 0 else "worse" if max(d) < 0 else "not distinguishable"
            body.append([f"{a} minus {b}", what, pair, ", ".join(f"{x:+.4f}" for x in d), f"{statistics.mean(d):+.4f}", verdict])
    md += [*table(["difference", "shows", "pair", "per seed", "mean", "verdict"], body), ""] if body else ["No pairs of finished runs yet.", ""]

    md += ["## What the agents did", "",
           "A run whose final weights equal those of its matched run was not changed by any agent.", ""]
    body = []
    for x in results:
        if x["name"] not in ("opal", "biopal", "biopal_qwen"):
            continue
        base = got.get(("main", x["pair"], "matched", x["seed"]))
        same = "" if base is None else "yes" if base["weights"] == x["weights"] else "no"
        body.append([x["id"], x["setting"], f"{x['rule_changes']} of {x['opal_decisions']}", f"{x['proposals']} of {x['gates']}" if x["gates"] != "" else "",
                     x["adopted"], same, fmt(x["value"]), fmt(x["usd"], 3), "yes" if x["picked_before_return"] else ""])
    md += [*table(["run", "Claude setting", "rule changes of decisions", "proposals of gates", "adopted", "equals matched",
                   "metric", "USD", "checkpoint from before the rows returned"], body), ""] if body else ["No finished agent runs yet.", ""]

    md += ["## Expert ablation (stage 4)", ""]
    for r in rows:
        if r["kind"] == "replay" and r["stage"] == "4":
            f = c.result_file(r)
            md.append(f"- `{r['id']}`: " + (f"result in `{f.relative_to(c.REPO)}`" if f.exists() else "no result"))
    md.append("")

    md += ["## Claude spend", "", *table(["budget line", "USD so far", "cap"],
                                       [[g, f"{per[g]:.2f}", f"{cap:g}"] for g, cap in c.CAPS.items()]
                                       + [["total", f"{sum(per.values()):.2f}", f"{c.TOTAL_CAP:g} (limit 150)"]]), "",
           "Counted as in README.md, section 4: ledgers, the full agent cap for every failed attempt, and the "
           "remaining cap of runs that are still going.", ""]

    missing = [r for r in rows if r["stage"] != "0" and not c.done(r)]
    md += [f"## Missing runs: {len(missing)}", ""]
    if missing:
        md += table(["run", "stage", "state", "why"],
                    [[r["id"], r["stage"], c.state_of(r, att, entries),
                      reasons.get(r["id"], "") or (att.get(r["id"], [{}])[-1].get("why", "") if att.get(r["id"]) else "not started")]
                     for r in missing])
    md += ["", "## Plots", ""]
    plots(got, md)
    (c.STATUS / "report.md").write_text("\n".join(md) + "\n")
    print(f"{len(results)} finished training runs in the tables, {len(missing)} runs missing")
    print(f"report: {c.STATUS / 'report.md'}\ntable:  {c.STATUS / 'results.tsv'}")


if __name__ == "__main__":
    main()
