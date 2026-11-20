#!/usr/bin/env python3
"""Compare the test results of finished runs.

  python compare_test.py runs/donors_*
  python compare_test.py runs/donors_plain runs/donors_hpo runs/donors_biopal_local

Each argument is an output folder of finetune.py or of hpo.py. One row per run: what trained it, on how many
rows, what it spent on Claude, and the validation and test metrics of the checkpoint that validation selected.
The best value of a test column is starred. The first run named is the reference for the differences below the
table. After that: runs that ended with identical weights, what the agents did, rows that were held out for a
part of the run, where the runs are not like for like, and runs without a test score. An HPO study has none until its winner is scored once with the
command hpo.py printed; this script never trains or scores anything.
"""
import argparse
import json
import math
from pathlib import Path

LOWER = {"loss", "rmse", "mae"}                     # less is better; for every other metric more is
SAME = ("task", "model", "objective", "seed", "epochs", "train split", "test rows", "data")


def count(n, word):
    return f"{n} {word}" + "s" * (n != 1)


def events(run):
    f = run / "opal" / "events.jsonl"
    return [json.loads(line) for line in f.open()] if f.exists() else []


def fill(row, r, run):
    """The fields of one finetune.py result."""
    rec, bio, ev = r["recipe"], r.get("biopal"), events(run)
    asked, applied = (sum(e.get("kind") == k for e in ev) for k in ("requested", "applied"))
    end = next((e for e in ev if e.get("kind") == "end"), {})
    fit, back = (rec.get("panel") or {}).get("F", {}).get("n"), rec.get("panel_release")
    if fit and back:
        row["did"].append(f"{r['split_sizes']['train'] - fit} of its train rows were held out for updates 1 to "
                          f"{back} of {r['steps']}, then trained on")
    row.update(val=r.get("best_validation"), test=r.get("test"), sha=r.get("final_sha256"), usd=None,
               train=r["split_sizes"]["train"] if back or not fit else fit, of=r["split_sizes"]["train"],
               same=dict(zip(SAME, (rec["task"], rec["model"], rec["objective"], rec["seed"], rec["epochs"],
                                    r["split_sizes"]["train"], r["split_sizes"].get("test"),
                                    str(r.get("data_sha256"))[:12]))))
    if r["mode"] == "plain":
        row["method"] = {"adamw": "AdamW", "sgd": "SGD"}.get(rec["optimizer"], rec["optimizer"])
    elif r["mode"] == "keep" and not bio:
        row["method"] = "OPAL without an agent"
    else:
        row["usd"] = bio["opal_cost_usd"] + bio["expert_cost_usd"] if bio else end.get("cost_usd", 0.)
        expert = (bio or {}).get("expert")
        row["method"] = "OPAL" if not expert else "BiOPAL, " + (
            "scripted expert" if expert["backend"] == "scripted" else
            f"{expert['model']} (local)" if expert["backend"] == "local" else
            f"{expert['model']} ({expert.get('effort')})")
        if r["mode"] != "keep":
            row["did"].append(f"OPAL's agent was asked {asked} times and changed the update rule {applied} times")
        if expert:
            took = [f"{a['action']} {a['target']} at update {a['update']}" for a in bio["adopted"]]
            row["did"].append(f"the expert proposed at {count(len(bio['gates']), 'gate')}; adopted: "
                              + ("; ".join(took) if took else "nothing"))
    if r.get("no_test"):
        row["missing"] = "run with --no-test"


def load(path):
    """One row for a run folder (finetune.py) or a study folder (hpo.py)."""
    run = Path(path)
    row = dict(run=run.name, method="", train=None, of=None, usd=None, val=None, test=None, sha=None, same=None,
               did=[], missing=None)
    if (run / "hpo_result.json").exists():
        h = json.loads((run / "hpo_result.json").read_text())
        done, best = [t for t in h["trials"] if t["state"] == "COMPLETE"], h.get("best")
        if not best:
            return dict(row, method=f"HPO ({h['mode']})", missing="no finished trial")
        scored = (run / "final" / "result.json").exists()
        source = run / "final" if scored else run / "trials" / f"trial_{best['number']:03d}"
        fill(row, json.loads((source / "result.json").read_text()), source)
        p = best["params"]
        row.update(method=f"HPO ({h['mode']}, {count(len(done), 'trial')})", did=[
            f"best of {count(len(done), 'trial')} on validation: trial {best['number']}, lr {p['lr']:.2e}, "
            f"weight decay {p['weight_decay']:.3f}, warmup {p['warmup_frac']:.3f}"])
        if not scored:
            row.update(test=None, sha=None, missing="its winner is not scored on test yet; from finetune/ run\n      "
                       + h["final_command"])
    elif (run / "result.json").exists():
        fill(row, json.loads((run / "result.json").read_text()), run)
    else:
        row["missing"] = "no result.json: " + ("the run did not finish" if run.is_dir() else "no such folder")
    return row


def table(head, lines):
    width = [max(len(line[i]) for line in [head] + lines) for i in range(len(head))]
    for line in [head] + lines:
        cells = (c.ljust(w) if i < 2 else c.rjust(w) for i, (c, w) in enumerate(zip(line, width)))
        print(("  " + "  ".join(cells)).rstrip())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="output folders of finetune.py or hpo.py; the first is the reference")
    rows = [load(path) for path in dict.fromkeys(p.parse_args().runs)]         # a folder named twice counts once
    tested = [r for r in rows if r["test"]]
    metrics = list(tested[0]["test"]) if tested else []
    best = {m: (min if m in LOWER else max)(r["test"][m] for r in tested if m in r["test"]) for m in metrics}

    def cell(r, m):
        if not r["test"] or m not in r["test"]:
            return "-"
        return f"{r['test'][m]:.4f}" + ("*" if r["test"][m] == best[m] else " ")

    print("Validation and test metrics of the checkpoint each run selected on validation (* best on test)\n")
    table(["run", "trained by", "train rows", "Claude $", "val loss"] + ["test " + m for m in metrics],
          [[r["run"], r["method"] or "-", str(r["train"] or "-"), "-" if r["usd"] is None else f"{r['usd']:.3f}",
            f"{r['val']['loss']:.4f}" if r["val"] else "-"] + [cell(r, m) for m in metrics] for r in rows])

    if len(tested) > 1:
        ref = tested[0]
        print(f"\nTest, difference to {ref['run']}\n")
        table(["run", ""] + metrics, [[r["run"], ""] + [f"{r['test'][m] - ref['test'][m]:+.4f}" if m in r["test"]
                                                         else "-" for m in metrics] for r in tested[1:]])
        if "accuracy" in ref["test"] and ref["same"]["test rows"]:
            a, n = ref["test"]["accuracy"], ref["same"]["test rows"]
            print(f"\n  {n} test rows: their sampling alone puts a standard error of about "
                  f"{math.sqrt(a * (1 - a) / n):.4f} on an accuracy of {a:.3f}.\n  Each run is one seed.")

    notes = []
    by_sha = {}
    for r in rows:
        if r["sha"]:
            by_sha.setdefault(r["sha"], []).append(r["run"])
    notes += [f"identical weights at the end of training: {' = '.join(names)}"
              for names in by_sha.values() if len(names) > 1]
    notes += [f"{r['run']}: {text}" for r in rows for text in r["did"]]
    known = [r for r in rows if r["same"]]
    for field in SAME:
        values = {}
        for r in known:
            values.setdefault(r["same"][field], []).append(r["run"])
        if len(values) > 1:
            notes.append(f"NOT like for like, {field} differs: "
                         + "; ".join(f"{v} in {', '.join(names)}" for v, names in values.items()))
    part = [f"{r['run']} ({r['train']} of {r['of']})" for r in known if r["train"] != r["of"]]
    if part and len(part) < len(known):
        notes.append("trained on a part of train (the rest is held out, e.g. for BiOPAL's gates): " + ", ".join(part))
    notes += [f"{r['run']}: {r['missing']}" for r in rows if r["missing"]]
    if notes:
        print("\nNotes\n" + "\n".join("  - " + n for n in notes))


if __name__ == "__main__":
    main()
