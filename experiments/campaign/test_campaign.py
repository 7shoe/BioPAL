"""Failure paths of campaign.py with fake commands in a temporary folder: no GPU, no Claude call, about 10 seconds.
Run it after any change to campaign.py:  python3 experiments/campaign/test_campaign.py"""
import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as c  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="runner_test_"))
c.HERE, c.STATUS, c.OUT, c.LOGS = tmp, tmp / "status", tmp / "out", tmp / "out" / "_logs"
c.MIN_LIMIT_MIN = 0.05                      # 3 seconds
c.CAPS = dict(pilot=20., seed42=3., seeds43_44=60., replays=5., agent_ablation=20.)


def row(id, stage="2", name="plain", paid="", needs="", soft="", seed=42, minutes=0.01, kind="train"):
    return dict(id=id, stage=stage, pair=id.split("/")[1], name=name, seed=seed, setting="default", kind=kind,
                env="x", need_gb=0, minutes=minutes, paid=paid, runs=1, needs=needs, soft=soft)


rows = [
    row("main/h3k27ac/plain_s42"),                                   # works
    row("main/mrl/plain_s42"),                                       # fails twice
    row("main/mrl/hpo_s42", name="hpo", needs="main/mrl/plain_s42"),   # needs a failed run -> skipped
    row("main/gfp/plain_s42"),                                       # fails once, then works
    row("main/melt/plain_s42"),                                      # hangs -> stopped at the time limit, twice
    row("main/ctcf/opal_s42", name="opal", paid="agent"),            # paid, cap 2: fits under the line of 3
    row("main/ctcf/biopal_s42", name="biopal", paid="agent+expert"),   # paid, cap 3: does not fit while the first runs
    row("main/mrl/plain_s43", stage="3", seed=43, soft="main/mrl/plain_s42"),   # arm failed twice at seed 42 -> skipped
    row("main/h3k27ac/plain_s43", stage="3", seed=43, soft="main/h3k27ac/plain_s42"),   # runs
]
with (tmp / "runs.tsv").open("w", newline="") as f:
    w = csv.DictWriter(f, c.FIELDS, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
(tmp / "status").mkdir()
(tmp / "status" / "readiness.json").write_text(json.dumps(dict(claude=True)))


def build(r, dry=False):
    out = c.outdir(r["id"])
    ok = f"import os, json; os.makedirs({str(out)!r}); open({str(out / 'best.pt')!r}, 'w').write('w'); json.dump(dict(x=1), open({str(out / 'result.json')!r}, 'w'))"
    half = f"import os, sys; os.makedirs({str(out)!r}); open({str(out / 'best.pt')!r}, 'w').write('w'); sys.exit(3)"
    if r["id"] == "main/mrl/plain_s42":
        code = half
    elif r["id"] == "main/gfp/plain_s42":
        code = half if not Path(str(out) + ".failed1").exists() else ok
    elif r["id"] == "main/melt/plain_s42":
        code = "import time; time.sleep(600)"
    elif r["paid"]:
        code = "import time; time.sleep(1.5); " + ok
    else:
        code = ok
    return [sys.executable, "-c", code], tmp, {}


c.build = build
t = time.time()
c.run(argparse.Namespace(stage=["2", "3"], catchup=False, dry_run=False, only=None, deadline=None, poll=1.))
att = c.attempts()
state = {r["id"]: c.state_of(r, att, []) for r in c.read_manifest()}
reasons = c.load_reasons()
print("\n".join(f"{k:26s} {v:8s} {reasons.get(k, '')}" for k, v in state.items()))
expect = {"main/h3k27ac/plain_s42": "done", "main/mrl/plain_s42": "failed", "main/mrl/hpo_s42": "pending",
          "main/gfp/plain_s42": "done", "main/melt/plain_s42": "failed", "main/ctcf/opal_s42": "done",
          "main/ctcf/biopal_s42": "done", "main/mrl/plain_s43": "pending", "main/h3k27ac/plain_s43": "done"}
assert state == expect, {k: (state[k], expect[k]) for k in state if state[k] != expect[k]}
assert reasons["main/mrl/hpo_s42"].startswith("skipped") and reasons["main/mrl/plain_s43"].startswith("skipped")
assert len(att["main/mrl/plain_s42"]) == 2 and len(att["main/gfp/plain_s42"]) == 2 and len(att["main/melt/plain_s42"]) == 2
assert "time limit" in att["main/melt/plain_s42"][0]["why"]
assert (c.OUT / "main/mrl/plain_s42.failed1").exists() and (c.OUT / "main/mrl/plain_s42.failed2").exists()
assert not list(c.OUT.rglob("*.failed*/best.pt")), "a failed attempt kept its checkpoint"
assert (c.OUT / "main/h3k27ac/plain_s42/best.pt").exists(), "a table run lost its checkpoint"
a, b = att["main/ctcf/opal_s42"][0], att["main/ctcf/biopal_s42"][0]
assert b["start"] >= a["start"] + a["seconds"] - 1, "the second paid run started while the first held the budget"
print(f"\nall failure paths behave as the protocol says ({time.time() - t:.0f} s); folder {tmp}")
