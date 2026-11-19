#!/usr/bin/env python3
"""Find input sequences that are duplicated, exactly or nearly, between the splits of a task.

For every task in the task registry (data/processed/tasks.yaml):

    val rows    that duplicate a test sequence          are listed for removal
    train rows  that duplicate a val or test sequence   are listed for removal

Test is never touched and the curated parquet files are never modified. The script
writes two files next to itself:
    removed.tsv  task, split, series_id, via - the rows finetune/finetune.py leaves out
    report.tsv   counts per task

Near-duplicate: a window covering half of one sequence (its start, middle or end;
at least 16 nucleotides or 8 residues) occurs letter for letter in the other sequence.

Tasks whose sequences resemble each other by construction (EXEMPT below) get the
exact check only; a near-duplicate filter would delete them.

Cost is linear in the number of characters (times the number of distinct sequence
lengths in a task): windows are compared through a rolling hash and every hash
match is confirmed on the letters.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPLITS = ("train", "val", "test")

# task name or prefix -> why near-duplicates belong to the task
EXEMPT = {
    "proteingym_": "mutational scan of one protein",
    "cds_beta_lactamase_": "mutational scan of one protein",
    "cds_fluorescence": "mutational scan of one protein",
    "cds_stability": "designed proteins and their point mutants",
    "rfam_": "RNA family membership",
    "rna_dst_ncrna_family_": "RNA family membership",
    "cripsrCas9": "guide/off-target pairs a few mismatches apart; split by Hamming clusters at curation",
    "sirna": "19-21 nt oligos tiled along their targets; holdout pruned by Hamming distance at curation",
}

MIN_WINDOW = {"dna": 16, "rna": 16, "protein": 8}
SKIP = {"dna": "N", "rna": "N", "protein": "X"}       # windows holding these letters are not compared
CHUNK = 4_000_000                                      # characters hashed at a time
MULT = np.uint64(0x9E3779B97F4A7C15)                   # odd, so it has an inverse mod 2**64
INV = np.uint64(pow(int(MULT), -1, 2 ** 64))
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
_powers = {"n": 0}


def exempt_reason(task: str) -> str | None:
    hits = [k for k in EXEMPT if task == k or (k.endswith("_") and task.startswith(k))]
    return EXEMPT[max(hits, key=len)] if hits else None


def powers(n: int):
    """MULT**i and MULT**-i (mod 2**64) for i < n"""
    if _powers["n"] < n:
        one = np.ones(1, np.uint64)
        _powers.update(n=n, up=np.cumprod(np.concatenate([one, np.full(n - 1, MULT)])),
                       down=np.cumprod(np.concatenate([one, np.full(n - 1, INV)])))
    return _powers["up"], _powers["down"]


class Text:
    """Sequences laid end to end, a 0 byte after each, with prefix sums for window hashes."""

    def __init__(self, seqs: list[str], skip: str):
        self.raw = ("\0".join(seqs) + "\0").encode("ascii")
        arr = np.frombuffer(self.raw, np.uint8)
        self.lens = np.fromiter(map(len, seqs), np.int64, len(seqs))
        self.starts = np.concatenate([[0], np.cumsum(self.lens + 1)[:-1]])
        up, self.down = powers(arr.size + 1)
        self.hash = np.zeros(arr.size + 1, np.uint64)
        np.cumsum((arr.astype(np.uint64) + np.uint64(1)) * up[:arr.size], out=self.hash[1:])
        self.bad = np.zeros(arr.size + 1, np.int64)
        np.cumsum(np.isin(arr, np.frombuffer(("\0" + skip).encode(), np.uint8)), out=self.bad[1:])

    def windows(self, k: int):
        """Hash of every window of k characters, and whether it lies inside one sequence."""
        h = (self.hash[k:] - self.hash[:-k]) * self.down[:len(self.raw) - k + 1]
        return h, self.bad[k:] == self.bad[:-k]


def chunks(seqs: list[str]):
    start = size = 0
    for i, s in enumerate(seqs):
        size += len(s) + 1
        if size >= CHUNK:
            yield start, seqs[start:i + 1]
            start, size = i + 1, 0
    if start < len(seqs):
        yield start, seqs[start:]


def scan(queries: list[str], refs: list[str], kmin: int, skip: str):
    """Look for the start, middle and end window of every query inside every ref.

    Returns (q_hit, r_hit): queries with a window found in a ref, refs holding a window of a query.
    """
    q_hit, r_hit = np.zeros(len(queries), bool), np.zeros(len(refs), bool)
    table: dict[int, dict] = {}                        # k -> {hash: {window: [query rows]}}
    for base, part in chunks(queries):
        text = Text(part, skip)
        ks = np.minimum(np.maximum(text.lens // 2, kmin), text.lens)
        for k in map(int, np.unique(ks[ks > 0])):
            h, ok = text.windows(k)
            slot = table.setdefault(k, {})
            for i in np.flatnonzero(ks == k).tolist():
                n = int(text.lens[i])
                for off in {0, (n - k) // 2, n - k}:
                    pos = int(text.starts[i]) + off
                    if ok[pos]:
                        slot.setdefault(int(h[pos]), {}).setdefault(
                            text.raw[pos:pos + k], []).append(base + i)
    wanted = {k: np.sort(np.fromiter(slot, np.uint64, len(slot))) for k, slot in table.items() if slot}
    for base, part in chunks(refs):
        text = Text(part, skip)
        for k, keys in wanted.items():
            if len(text.raw) < k:
                continue
            h, ok = text.windows(k)
            at = np.searchsorted(keys, h)
            at[at == keys.size] = 0
            pos = np.flatnonzero((keys[at] == h) & ok)
            owner = np.searchsorted(text.starts, pos, side="right") - 1
            for p, j in zip(pos.tolist(), owner.tolist()):
                rows = table[k][int(h[p])].get(text.raw[p:p + k])      # the letters, not only the hash
                if rows is not None:
                    r_hit[base + j] = True
                    q_hit[rows] = True
    return q_hit, r_hit


def near(rows: list[str], held: list[str], kmin: int, skip: str) -> np.ndarray:
    """Which of `rows` nearly duplicate a sequence in `held`."""
    if not rows or not held:
        return np.zeros(len(rows), bool)
    _, holds_window = scan(held, rows, kmin, skip)
    window_found, _ = scan(rows, held, kmin, skip)
    return holds_window | window_found


def audit(task: str, meta: dict, base: Path) -> tuple[pd.DataFrame, dict]:
    t0 = time.time()
    modality = meta["modality"]
    reason = exempt_reason(task)
    df = pd.concat([pd.read_parquet(base / task / f"task_{k}.parquet",
                                    columns=["series_id", "sequence", "split"])
                    for k in meta["tasks"]]).drop_duplicates(ignore_index=True)
    if df["series_id"].duplicated().any():
        raise SystemExit(f"{task}: the task files disagree on series_id, sequence or split")
    df["norm"] = df["sequence"].str.strip().str.upper()
    if modality in ("dna", "rna"):
        df["norm"] = df["norm"].str.replace("U", "T", regex=False)
    by = {s: sorted(set(df.loc[df["split"] == s, "norm"])) for s in SPLITS}

    def find(split: str, held: list[str]) -> dict[str, str]:
        """sequence -> how it matches a held-out sequence"""
        rows = by[split]
        found = dict.fromkeys(set(rows) & set(held), "exact")
        if reason is None:
            views = [("window", held)]
            if modality == "dna":
                views.append(("revcomp", [s.translate(COMPLEMENT)[::-1] for s in held]))
            for via, other in views:
                hit = near(rows, other, MIN_WINDOW[modality], SKIP[modality])
                for i in np.flatnonzero(hit).tolist():
                    found.setdefault(rows[i], via)
        return found

    drop = {"val": find("val", by["test"])}
    drop["train"] = find("train", [s for s in by["val"] if s not in drop["val"]] + by["test"])
    removed = pd.concat([df[(df["split"] == s) & df["norm"].isin(d)].assign(via=lambda x, d=d: x["norm"].map(d))
                         for s, d in drop.items()])
    removed = removed.assign(task=task)[["task", "split", "series_id", "via"]]

    n = df["split"].value_counts()
    gone = removed["split"].value_counts()
    via = removed["via"].value_counts()
    row = dict(task=task, source=meta["source"], modality=modality,
               filter="exact only" if reason else "near-duplicates", exempt_because=reason or "",
               train=int(n.get("train", 0)), val=int(n.get("val", 0)), test=int(n.get("test", 0)),
               duplicates_within_split=int(df.duplicated(["split", "norm"]).sum()),
               removed_train=int(gone.get("train", 0)), removed_val=int(gone.get("val", 0)),
               removed_train_pct=round(100 * gone.get("train", 0) / max(1, n.get("train", 0)), 2),
               removed_val_pct=round(100 * gone.get("val", 0) / max(1, n.get("val", 0)), 2),
               via_exact=int(via.get("exact", 0)), via_window=int(via.get("window", 0)),
               via_revcomp=int(via.get("revcomp", 0)), seconds=round(time.time() - t0, 1))
    return removed, row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", type=Path, default=REPO / "paths" / "paths.yaml")
    ap.add_argument("--machine", default=platform.node().split(".")[0])
    ap.add_argument("--registry", type=Path, default=REPO / "data" / "processed" / "tasks.yaml")
    ap.add_argument("--tasks", nargs="*", help="subset of registry tasks (prints only, writes nothing)")
    ap.add_argument("--out", type=Path, default=HERE, help="directory for removed.tsv and report.tsv")
    args = ap.parse_args()

    machine = yaml.safe_load(args.paths.read_text())["machines"].get(args.machine) or {}
    if not machine.get("curated_base"):
        ap.error(f"No curated_base for machine {args.machine!r} in {args.paths}; use --machine")
    base = Path(machine["curated_base"])
    registry = yaml.safe_load(args.registry.read_text())["tasks"]
    unknown = set(args.tasks or []) - set(registry)
    if unknown:
        ap.error(f"not in the registry: {sorted(unknown)}")
    print(f"machine={args.machine} curated_base={base}\n")

    removed, rows = [], []
    for task, meta in registry.items():
        if args.tasks and task not in args.tasks:
            continue
        gone, row = audit(task, meta, base)
        removed.append(gone)
        rows.append(row)
        what = (f"exempt ({row['exempt_because']}): exact check only, {row['via_exact']} shared"
                if row["exempt_because"] else
                f"remove train {row['removed_train']:,} ({row['removed_train_pct']}%), "
                f"val {row['removed_val']:,} ({row['removed_val_pct']}%)")
        dup = f"  !! {row['duplicates_within_split']} duplicates inside a split" \
            if row["duplicates_within_split"] else ""
        print(f"== {task} [{row['modality']}] train={row['train']:,} val={row['val']:,} "
              f"test={row['test']:,}\n    {what}{dup}  ({row['seconds']} s)", flush=True)

    report = pd.DataFrame(rows)
    removed = pd.concat(removed).sort_values(["task", "split", "series_id"], ignore_index=True)
    filtered = report[report["exempt_because"] == ""]
    print(f"\nSUMMARY  {len(report)} tasks, {len(filtered)} filtered for near-duplicates, "
          f"{len(report) - len(filtered)} exempt; {int(report['train'].sum() + report['val'].sum() + report['test'].sum()):,} "
          f"sequences in {report['seconds'].sum():.0f} s")
    print(f"  rows to remove: train {int(report['removed_train'].sum()):,}, val {int(report['removed_val'].sum()):,} "
          f"(exact {int(report['via_exact'].sum()):,}, window {int(report['via_window'].sum()):,}, "
          f"reverse complement only {int(report['via_revcomp'].sum()):,})")
    if args.tasks:
        print("\n[info] partial run (--tasks): not writing removed.tsv / report.tsv")
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        removed.to_csv(args.out / "removed.tsv", sep="\t", index=False)
        report.to_csv(args.out / "report.tsv", sep="\t", index=False)
        print(f"\n[ok] wrote {args.out / 'removed.tsv'} ({len(removed):,} rows) and {args.out / 'report.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
