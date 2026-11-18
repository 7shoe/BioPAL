#!/usr/bin/env python3
"""Audit the curated downstream task datasets (the <curated_base>/<dataset>/ dirs).

Checks per dataset:
    contract    task_classification/task_regression.parquet readable, required
                columns present, labels valid (binary y_cls / finite y_reg),
                length column matches the sequences, no duplicate sequences
                or series_ids
    splits      train/val/test all present and nonempty, no unknown split
                values, both classes in every split (classification), and
                NO exact sequence shared between splits (leakage)
    alphabet    sequence characters consistent with the source's modality
                (dna/rna -> nucleotide codes, protein -> amino-acid codes)
    provenance  prep_log.json present; flags curations built from the official
                TEST split only (the May 2026 nt_tasks_* bug class), and
                curations older than the latest raw download of their source
                (stale: raw changed after curation)

Raw sources, modalities and machine roots come from ../raw/paths.yaml
(`curated_base` per machine). Exit codes: 0 all OK, 1 any FAIL, 2 only WARNs.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

REQUIRED = {"classification": {"series_id", "sequence", "length", "split", "y_cls"},
            "regression": {"series_id", "sequence", "length", "split", "y_reg"}}
ALPHABETS = {"dna": set("ACGTN"), "rna": set("ACGTUN"),
             "protein": set("ACDEFGHIKLMNPQRSTVWYBXZJUO*-")}
# curated dirs whose name is not <source_key>_<task>
ALIASES = {"cripsrCas9": "crispr_cas9", "sirna": "sirna_efficacy"}


def source_key(dataset: str, keys: list[str]) -> str | None:
    if dataset in ALIASES:
        return ALIASES[dataset]
    matches = [k for k in keys if dataset == k or dataset.startswith(k + "_")]
    return max(matches, key=len) if matches else None


def latest_raw_download(raw_dir: Path) -> str:
    """Latest recorded download date (YYYY-MM-DD) among a raw source's manifests."""
    latest = ""
    for mf in list(raw_dir.rglob("download_manifest.json")) + \
            [p for p in [raw_dir / "provenance.json"] if p.exists()]:
        m = json.loads(mf.read_text())
        ts = m.get("downloaded_at_utc") or m.get("downloaded_at") or m.get("retrieved_at") or ""
        latest = max(latest, ts[:10])
    return latest


def audit_frame(df: pd.DataFrame, kind: str, alphabet: set[str] | None) -> list[str]:
    out = []
    missing = REQUIRED[kind] - set(df.columns)
    if missing:
        return [f"FAIL {kind}: missing columns {sorted(missing)}"]

    unknown = set(df["split"].unique()) - {"train", "val", "test"}
    if unknown:
        out.append(f"FAIL {kind}: unknown split values {sorted(unknown)}")
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        if len(sub) == 0:
            out.append(f"FAIL {kind}: split {s!r} empty")
        elif kind == "classification" and sub["y_cls"].nunique() < 2:
            out.append(f"FAIL {kind}: split {s!r} has a single class")

    if kind == "classification" and not set(df["y_cls"].unique()) <= {0, 1}:
        out.append(f"FAIL {kind}: y_cls not binary: {sorted(df['y_cls'].unique())[:6]}")
    if kind == "regression":
        bad = int((~pd.to_numeric(df["y_reg"], errors="coerce").notna()).sum())
        if bad:
            out.append(f"FAIL {kind}: {bad} null/non-numeric y_reg values")

    n_len = int((df["sequence"].str.len() != df["length"]).sum())
    if n_len:
        out.append(f"FAIL {kind}: length column wrong for {n_len} rows")
    for col in ("sequence", "series_id"):
        d = int(df[col].duplicated().sum())
        if d:
            out.append(f"FAIL {kind}: {d} duplicate {col} values")

    by_split = {s: set(g["sequence"]) for s, g in df.groupby("split")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        n = len(by_split.get(a, set()) & by_split.get(b, set()))
        if n:
            out.append(f"FAIL {kind}: {n} sequences shared between {a} and {b} (leakage)")

    if alphabet is not None:
        chars = set().union(*df["sequence"].map(set))
        if chars - alphabet:
            out.append(f"FAIL {kind}: unexpected characters {sorted(chars - alphabet)[:8]}")
    return out


def infer_modality(alphabet: set[str]) -> str:
    for m in ("dna", "rna", "protein"):
        if alphabet <= ALPHABETS[m]:
            return m
    return "?"


def audit_dataset(d: Path, spec: dict | None, raw_base: Path,
                  stale_cache: dict[str, str], meta: dict) -> list[str]:
    out = []
    frames = {k: d / f"task_{k}.parquet" for k in REQUIRED if (d / f"task_{k}.parquet").exists()}
    prep = d / "prep_log.json"
    log = json.loads(prep.read_text()) if prep.exists() else {}
    # Curation may change modality (e.g. cds_* translate coding DNA to protein),
    # so the alphabet recorded at curation wins over the raw source's modality.
    # The field holds either a modality name ("rna") or a character set ("ACGT").
    if str(log.get("alphabet", "")).lower() in ALPHABETS:
        modality = str(log["alphabet"]).lower()
        alphabet = ALPHABETS[modality]
    elif log.get("alphabet"):
        alphabet = set(log["alphabet"]) | {"N"}
        modality = infer_modality(alphabet)
    else:
        modality = (spec or {}).get("modality")
        alphabet = ALPHABETS.get(modality)

    dfs = {}
    for kind, path in frames.items():
        try:
            dfs[kind] = pd.read_parquet(path)
        except Exception as e:
            out.append(f"FAIL {kind}: unreadable ({type(e).__name__}: {e})")
    for kind, df in dfs.items():
        out += audit_frame(df, kind, alphabet)
    if len(dfs) == 2 and set(dfs["classification"]["series_id"]) != set(dfs["regression"]["series_id"]):
        out.append("WARN classification and regression cover different series_ids")

    meta.update(modality=modality, tasks=sorted(dfs))
    first = next(iter(dfs.values()), None)
    if first is not None:
        counts = first["split"].value_counts()
        splits = ", ".join(f"{s}={counts.get(s, 0):,}" for s in ("train", "val", "test"))
        print(f"    [{modality or '?'}] {splits}")

    if not log:
        out.append("WARN prep_log.json missing (no provenance)")
        return out
    src = log.get("source", {})
    if "test only" in str(src.get("split_used", "")).lower():
        out.append("WARN curated from the official TEST split only (superseded "
                   "- use the *_official_* dataset)")
    if spec is not None:
        key = spec["_key"]
        if key not in stale_cache:
            stale_cache[key] = latest_raw_download(raw_base / spec["path"])
        curated_on = str(log.get("created_at", ""))[:10]
        # >1 day slack: curation timestamps are naive local time, manifests UTC
        if curated_on and stale_cache[key] and \
                (date.fromisoformat(stale_cache[key]) - date.fromisoformat(curated_on)).days > 1:
            out.append(f"WARN stale: raw source re-downloaded {stale_cache[key]}, "
                       f"curated {curated_on}")
    else:
        out.append("WARN no raw source in paths.yaml matches this dataset")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path,
                    default=Path(__file__).resolve().parents[2] / "paths" / "paths.yaml")
    ap.add_argument("--machine", default=platform.node().split(".")[0])
    ap.add_argument("--datasets", nargs="*", help="subset of curated dataset dir names")
    ap.add_argument("--yaml", type=Path, metavar="FILE",
                    default=Path(__file__).parent / "tasks.yaml",
                    help="task-registry YAML to write (datasets whose status is OK; "
                         "default: tasks.yaml next to this script)")
    ap.add_argument("--no-yaml", action="store_true", help="audit only, write no registry")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.paths.read_text())
    machine = cfg["machines"].get(args.machine) or {}
    if not machine.get("curated_base") or not machine.get("raw_base"):
        sys.exit(f"No raw_base/curated_base for machine {args.machine!r} in {args.paths}")
    curated, raw_base = Path(machine["curated_base"]), Path(machine["raw_base"])
    for key, spec in cfg["datasets"].items():
        spec["_key"] = key
    print(f"machine={args.machine} curated_base={curated}\n")

    status, stale_cache, meta = {}, {}, {}
    for d in sorted(curated.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or \
                not any((d / f"task_{k}.parquet").exists() for k in REQUIRED):
            continue
        if args.datasets and d.name not in args.datasets:
            continue
        key = source_key(d.name, list(cfg["datasets"]))
        print(f"== {d.name}")
        meta[d.name] = {"source": key}
        findings = audit_dataset(d, cfg["datasets"].get(key), raw_base, stale_cache,
                                 meta[d.name])
        for f in findings:
            print(f"    {f}")
        status[d.name] = ("FAIL" if any(f.startswith("FAIL") for f in findings)
                          else "WARN" if findings else "OK")
        print(f"    -> {status[d.name]}\n")

    n = len(status)
    print(f"SUMMARY  {n} datasets: "
          f"{sum(s == 'OK' for s in status.values())} OK, "
          f"{sum(s == 'WARN' for s in status.values())} WARN, "
          f"{sum(s == 'FAIL' for s in status.values())} FAIL")
    for name, s in status.items():
        if s != "OK":
            print(f"  {name}: {s}")

    if args.datasets and not args.no_yaml:
        print("\n[info] partial audit (--datasets): not writing the task registry")
    elif not args.no_yaml:
        ok = {name: meta[name] for name, s in status.items() if s == "OK"}
        header = (f"# Task registry: curated datasets that passed audit_tasks.py\n"
                  f"# Generated on {date.today()} (machine {args.machine}); do not edit -\n"
                  f"# regenerate with: python audit_tasks.py --yaml {args.yaml.name}\n"
                  f"# Dataset dirs live under the machine's curated_base in paths/paths.yaml.\n")
        args.yaml.write_text(header + yaml.safe_dump({"tasks": ok}, sort_keys=True))
        print(f"\n[ok] wrote {args.yaml} ({len(ok)} tasks)")
    return 1 if "FAIL" in status.values() else 2 if "WARN" in status.values() else 0


if __name__ == "__main__":
    sys.exit(main())
