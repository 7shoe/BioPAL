#!/usr/bin/env python3
"""Audit the raw datasets listed in paths.yaml on the current machine.

Checks per dataset:
    presence   dataset directory (and any `expect` entries) exists
    integrity  every file recorded in download_manifest.json files exists;
               recorded sizes match; recorded parquet row counts match the
               parquet footer (cheap - no data read); sha256 only with --hash
    splits     split names and row counts recorded by the manifests, reported
               so train/val/test coverage is visible at a glance
    freshness  Hugging Face upstream lastModified vs. local download time
               (skipped with --offline and for non-HF sources)

This is the quick, routine audit. Deep content verification (full reads,
upstream byte identity, per-file commit archaeology) lives in
audit_raw_deep.py and its findings in RAW_AUDIT_FINDINGS.md.

Exit codes: 0 all OK, 1 any FAIL, 2 only WARNs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import urllib.request
from pathlib import Path

import yaml

PAYLOAD_SUFFIXES = {
    ".parquet",
    ".csv",
    ".tsv",
    ".fa",
    ".fasta",
    ".txt",
    ".pkl",
    ".zip"
    }


def reroot(recorded: str, dataset_root: Path, original_name: str | None = None) -> Path:
    """Map a (possibly stale absolute) recorded path into dataset_root."""
    parts = Path(recorded).parts
    for anchor in (dataset_root.name, original_name):
        if anchor in parts:
            rel = Path(*parts[parts.index(anchor) + 1:])
            return dataset_root / rel
    return dataset_root / recorded  # already relative


def manifest_records(manifest: Path, root: Path, original_name: str | None = None) -> list[dict]:
    """Normalize any known manifest schema to [{path, n_rows?, size_bytes?, sha256?, split?}]."""
    m = json.loads(manifest.read_text())
    recs: list[dict] = []
    if "files" in m:                                   # rfam, nt official_train
        for f in m["files"]:
            p = (reroot(f["path"], root, original_name) if Path(f["path"]).is_absolute()
                 else manifest.parent / f["path"])
            recs.append({**f, "path": p, "split": f.get("split") or m.get("split")})
    elif "splits" in m:                                # fish_gue, gen_lrb, plant_gen, rna_dst
        for split, info in m["splits"].items():
            recs.append({"path": reroot(info["path"], root, original_name), "n_rows": info.get("n_rows"),
                         "split": split})
    elif "configs" in m:                               # cds
        for cfg, splits in m["configs"].items():
            for split, info in splits.items():
                recs.append({"path": reroot(info["path"], root, original_name), "n_rows": info.get("n_rows"),
                             "split": f"{cfg}/{split}"})
    elif "parquet" in m:                               # nt_tasks test.parquet
        recs.append({"path": reroot(m["parquet"], root, original_name), "n_rows": m.get("n_rows"),
                     "split": m.get("split")})
    elif "local_path" in m:                            # siRNA provenance.json
        recs.append({"path": reroot(m["local_path"], root, original_name), "size_bytes": m.get("size_bytes"),
                     "sha256": m.get("sha256"), "split": None})
    return recs


def downloaded_at(manifest: Path) -> str | None:
    m = json.loads(manifest.read_text())
    return m.get("downloaded_at_utc") or m.get("downloaded_at") or m.get("retrieved_at")


def parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_last_modified(repo_id: str) -> str:
    req = urllib.request.Request(f"https://huggingface.co/api/datasets/{repo_id}",
                                 headers={"User-Agent": "biopal-raw-audit"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["lastModified"]


def audit_dataset(name: str, spec: dict, root: Path, *, offline: bool, do_hash: bool) -> list[str]:
    """Return findings as ['LEVEL message', ...]; empty means OK."""
    out: list[str] = []
    if not root.is_dir():
        return [f"FAIL dataset directory missing: {root}"]
    for rel in spec.get("expect", []):
        if not (root / rel).exists():
            out.append(f"FAIL expected entry missing: {rel}")

    manifests = sorted(root.rglob("download_manifest.json")) + \
        [p for p in [root / "provenance.json"] if p.exists()]
    if not manifests and spec["source"]["type"] == "huggingface":
        out.append("WARN no download_manifest.json found")

    n_files = n_rows_ok = n_rows_checked = 0
    splits: dict[str, list[int | None]] = {}
    newest_download = None
    for mf in manifests:
        ts = downloaded_at(mf)
        newest_download = max(newest_download or "", ts or "")
        for rec in manifest_records(mf, root):
            p = rec["path"]
            label = p.relative_to(root) if p.is_relative_to(root) else p
            payload = p.suffix.lower() in PAYLOAD_SUFFIXES and not any(
                s.startswith(".") for s in p.relative_to(root).parts[:-1] if p.is_relative_to(root))
            if rec.get("split"):
                splits.setdefault(str(rec["split"]), []).append(rec.get("n_rows"))
            if not p.exists():
                out.append(f"{'FAIL' if payload else 'WARN'} missing file: {label}")
                continue
            n_files += 1
            size = rec.get("size_bytes")
            if size is not None and p.stat().st_size != size:
                out.append(f"{'FAIL' if payload else 'WARN'} size mismatch: {label} "
                           f"({p.stat().st_size} != {size})")
            if rec.get("n_rows") is not None and p.suffix == ".parquet":
                n_rows_checked += 1
                if parquet_rows(p) == rec["n_rows"]:
                    n_rows_ok += 1
                else:
                    out.append(f"FAIL row count mismatch: {label} "
                               f"({parquet_rows(p)} != {rec['n_rows']})")
            if do_hash and rec.get("sha256") and p.exists():
                if sha256_file(p) != rec["sha256"]:
                    out.append(f"FAIL sha256 mismatch: {label}")

    if splits:
        def fmt(name: str, counts: list[int | None]) -> str:
            label = (f"{name}={sum(counts):,}" if all(n is not None for n in counts)
                     else name)
            return label + (f" ({len(counts)} files)" if len(counts) > 1 else "")
        print(f"    splits: {', '.join(fmt(s, ns) for s, ns in sorted(splits.items()))}")
    print(f"    files: {n_files} present, parquet rows verified {n_rows_ok}/{n_rows_checked}")

    src = spec["source"]
    if src["type"] == "huggingface" and not offline:
        try:
            upstream = hf_last_modified(src["repo_id"])
            if newest_download and upstream > newest_download:
                out.append(f"WARN upstream changed after download "
                           f"(upstream {upstream} > local {newest_download}); "
                           "may be additive - see RAW_AUDIT_FINDINGS.md")
            else:
                print(f"    freshness: upstream {upstream} <= downloaded {newest_download}")
        except Exception as e:
            out.append(f"WARN freshness check failed: {type(e).__name__}: {e}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path,
                    default=Path(__file__).resolve().parents[2] / "paths" / "paths.yaml")
    ap.add_argument("--machine", help="key in machines: (default: current hostname)")
    ap.add_argument("--datasets", nargs="*", help="subset of dataset names")
    ap.add_argument("--offline", action="store_true", help="skip upstream freshness checks")
    ap.add_argument("--hash", action="store_true", help="verify recorded sha256 (slow)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.paths.read_text())
    machine = args.machine or platform.node().split(".")[0]
    base = (cfg["machines"].get(machine) or {}).get("raw_base")
    if not base:
        sys.exit(f"No raw_base for machine {machine!r}; known: {sorted(cfg['machines'])}")
    base = Path(base)
    print(f"machine={machine} base={base}\n")

    status: dict[str, str] = {}
    for name, spec in cfg["datasets"].items():
        if args.datasets and name not in args.datasets:
            continue
        src = spec["source"]
        print(f"== {name} ({spec['path']}) [{src['type']}: {src.get('repo_id') or src.get('url')}]")
        findings = audit_dataset(name, spec, base / spec["path"],
                                 offline=args.offline, do_hash=args.hash)
        for f in findings:
            print(f"    {f}")
        status[name] = ("FAIL" if any(f.startswith("FAIL") for f in findings)
                        else "WARN" if findings else "OK")
        print(f"    -> {status[name]}\n")

    width = max(map(len, status))
    print("SUMMARY")
    for name, s in status.items():
        print(f"  {name:<{width}}  {s}")
    return 1 if "FAIL" in status.values() else 2 if "WARN" in status.values() else 0


if __name__ == "__main__":
    sys.exit(main())
