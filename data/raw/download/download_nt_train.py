#!/usr/bin/env python3
"""Download the official NT train split (18 tasks) at a pinned revision.

Target layout under the configured nt_tasks directory:
    test.parquet                          existing combined official test
    download_manifest.json                existing test manifest
    official_train/<task>/train.parquet   downloaded train split
    official_train/download_manifest.json train manifest
    task_split_index.json                 task -> official train/test locations

Only the 18 <task>/train.parquet files are fetched (no .fna duplicates, no
per-task test files). Each file streams to a .tmp path, is verified against
the upstream LFS sha256 for the pinned revision, and only then renamed.
Existing files are kept only if they match the expected hash; a mismatch is
reported and never overwritten. Rerunnable. No curation/filtering/splitting.
"""
from __future__ import annotations

import argparse, hashlib, json, platform, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

REVISION = "851f9946252e90c665cdb3cc3eedb78f1f26197c"  # identified by the raw audit
EXPECTED_TASKS = sorted([
    "H2AFZ", "H3K27ac", "H3K27me3", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3",
    "H3K9ac", "H3K9me3", "H4K20me1", "enhancers", "enhancers_types",
    "promoter_all", "promoter_no_tata", "promoter_tata",
    "splice_sites_acceptors", "splice_sites_all", "splice_sites_donors"])
EXPECTED_TRAIN_TOTAL = 493_242  # upstream default/train per datasets-server (audit)

def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "nt-train-download"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path,
                        default=Path(__file__).resolve().parents[3] / "paths" / "paths.yaml")
    parser.add_argument("--machine", default=platform.node().split(".")[0])
    args = parser.parse_args()

    cfg = yaml.safe_load(args.paths.read_text())
    raw_base = (cfg["machines"].get(args.machine) or {}).get("raw_base")
    if not raw_base:
        parser.error(f"No raw_base for machine {args.machine!r} in {args.paths}")
    dataset = cfg["datasets"]["nt_tasks"]
    repo = dataset["source"]["repo_id"]
    nt_dir = Path(raw_base) / dataset["path"]
    out = nt_dir / "official_train"

    # 1. Confirm the pinned revision holds exactly the expected files.
    tree = api(f"https://huggingface.co/api/datasets/{repo}/tree/"
               f"{REVISION}?recursive=true")
    upstream = {f["path"]: f for f in tree if f.get("type") == "file"
                and f["path"].endswith("/train.parquet")}
    want = {f"{t}/train.parquet" for t in EXPECTED_TASKS}
    if set(upstream) != want:
        print(f"ERROR: pinned revision file set mismatch: only_upstream="
              f"{sorted(set(upstream) - want)} missing={sorted(want - set(upstream))}",
              file=sys.stderr)
        return 1
    not_lfs = [p for p, f in upstream.items() if not f.get("lfs")]
    if not_lfs:
        print(f"ERROR: expected LFS entries, got git blobs for {not_lfs}", file=sys.stderr)
        return 1

    # 2. Download with verify-then-rename; never clobber a mismatching file.
    out.mkdir(parents=True, exist_ok=True)
    records, conflicts, downloaded_bytes = [], [], 0
    for task in EXPECTED_TASKS:
        up = upstream[f"{task}/train.parquet"]
        exp_sha, exp_size = up["lfs"]["oid"], up["lfs"]["size"]
        dest = out / task / "train.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            got = sha256_file(dest)
            if got == exp_sha and dest.stat().st_size == exp_size:
                print(f"[keep] {task}: existing file verified against LFS sha256")
            else:
                conflicts.append(dict(task=task, path=str(dest), local_sha256=got,
                                      expected_sha256=exp_sha))
                print(f"[CONFLICT] {task}: existing file does not match expected hash "
                      "- left untouched, not re-downloaded", file=sys.stderr)
                continue
        else:
            tmp = dest.with_suffix(".parquet.tmp")
            url = (f"https://huggingface.co/datasets/{repo}/resolve/"
                   f"{REVISION}/{task}/train.parquet")
            h = hashlib.sha256()
            req = urllib.request.Request(url, headers={"User-Agent": "nt-train-download"})
            with urllib.request.urlopen(req, timeout=300) as r, tmp.open("wb") as f:
                for chunk in iter(lambda: r.read(16 * 1024 * 1024), b""):
                    f.write(chunk); h.update(chunk)
            if h.hexdigest() != exp_sha or tmp.stat().st_size != exp_size:
                print(f"[FAIL] {task}: downloaded bytes do not match LFS sha256; "
                      f"left at {tmp}", file=sys.stderr)
                conflicts.append(dict(task=task, path=str(tmp),
                                      local_sha256=h.hexdigest(), expected_sha256=exp_sha))
                continue
            tmp.rename(dest)
            downloaded_bytes += exp_size
            print(f"[ok] {task}: {exp_size} bytes, sha256 verified")
        # full batched read: rows, schema, length range
        pf = pq.ParquetFile(dest)
        lmin = lmax = None; rows = 0
        for b in pf.iter_batches(batch_size=65536, columns=["sequence"]):
            mm = pc.min_max(pc.utf8_length(b.column("sequence"))).as_py()
            lmin = mm["min"] if lmin is None else min(lmin, mm["min"])
            lmax = mm["max"] if lmax is None else max(lmax, mm["max"])
            rows += b.num_rows
        records.append(dict(
            task=task, split="train",
            upstream_path=f"{task}/train.parquet",
            path=f"{task}/train.parquet",          # relative to this manifest's directory
            local_relpath=f"official_train/{task}/train.parquet",  # relative to _nt_tasks
            size_bytes=exp_size, sha256=exp_sha,
            upstream_lfs_sha256=exp_sha, verification="lfs-sha256",
            n_rows=rows, len_min=lmin, len_max=lmax,
            columns=pf.schema_arrow.names,
            dtypes={f.name: str(f.type) for f in pf.schema_arrow}))

    if conflicts:
        print(json.dumps(dict(conflicts=conflicts), indent=2))
        return 1

    # 3. Manifest for this download (separate from the test manifest).
    (out / "download_manifest.json").write_text(json.dumps(dict(
        dataset_repo_id=repo, revision=REVISION, split="train",
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
        note="official per-task train.parquet files only; test data lives one level up "
             "(test.parquet, combined, filter column 'task') and is NOT covered here",
        files=records, host=platform.node(), python=sys.version), indent=2))

    # 4. Machine-readable task/split index (paths relative to _nt_tasks).
    index = dict(
        dataset_repo_id=repo,
        train_revision=REVISION,
        test_manifest="download_manifest.json",
        train_manifest="official_train/download_manifest.json",
        schema_note="train and test share one schema: sequence,name,label(int32),task; "
                    "per-task train files carry a redundant constant task column; "
                    "test.parquet is the combined official test - select rows where "
                    "task == <task>",
        tasks={r["task"]: dict(
            train=dict(path=r["local_relpath"], split="train"),
            test=dict(path="test.parquet", split="test",
                      filter_column="task", filter_value=r["task"]))
               for r in records})
    (nt_dir / "task_split_index.json").write_text(json.dumps(index, indent=2))

    total = sum(r["n_rows"] for r in records)
    print(f"\n[done] {len(records)} tasks, downloaded {downloaded_bytes} new bytes, "
          f"total train rows {total} (expected {EXPECTED_TRAIN_TOTAL}: "
          f"{'MATCH' if total == EXPECTED_TRAIN_TOTAL else 'MISMATCH'})")
    return 0 if total == EXPECTED_TRAIN_TOTAL else 1

if __name__ == "__main__":
    sys.exit(main())
