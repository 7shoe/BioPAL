#!/usr/bin/env python3
"""Re-curate NT downstream tasks from the OFFICIAL train/test splits.

Replaces the earlier test-only curation (which re-split the official test
split internally) now that the official per-task train files are downloaded
and verified (see data/raw/download/download_nt_train.py and data/raw/reports/nt_train_test_overlap.json).

Per task:
    train pool = _raw/_nt_tasks/official_train/<task>/train.parquet (official train)
    test  pool = _raw/_nt_tasks/test.parquet rows where task == <task> (official test)

Curation rules (same house rules as the previous NT curation, applied per pool):
    - clean to ACGT, exact-deduplicate, drop exact duplicates with conflicting labels
    - the official train/test boundary is preserved; any train-pool sequence that
      also appears in the test pool is dropped from train (audit measured 0)
    - val is carved out of the official train pool by split-group blocking
      (length bin + k-mer minimizer), never from test
    - test is the full deduplicated official test pool - no subsampling
    - classification only; y_cls is the official binary label

Output (existing nt_tasks_* directories are left untouched):
    <dest-root>/nt_tasks_official_<task_slug>/
        task_classification.parquet, prep_log.json, results/...
    <dest-root>/nt_tasks_official_curation_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

 
PIPELINE_DIR = Path("/home/csiebenschuh/Projects/BinarySignature/dsets/pipelines/nt_tasks")
sys.path.insert(0, str(PIPELINE_DIR))

import curation
from utils.deduplication import (
    DedupConfig, add_split_group, exact_deduplicate_binary_task, slugify,
    split_by_groups, stable_sequence_id,
)

# Hardcoded paths rather than `paths YAML` since it is one-off
NT_DIR = Path("/homes/csiebenschuh/data/bio_sequences/_raw/_nt_tasks")
DEFAULT_DEST_ROOT = Path("/homes/csiebenschuh/data/bio_sequences")
DEFAULT_PREFIX = "nt_tasks_official"
# The working set previously curated (nt_tasks_curation_manifest.json).
DEFAULT_TASKS = ["H3K27ac", "H3K9me3", "splice_sites_acceptors",
                 "splice_sites_all", "splice_sites_donors"]


def dedup_pool(df: pd.DataFrame, *, prefix: str, args) -> tuple[pd.DataFrame, dict]:
    return exact_deduplicate_binary_task(
        df, dataset_prefix=prefix, sequence_col="sequence", label_col="label",
        min_len=args.min_len, max_len=args.max_len)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    ap.add_argument("--dataset-prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--max-train", type=int, default=20_000)
    ap.add_argument("--n-val", type=int, default=1_500)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=4_096)
    ap.add_argument("--group-kmer-k", type=int, default=8)
    ap.add_argument("--group-length-bin-width", type=int, default=50)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    index = json.loads((NT_DIR / "task_split_index.json").read_text())
    train_manifest = json.loads((NT_DIR / "official_train" / "download_manifest.json").read_text())
    test_manifest = json.loads((NT_DIR / "download_manifest.json").read_text())
    test_all = pd.read_parquet(NT_DIR / "test.parquet")

    bioseq_prep = curation.find_bioseq_prep()
    written, skipped = [], []

    for task in args.tasks:
        entry = index["tasks"].get(task)
        if entry is None:
            skipped.append({"task": task, "reason": "not in task_split_index.json"})
            continue
        slug = slugify(task)
        dataset_name = f"{args.dataset_prefix}_{slug}"
        dest = args.dest_root / dataset_name
        print(f"\n{'=' * 100}\n[task] {task!r} -> {dataset_name}\n{'=' * 100}")

        train_raw = pd.read_parquet(NT_DIR / entry["train"]["path"])
        test_raw = test_all[test_all[entry["test"]["filter_column"]].astype(str)
                            .eq(entry["test"]["filter_value"])]

        train_pool, train_summary = dedup_pool(train_raw, prefix=dataset_name, args=args)
        test_pool, test_summary = dedup_pool(test_raw, prefix=dataset_name, args=args)

        # Enforce the official boundary: no train sequence may appear in test.
        test_seqs = set(test_pool["sequence"])
        leak_mask = train_pool["sequence"].isin(test_seqs)
        n_leaked = int(leak_mask.sum())
        if n_leaked:
            print(f"[warn] dropping {n_leaked} train sequences present in official test")
            train_pool = train_pool[~leak_mask].reset_index(drop=True)

        # Carve val out of the official train pool by group blocking (no test carve).
        grouped = add_split_group(train_pool, kmer_k=args.group_kmer_k,
                                  length_bin_width=args.group_length_bin_width)
        trainval = split_by_groups(grouped, max_train=args.max_train,
                                   n_val=args.n_val, n_test=0, seed=args.seed)
        assert not (trainval["split"] == "test").any()

        test_final = add_split_group(test_pool, kmer_k=args.group_kmer_k,
                                     length_bin_width=args.group_length_bin_width)
        test_final["split"] = "test"
        task_df = pd.concat([trainval, test_final], ignore_index=True)

        for s in ("train", "val", "test"):
            sub = task_df[task_df["split"] == s]
            if len(sub) == 0 or sub["y_cls"].nunique() < 2:
                raise ValueError(f"{dataset_name}: split {s!r} empty or single-class")

        baseline = curation.composition_baseline(task_df)
        summary = {
            "dataset_name": dataset_name,
            "source_task": task,
            "source_repo_note": "Official train and test splits (pinned revisions).",
            "train_input": str(NT_DIR / entry["train"]["path"]),
            "test_input": str(NT_DIR / "test.parquet"),
            "train_revision": index["train_revision"],
            "test_downloaded_at": test_manifest.get("downloaded_at_utc"),
            "train_downloaded_at": train_manifest.get("downloaded_at_utc"),
            "dedup_train_pool": train_summary,
            "dedup_test_pool": test_summary,
            "n_train_sequences_dropped_in_test": n_leaked,
            "split_counts": {k: int(v) for k, v in
                             task_df["split"].value_counts().to_dict().items()},
            "class_counts_by_split": {
                s: {str(k): int(v) for k, v in sub["y_cls"].value_counts().to_dict().items()}
                for s, sub in task_df.groupby("split")},
            "composition_baseline": baseline,
            "config": {"max_train": args.max_train, "n_val": args.n_val,
                       "seed": args.seed, "group_kmer_k": args.group_kmer_k,
                       "group_length_bin_width": args.group_length_bin_width,
                       "min_len": args.min_len, "max_len": args.max_len},
        }
        print(json.dumps({k: summary[k] for k in
                          ("split_counts", "class_counts_by_split",
                           "n_train_sequences_dropped_in_test")}, indent=2))

        curation.remove_if_needed(dest, args.overwrite)
        cls_df = curation.build_classification_frame(task_df)
        source = {
            "name": "Nucleotide Transformer downstream tasks",
            "repo_id": index["dataset_repo_id"],
            "split_used": "official train (train/val via internal group blocking) "
                          "+ official test (test, complete)",
            "raw_path": str(NT_DIR),
            "train_revision": index["train_revision"],
            "source_task": task,
            "license": "see Hugging Face dataset card",
            "notes": "train/val come exclusively from the official train split; "
                     "test is the full deduplicated official test split.",
        }
        notes = (
            f"NT downstream task {task!r} re-curated from the OFFICIAL splits. "
            "train/val are carved from the official train split by split-group "
            "blocking (length bin + 8-mer minimizer, seed "
            f"{args.seed}); test is the complete deduplicated official test split. "
            "Exact duplicates with conflicting labels dropped per pool; train "
            "sequences appearing in test dropped from train "
            f"(n={n_leaked}). Supersedes the earlier test-only nt_tasks_{slug} "
            "curation, which remains on disk for provenance."
        )
        written_dir = bioseq_prep.write_dataset(
            args.dest_root, dataset_name, classification=cls_df, source=source,
            notes=notes, prep_script=str(Path(__file__).resolve()), strict=True)
        curation.write_aux(Path(written_dir), summary, task_df)
        if not bioseq_prep.validate_dataset_dir(written_dir):
            raise RuntimeError(f"validation failed for {written_dir}")
        written.append(str(written_dir))

    print("\n[done] written:")
    for w in written:
        print("  -", w)
    if skipped:
        print("[info] skipped:", skipped)
    (args.dest_root / f"{args.dataset_prefix}_curation_manifest.json").write_text(
        json.dumps({"written": written, "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
