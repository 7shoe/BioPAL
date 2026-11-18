#!/usr/bin/env python3
"""Print a table of raw datasets: modality and sample counts per split.

Counts come from manifests, Parquet footers, or source-specific raw tables.
Unsplit samples appear under 'other'. '?' means the count is unavailable;
'-' means the bucket has no recorded samples.
"""
from __future__ import annotations

import argparse
import csv
import platform
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import yaml

from audit_raw import PAYLOAD_SUFFIXES, manifest_records, parquet_rows

COLS = ["train", "val", "test", "other"]


def split_bucket(name: str) -> str:
    s = str(name).lower().rsplit("/", 1)[-1]          # cds records 'config/split'
    s = {"validation": "val"}.get(s, s)
    return s if s in COLS else "other"


def table_rows(stream, delimiter: str) -> int:
    reader = csv.reader(stream, delimiter=delimiter)
    next(reader, None)  # header
    return sum(1 for row in reader if row)


def proteingym_rows(root: Path) -> int | None:
    folder = "DMS_ProteinGym_substitutions"
    archive = root / f"{folder}.zip"
    if archive.is_file():
        # Use one copy of each assay, even with a partially extracted archive.
        with ZipFile(archive) as bundle:
            assays = [name for name in bundle.namelist()
                      if Path(name).parent == Path(folder) and name.endswith(".csv")]
            total = 0
            for name in assays:
                with bundle.open(name) as raw, TextIOWrapper(
                        raw, encoding="utf-8-sig", newline="") as stream:
                    total += table_rows(stream, ",")
            return total if assays else None

    assays = sorted((root / folder).glob("*.csv"))
    total = 0
    for path in assays:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            total += table_rows(stream, ",")
    return total if assays else None


def split_counts(root: Path, dataset: str = "") -> dict[str, tuple[int, int]]:
    """bucket -> (rows_recorded, files_without_counts)"""
    if dataset in {"proteingym", "crispr_cas9", "sirna_efficacy"}:
        count = None
        if dataset == "proteingym":
            count = proteingym_rows(root)
        elif dataset == "crispr_cas9":
            manifest = root / "unpacked" / "manifest.tsv"
            if manifest.is_file():
                with manifest.open(encoding="utf-8-sig", newline="") as stream:
                    cohorts = csv.DictReader(stream, delimiter="\t")
                    count = sum(int(row["n_samples"]) for row in cohorts)
        else:
            table = root / "information_all.txt"
            if table.is_file():
                with table.open(encoding="utf-8-sig", newline="") as stream:
                    count = table_rows(stream, "\t")
        counts = dict.fromkeys(COLS, (0, 0))
        counts["other"] = (0, 1) if count is None else (count, 0)
        return counts

    out = {c: [0, 0] for c in COLS}
    manifests = sorted(root.rglob("download_manifest.json")) + \
        [p for p in [root / "provenance.json"] if p.exists()]
    for mf in manifests:
        # Historical HF manifests may contain absolute paths under _<dataset>.
        for rec in manifest_records(mf, root, original_name=f"_{dataset}" if dataset else None):
            p = Path(rec["path"])
            n = rec.get("n_rows")
            if n is None and p.suffix == ".parquet" and p.exists():
                n = parquet_rows(p)                   # cheap: footer metadata only
            b = out[split_bucket(rec["split"]) if rec.get("split") else "other"]
            if n is not None:
                b[0] += n
            elif p.suffix.lower() in PAYLOAD_SUFFIXES:
                b[1] += 1
    return {c: tuple(v) for c, v in out.items()}


def cell(rows: int, unknown: int) -> str:
    if rows == 0:
        return "?" if unknown else "-"
    return f"{rows:,}" + ("+?" if unknown else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path,
                    default=Path(__file__).resolve().parents[2] / "paths" / "paths.yaml")
    ap.add_argument("--machine", default=platform.node().split(".")[0])
    args = ap.parse_args()

    cfg = yaml.safe_load(args.paths.read_text())
    raw_base = (cfg["machines"].get(args.machine) or {}).get("raw_base")
    if not raw_base:
        ap.error(f"No raw_base for machine {args.machine!r} in {args.paths}; use --machine")
    base = Path(raw_base)

    rows = []
    for name, spec in cfg["datasets"].items():
        counts = split_counts(base / spec["path"], name)
        total_rows = sum(r for r, _ in counts.values())
        total_unknown = sum(u for _, u in counts.values())
        rows.append([name, spec.get("modality", "?")]
                    + [cell(*counts[c]) for c in COLS]
                    + [cell(total_rows, total_unknown)])

    modality_order = {"dna": 0, "rna": 1, "protein": 2}
    rows.sort(key=lambda row: modality_order.get(row[1], len(modality_order)))

    header = ["dataset", "modality"] + COLS + ["total"]
    widths = [max(len(r[i]) for r in rows + [header]) for i in range(len(header))]
    fmt = "  ".join(f"{{:{'<' if i < 2 else '>'}{w}}}" for i, w in enumerate(widths))
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))


if __name__ == "__main__":
    main()
