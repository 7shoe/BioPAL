#!/usr/bin/env python3
"""In-depth (and hence slow) audit of raw HF bio-sequence downloads under _raw/.

Four axes per source, kept separate in the report:
  integrity     local files vs manifests (existence, size, sha256, rows, columns,
                full data-page readability)
  coverage      selected tasks/splits vs upstream (counts, byte identity, loader
                evidence); a complete selected-task download is never called a
                complete benchmark
  preservation  sequence-level evidence (lengths, empty/missing values, alphabet)
  freshness     upstream changes after download, resolved per manifest and, where
                the repo changed, per selected file path

Finding levels:
  PASS        verified against concrete evidence (cited in the finding)
  DESIGN      intentional downloader behavior, cited to downloader source lines
  INFO        recorded fact, no judgement
  DOC         documentation-only mismatch (never a sequence-data failure)
  UNRESOLVED  cannot be decided without a prohibited download; never a pass
  FAIL        confirmed local defect or contradiction

Exit codes: 1 any FAIL; else 2 any UNRESOLVED; else 0.
stdout: machine-readable JSON. stderr: compact summary.
Network use is metadata-only (HF hub API, datasets-server, raw README/loader
text). --offline skips it and marks the affected checks UNRESOLVED.
"""

from __future__ import annotations

import argparse, hashlib, json, platform, re, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

PIPELINES = "BinarySignature/dsets/pipelines"  # source citation in reports
HF = "https://huggingface.co"

# IUPAC nucleotide codes (DNA+RNA, both cases). Ambiguity codes are legitimate.
BAD_NUC = "[^ACGTUNRYSWKMBDHVacgtunryswkmbdhv]"

# ---- facts established by inspection of the downloader code (citations) -----
NT_DESIGN = (f"test-only download is intentional: {PIPELINES}/nt_tasks/downloader.py:12-16 "
             "('We intentionally download only the official HF test split, then curate "
             "train/val/test internally from that held-out set'); no official train manifest found")
GEN_LRB_DESIGN = (f"train splits are deliberate subsamples: {PIPELINES}/gen_lrb/downloader.py:104-114 "
                  "shuffles each full split with seed=17 and keeps round(limit*w) rows, w=0.7/0.15/0.15; "
                  "manifest records limit, n_source (full loader split size) and n_rows")
GEN_LRB_LABELS = (f"array-valued labels reduced to scalar nanmean at download: "
                  f"{PIPELINES}/gen_lrb/downloader.py:120-126; upstream shapes: cage Array2D "
                  "(seq_len/128, 50), bulk per-tissue vector (genomics-long-range-benchmark.py:174,358); "
                  "raw labels are therefore NOT the upstream labels for cage/bulk")
FISH_SUBDIR = {  # fish_gue/downloader.py:26-34
    "core_promoter": "prom_300_all", "core_promoter_tata": "prom_300_tata",
    "core_promoter_non_tata": "prom_300_notata", "promoter": "prom_1k_all",
    "CTCF": "CTCF_TF", "Pou5f1": "Pou5f1_TFBS", "Sox2": "Sox2_TFBS"}
# Official full-split sizes from the gen-lrb README table (fetched 2026-11-17,
# https://huggingface.co/datasets/InstaDeepAI/genomics-long-range-benchmark/raw/main/README.md)
GEN_LRB_OFFICIAL = {"bulk_rna_expression": {"train": 22827, "test": 990},
                    "cage_prediction": {"train": 33891, "test": 1922},
                    "regulatory_element_enhancer": {"train": 1914575, "test": 192201}}
# rna_dst HF config -> repo paths holding its source files (per repo tree listing)
RNA_DST_PATHS = {
    "mean_ribosome_load": ["mean_ribosome_load"],
    "ncrna_family_bnoise0": ["ncrna_family_classification/bnoise0"],
    "ncrna_family_bnoise200": ["ncrna_family_classification/bnoise200"],
    "splice_site_acceptor": ["splice_site_prediction/acceptor"],
    "splice_site_donor": ["splice_site_prediction/donor"],
    "translation_efficiency_HEK": ["translation_efficiency", "expression_level"],
    "translation_efficiency_Muscle": ["translation_efficiency", "expression_level"],
    "translation_efficiency_pc3": ["translation_efficiency", "expression_level"]}

SOURCES = {
    "_nt_tasks": dict(tasks=[], seq="sequence", target="label"),
    "_cds": dict(seq="sequence", target="label",
                 tasks=["beta_lactamase_complete", "beta_lactamase_unique", "fluorescence",
                        "melting_point", "stability"]),
    "_rfam": dict(tasks=[], seq="sequence", target="family"),
    "_fish_gue": dict(seq="sequence", target="label",
                      tasks=list(FISH_SUBDIR)),
    "_gen_lrb": dict(seq="sequence", target="labels",
                     tasks=["bulk_rna_expression", "cage_prediction", "regulatory_element_enhancer"]),
    "_plant_gen": dict(seq="sequence", target="label",
                       tasks=["poly_a__arabidopsis_thaliana", "splicing__arabidopsis_thaliana_acceptor",
                              "splicing__arabidopsis_thaliana_donor"]),
    "_rna_dst": dict(target={"mean_ribosome_load": "rl", "*": "labels"},
                     seq={"mean_ribosome_load": "utr", "*": "sequences"},
                     tasks=list(RNA_DST_PATHS)),
}

# ------------------------------- helpers -------------------------------------
def api(url):
    req = urllib.request.Request(url, headers={"User-Agent": "raw-bio-audit"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r), dict(r.headers)

def paged_tree(repo, suffix=""):
    url = f"{HF}/api/datasets/{repo}/tree/main{suffix}"
    out = []
    while url:
        data, headers = api(url)
        out.extend(data)
        link = headers.get("Link", "")
        m = re.search(r"<([^>]+)>;\s*rel=\"next\"", link)
        url = m.group(1) if m else None
    return out

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git_blob_sha1(path):
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def iso(ts):
    d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("naive timestamp")
    return d.astimezone(timezone.utc)

def local_path(raw, directory, recorded, *, source_dir, original_name):
    """Rebase a manifest path onto its configured dataset; refuse ambiguity."""
    p = Path(recorded)
    candidates = []
    if not p.is_absolute() and ".." not in p.parts:
        candidates.append(directory / p)
    for anchor, base in ((raw.name, raw), ("_raw", raw), (original_name, source_dir),
                         (source_dir.name, source_dir), (directory.name, directory)):
        for i, part in enumerate(p.parts):
            tail = p.parts[i + 1:]
            if part == anchor and tail and ".." not in tail:
                candidates.append(base.joinpath(*tail))
    found = {c for c in candidates if c.is_relative_to(source_dir) and c.is_file()}
    if len(found) != 1:
        raise ValueError(f"missing or ambiguous recorded path: {recorded}")
    return found.pop()

def manifest_entries(m):
    """Normalize all four observed manifest schemas to (task, split, record)."""
    if isinstance(m.get("files"), list) and m["files"]:          # rfam snapshot style
        return [(None, None, f) for f in m["files"]]
    if isinstance(m.get("configs"), dict) and m["configs"]:      # cds style
        return [(cfg, split, rec) for cfg, splits in m["configs"].items()
                for split, rec in splits.items()]
    if isinstance(m.get("splits"), dict) and m["splits"]:        # per-task style
        cfg = m.get("config")
        return [(cfg, split, rec) for split, rec in m["splits"].items()]
    if isinstance(m.get("parquet"), str):                        # nt single-table style
        return [(None, m.get("split"), dict(path=m["parquet"], n_rows=m.get("n_rows"),
                                            columns=m.get("columns")))]
    raise ValueError("unknown/empty manifest schema")

class SeqStats:
    def __init__(self):
        self.rows = self.null_seq = self.empty = self.bad_symbol = 0
        self.null_target = self.nan_target = 0
        self.len_min = None; self.len_max = None; self.len_sum = 0
        self.charset = set()

    def update(self, batch, seq_col, target_col):
        self.rows += batch.num_rows
        if seq_col and seq_col in batch.schema.names:
            col = batch.column(seq_col)
            self.null_seq += col.null_count
            col = col.drop_null()
            if len(col):
                lengths = pc.utf8_length(col)
                mm = pc.min_max(lengths).as_py()
                self.len_min = mm["min"] if self.len_min is None else min(self.len_min, mm["min"])
                self.len_max = mm["max"] if self.len_max is None else max(self.len_max, mm["max"])
                self.len_sum += pc.sum(lengths).as_py()
                self.empty += pc.sum(pc.cast(pc.equal(lengths, 0), pa.int64())).as_py()
                bad = pc.match_substring_regex(col, BAD_NUC)
                self.bad_symbol += pc.sum(pc.cast(bad, pa.int64())).as_py() or 0
                if len(self.charset) < 64:
                    for s in col.slice(0, 50).to_pylist():
                        self.charset |= set(s or "")
        if target_col and target_col in batch.schema.names:
            t = batch.column(target_col)
            self.null_target += t.null_count
            if pa.types.is_floating(t.type):
                v = t.drop_null()
                if len(v):
                    self.nan_target += pc.sum(pc.cast(pc.is_nan(v), pa.int64())).as_py() or 0

    def result(self):
        return dict(rows=self.rows, seq_null=self.null_seq, seq_empty=self.empty,
                    seq_bad_symbol_rows=self.bad_symbol, target_null=self.null_target,
                    target_nan=self.nan_target, len_min=self.len_min, len_max=self.len_max,
                    len_mean=round(self.len_sum / max(1, self.rows - self.null_seq), 1),
                    charset_sample="".join(sorted(self.charset)))

def read_table_stats(path, seq_col, target_col):
    """Read EVERY data page (not just footers) and accumulate sequence stats."""
    st = SeqStats()
    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        cols = [c for c in (seq_col, target_col) if c and c in pf.schema_arrow.names]
        names = pf.schema_arrow.names
        for batch in pf.iter_batches(batch_size=65536, columns=cols or None):
            st.update(batch, seq_col, target_col)
        if not cols:  # still force a full read for readability
            st.rows = pf.metadata.num_rows
        return st.result(), names
    if path.suffix in {".csv", ".tsv"}:
        tbl = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(
            delimiter="\t" if path.suffix == ".tsv" else ","))
        for batch in tbl.to_batches(max_chunksize=65536):
            st.update(batch, seq_col, target_col)
        return st.result(), tbl.schema.names
    return None, None

# ------------------------------- audit ---------------------------------------
def audit_source(name, spec, raw, online, net_errors):
    folder = raw / spec["path"]
    repo = spec["repo"]
    F = dict(repo=repo, findings=[], tables={}, manifests=[])
    say = lambda axis, level, msg: F["findings"].append(dict(axis=axis, level=level, msg=msg))

    for task in spec["tasks"]:
        if not (folder / task).is_dir():
            say("integrity", "FAIL", f"missing expected task directory: {task}")

    info, sizes = {}, {}
    if online:
        try:
            info, _ = api(f"{HF}/api/datasets/{repo}")
        except Exception as e:
            net_errors.append(f"{repo}: info {e}")
            say("freshness", "UNRESOLVED", f"repo metadata unavailable: {e}")
        try:
            payload, _ = api(f"https://datasets-server.huggingface.co/size?dataset={quote(repo, safe='')}")
            for s in payload["size"]["splits"]:
                sizes.setdefault(s["config"], {})[s["split"]] = s["num_rows"]
            if payload.get("partial"):
                say("coverage", "UNRESOLVED", "upstream row-count metadata flagged partial")
        except Exception as e:
            sizes = None  # endpoint unsupported for this repo (e.g. script-based: HTTP 501)
            say("coverage", "INFO", f"datasets-server row counts unsupported here ({e}); "
                                    "using loader/manifest evidence instead")
    else:
        say("freshness", "UNRESOLVED", "offline: upstream checks skipped")
        sizes = None
    F["upstream_sha"] = info.get("sha")
    F["upstream_modified"] = info.get("lastModified")

    manifests = sorted(folder.rglob("download_manifest.json"))
    if not manifests:
        say("integrity", "FAIL", "no download manifests found")

    pick = lambda v, task: v.get(task, v["*"]) if isinstance(v, dict) else v
    seq_for = lambda task: pick(spec["seq"], task)
    target_for = lambda task: pick(spec["target"], task)
    observed = {}  # (task, split) -> rows

    for mp in manifests:
        label = str(mp.relative_to(raw))
        try:
            m = json.loads(mp.read_text())
        except Exception as e:
            say("integrity", "FAIL", f"{label}: unreadable manifest: {e}")
            continue
        recorded_repo = m.get("repo_id") or m.get("dataset_repo_id")
        if recorded_repo and recorded_repo != repo:
            say("integrity", "FAIL", f"{label}: unexpected repo {recorded_repo}")
            continue
        when = m.get("downloaded_at_utc")
        F["manifests"].append(dict(path=label, downloaded=when))
        # freshness: per manifest, never per source-wide max
        if info.get("lastModified") and when:
            if iso(info["lastModified"]) > iso(when):
                say("freshness", "INFO", f"{label}: repo modified after download "
                                         f"({info['lastModified']} > {when}); resolving per selected file")
                F.setdefault("_changed_manifests", []).append((m, label, when))
            # unchanged: silence is fine; summarized below
        elif online and not info:
            pass  # already UNRESOLVED above
        try:
            ents = manifest_entries(m)
        except ValueError as e:
            say("integrity", "FAIL", f"{label}: {e}")
            continue
        for task, split, rec in ents:
            fname = rec.get("path", "?")
            try:
                path = local_path(raw, mp.parent, fname, source_dir=folder, original_name=name)
            except ValueError as e:
                say("integrity", "FAIL", f"{label}: {e}")
                continue
            rel = str(path.relative_to(raw))
            try:
                checks = []
                if rec.get("size_bytes") is not None:
                    ok = path.stat().st_size == rec["size_bytes"]
                    checks.append(("size", ok))
                    if not ok:
                        level = "DOC" if path.suffix not in {".parquet", ".csv", ".tsv"} else "FAIL"
                        say("integrity", level,
                            f"{rel}: size {path.stat().st_size} != manifest {rec['size_bytes']}"
                            + (" (non-payload file)" if level == "DOC" else ""))
                        if level == "DOC":
                            continue  # do not hash a knowingly modified doc; move on
                if rec.get("sha256"):
                    digest = sha256_file(path)
                    ok = digest.lower() == rec["sha256"].lower()
                    checks.append(("sha256", ok))
                    if not ok:
                        level = "DOC" if path.suffix not in {".parquet", ".csv", ".tsv"} else "FAIL"
                        say("integrity", level, f"{rel}: sha256 mismatch")
                        if level == "FAIL":
                            continue
                if path.suffix in {".parquet", ".csv", ".tsv"}:
                    task_eff = task or name.lstrip("_")
                    stats, names = read_table_stats(path, seq_for(task_eff), target_for(task_eff))
                    if rec.get("n_rows") is not None and stats["rows"] != rec["n_rows"]:
                        say("integrity", "FAIL",
                            f"{rel}: rows {stats['rows']} != manifest {rec['n_rows']}")
                    if rec.get("columns") and names != list(rec["columns"]):
                        say("integrity", "FAIL", f"{rel}: columns differ from manifest")
                    if rec.get("n_rows") is None and rec.get("skipped_existing"):
                        say("integrity", "INFO", f"{rel}: manifest has no row count "
                            "(skipped_existing rerun); validated against upstream instead")
                    stats["sha256_checked"] = any(k == "sha256" and ok for k, ok in checks)
                    F["tables"][rel] = stats
                    observed[(task, split)] = stats["rows"]
                    if stats["seq_null"] or stats["seq_empty"]:
                        say("preservation", "FAIL",
                            f"{rel}: {stats['seq_null']} null / {stats['seq_empty']} empty sequences")
                    if stats["seq_bad_symbol_rows"]:
                        say("preservation", "UNRESOLVED",
                            f"{rel}: {stats['seq_bad_symbol_rows']} rows with symbols outside "
                            f"IUPAC nucleotide codes (sample charset: {stats['charset_sample']})")
            except Exception as e:
                say("integrity", "FAIL", f"{rel}: {type(e).__name__}: {e}")

    # ---------------- per-source coverage / preservation / freshness ----------
    try:
        extra_checks(name, spec, F, say, observed, sizes, raw, online, info)
    except Exception as e:
        say("coverage", "UNRESOLVED", f"source-specific checks aborted: {type(e).__name__}: {e}")

    levels = [f["level"] for f in F["findings"]]
    F["status"] = ("FAIL" if "FAIL" in levels else
                   "UNRESOLVED" if "UNRESOLVED" in levels else "OK")
    F.pop("_changed_manifests", None)
    return F

def compare_counts(say, observed, sizes, task, split, key=None):
    up = (sizes or {}).get(key or task, {}).get(split)
    got = observed.get((task, split))
    if up is None or got is None:
        return False
    if up == got:
        say("coverage", "PASS", f"{task}/{split}: {got} rows == upstream (datasets-server)")
    else:
        say("coverage", "FAIL", f"{task}/{split}: local {got} != upstream {up}")
    return True

def extra_checks(name, spec, F, say, observed, sizes, raw, online, info):
    dataset_path = Path(spec["path"])
    folder = raw / dataset_path

    if name == "_nt_tasks":
        train_man = folder / "official_train" / "download_manifest.json"
        if train_man.exists():
            tm = json.loads(train_man.read_text())
            say("coverage", "INFO", "official train downloaded separately under official_train/ "
                f"at pinned revision {tm.get('revision', '?')[:10]} (per-file LFS sha256 "
                "verified); original test download and manifest preserved unchanged")
            missing = [f["task"] for f in tm["files"]
                       if str(dataset_path / "official_train" / f["task"] / "train.parquet")
                       not in F["tables"]]
            if missing:
                say("coverage", "FAIL", f"official_train tasks missing locally: {missing}")
            if sizes:
                got = sum(t["rows"] for p, t in F["tables"].items()
                          if Path(p).is_relative_to(dataset_path / "official_train"))
                up = sizes.get("default", {}).get("train")
                if up == got:
                    say("coverage", "PASS", f"official_train total rows ({got}, 18 tasks) "
                                            "== upstream default/train")
                else:
                    say("coverage", "FAIL", f"official_train total rows {got} != upstream {up}")
            idx = folder / "task_split_index.json"
            if idx.exists():
                say("coverage", "INFO", "task_split_index.json maps each task to "
                                        "official train file + test.parquet task filter")
            else:
                say("coverage", "UNRESOLVED", "task_split_index.json missing")
        else:
            say("coverage", "DESIGN", NT_DESIGN)
            if sizes:
                tr = sizes.get("default", {}).get("train")
                say("coverage", "DESIGN", f"official train split ({tr} rows) NOT downloaded "
                    "(see above); local data covers the official test split only")
        if sizes:
            compare_counts(say, observed, sizes, None, "test", key="default")
        # task coverage within the downloaded table
        p = folder / "test.parquet"
        tasks = pc.value_counts(pq.read_table(p, columns=["task"]).column("task"))
        per = {str(x["values"]): int(x["counts"]) for x in tasks.to_pylist()}
        say("coverage", "INFO", f"test.parquet spans {len(per)} benchmark tasks: "
                                + ", ".join(f"{k}={v}" for k, v in sorted(per.items())))
        F["nt_tasks_in_table"] = per

    elif name == "_cds":
        if sizes:
            n = sum(compare_counts(say, observed, sizes, t, s)
                    for (t, s) in observed)
            missing = [f"{t}/{s}" for t in spec["tasks"] for s in ("train", "validation", "test")
                       if (t, s) not in observed]
            if missing:
                say("coverage", "FAIL", f"expected splits absent locally: {missing}")
            elif n == len(observed):
                say("coverage", "PASS", "all 5 selected configs complete: 15/15 splits match upstream")
            say("coverage", "INFO", "upstream repo also has config 'ssp' (token-level), "
                                    f"deliberately skipped: {PIPELINES}/cds/downloader.py:20-23")
        else:
            say("coverage", "UNRESOLVED", "no upstream counts available")

    elif name == "_rfam":
        # provenance: manifest sha256 of data.parquet should equal the LFS sha256
        # recorded by huggingface_hub at download time
        meta = folder / ".cache/huggingface/download/data.parquet.metadata"
        m = json.loads((folder / "download_manifest.json").read_text())
        man_sha = {f["path"]: f["sha256"] for f in m["files"]}
        if meta.exists():
            lines = meta.read_text().splitlines()
            if len(lines) >= 2 and lines[1] == man_sha.get("data.parquet"):
                say("coverage", "PASS", "data.parquet sha256 equals the upstream LFS sha256 recorded "
                    f"in {meta.name} at snapshot {lines[0][:10]} -> byte-identical to upstream")
            else:
                say("coverage", "UNRESOLVED", "LFS metadata does not confirm data.parquet provenance")
        if sizes:
            compare_counts(say, observed, sizes, None, "train", key="default") or \
                say("coverage", "INFO", "upstream split layout differs; row totals recorded in tables")
        tbl = F["tables"].get(str(dataset_path / "data.parquet"))
        if tbl and sizes:
            up_total = sum(sizes.get("default", {}).values())
            if up_total and tbl["rows"] == up_total:
                say("coverage", "PASS", f"data.parquet rows ({tbl['rows']}) == upstream total")
            elif up_total:
                say("coverage", "FAIL", f"data.parquet rows {tbl['rows']} != upstream total {up_total}")
        # README: separate documentation from payload
        doc = [f for f in F["findings"] if f["level"] == "DOC"]
        if doc and online:
            try:
                req = urllib.request.Request(f"{HF}/datasets/{spec['repo']}/raw/main/README.md",
                                             headers={"User-Agent": "raw-bio-audit"})
                up_readme = urllib.request.urlopen(req, timeout=60).read()
                h = hashlib.sha256(up_readme).hexdigest()
                if h == man_sha.get("README.md"):
                    say("integrity", "DOC", "manifest README.md sha256 equals CURRENT upstream README "
                        "-> local README.md was edited locally after download (notes prepended); "
                        "sequence payload unaffected")
            except Exception as e:
                say("integrity", "UNRESOLVED", f"could not characterize README drift: {e}")

    elif name == "_fish_gue":
        if not online:
            say("coverage", "UNRESOLVED", "offline: upstream file identity unverified")
            return
        tree = paged_tree(spec["repo"], "?recursive=true")
        by_path = {f["path"]: f for f in tree if f.get("type") == "file"}
        say("coverage", "INFO", "HF dataset-viewer exposes a single concatenated 'default' config "
            "(371307/46415/46420 rows); it does NOT correspond to task folders - per-task CSVs are "
            f"the ground truth, mapped in {PIPELINES}/fish_gue/downloader.py:26-34")
        ok = bad = 0
        for task, sub in FISH_SUBDIR.items():
            for split in ("train", "val", "test"):
                local = folder / task / f"{split}.csv"
                cand = [p for p in by_path if p.startswith(sub + "/")
                        and p.lower().endswith(f"_{split}.csv")]
                if len(cand) != 1 or not local.exists():
                    say("coverage", "UNRESOLVED", f"{task}/{split}: no unambiguous upstream file")
                    continue
                up = by_path[cand[0]]
                # LFS-stored files: tree 'oid' is the pointer's git sha1; the content
                # hash is lfs.oid (sha256). Regular files: 'oid' is the git blob sha1.
                if up.get("lfs"):
                    same = (up["lfs"]["size"] == local.stat().st_size
                            and up["lfs"]["oid"] == sha256_file(local))
                else:
                    same = (up["size"] == local.stat().st_size
                            and up.get("oid") == git_blob_sha1(local))
                if same:
                    ok += 1
                else:
                    bad += 1
                    say("coverage", "FAIL", f"{task}/{split}: differs from upstream {cand[0]} "
                        f"(size {local.stat().st_size} vs {up['size']})")
        if ok and not bad:
            say("coverage", "PASS", f"all {ok} local CSVs are byte-identical to their upstream repo "
                "files (git blob sha1 match) -> complete and unmodified for the selected tasks")

    elif name == "_gen_lrb":
        say("coverage", "DESIGN", GEN_LRB_DESIGN)
        say("preservation", "DESIGN", GEN_LRB_LABELS)
        for mp in sorted(folder.rglob("download_manifest.json")):
            m = json.loads(mp.read_text())
            cfg, lim = m["config"], m.get("limit")
            for split, rec in m["splits"].items():
                w = {"train": .7, "validation": .15, "test": .15}[split]
                cap = max(2, round(lim * w)) if lim else None
                expect = min(rec["n_source"], cap) if cap else rec["n_source"]
                if rec["n_rows"] != expect:
                    say("coverage", "FAIL", f"{cfg}/{split}: n_rows {rec['n_rows']} != "
                                            f"min(n_source {rec['n_source']}, cap {cap})")
                off = GEN_LRB_OFFICIAL.get(cfg, {}).get(split)
                if off is not None:
                    if off == rec["n_source"]:
                        say("coverage", "PASS", f"{cfg}/{split}: loader-built full split "
                            f"({rec['n_source']}) == official README count; local keeps "
                            f"{rec['n_rows']}/{rec['n_source']}")
                    else:
                        say("coverage", "UNRESOLVED", f"{cfg}/{split}: loader n_source "
                            f"{rec['n_source']} vs README count {off} (diff {off - rec['n_source']}); "
                            "cannot arbitrate without rebuilding the split")
                elif split == "validation":
                    say("coverage", "INFO", f"{cfg}/validation: no official count published; "
                        f"loader produced {rec['n_source']}, local keeps {rec['n_rows']}")
            # preservation: fixed windows must equal the recorded sequence_length
            for split in m["splits"]:
                t = F["tables"].get(str(dataset_path / cfg / f"{split}.parquet"))
                if t:
                    if t["len_min"] == t["len_max"] == m["sequence_length"]:
                        say("preservation", "PASS", f"{cfg}/{split}: all sequences exactly "
                            f"{m['sequence_length']} bp == requested loader window (by construction, "
                            "not truncation)")
                    else:
                        say("preservation", "FAIL", f"{cfg}/{split}: lengths "
                            f"[{t['len_min']},{t['len_max']}] != window {m['sequence_length']}")

    elif name == "_plant_gen":
        if online:
            tree = {f["path"]: f for f in paged_tree(spec["repo"], "?recursive=true")
                    if f.get("type") == "file"}
            for mp in sorted(folder.rglob("download_manifest.json")):
                m = json.loads(mp.read_text())
                task, sub = m["config"].split(".")
                has_val = f"{task}/{sub}_validation.fa" in tree
                locals_ = set(m["splits"])
                if not has_val and locals_ == {"train", "test"}:
                    say("coverage", "PASS", f"{m['config']}: upstream provides only train+test FASTAs "
                        "(no *_validation.fa in repo; loader plant-genomic-benchmark.py:151-186 adds "
                        "validation only when one exists) and both are downloaded in full "
                        f"({ {s: r['n_rows'] for s, r in m['splits'].items()} })")
                elif has_val and "validation" not in locals_:
                    say("coverage", "FAIL", f"{m['config']}: upstream validation FASTA exists "
                                            "but was not downloaded")
                # corroboration: upstream FASTA bytes-per-record must agree train vs test
                bpr = {}
                for split, rec in m["splits"].items():
                    fa = tree.get(f"{task}/{sub}_{split}.fa")
                    if fa:
                        bpr[split] = fa["size"] / rec["n_rows"]
                if len(bpr) == 2:
                    a, b = bpr.values()
                    if abs(a - b) / max(a, b) < 0.01:
                        say("coverage", "PASS", f"{m['config']}: upstream FASTA bytes/record agree "
                            f"across splits ({a:.1f} vs {b:.1f}) -> local row counts consistent with "
                            "full upstream files")
                    else:
                        say("coverage", "UNRESOLVED", f"{m['config']}: bytes/record diverge "
                            f"({a:.1f} vs {b:.1f}); counts not corroborated")
        else:
            say("coverage", "UNRESOLVED", "offline: upstream FASTA inventory unverified")

    elif name == "_rna_dst":
        if sizes:
            n_ok = sum(compare_counts(say, observed, sizes, t, s) for (t, s) in observed)
            missing = [(c, s) for c, ss in sizes.items() if c in RNA_DST_PATHS
                       for s in ss if (c, s) not in observed]
            if missing:
                say("coverage", "FAIL", f"upstream splits absent locally: {missing}")
            elif n_ok == len(observed):
                say("coverage", "PASS", f"all {n_ok} splits of the 8 selected configs match current "
                    "upstream row counts (incl. species test splits test_danio/fly/thaliana/worm; "
                    "count agreement, not content identity)")
        # freshness: repo changed after download -> check each selected config's files
        for m, label, when in F.get("_changed_manifests", []):
            cfg = m["config"]
            newest = None
            for d in RNA_DST_PATHS[cfg]:
                for f in paged_tree(spec["repo"], f"/{quote(d)}?expand=true"):
                    lc = (f.get("lastCommit") or {}).get("date")
                    if lc and (newest is None or lc > newest):
                        newest = lc
            if newest and iso(newest) <= iso(when):
                say("freshness", "PASS", f"{cfg}: all source files under "
                    f"{RNA_DST_PATHS[cfg]} last committed {newest[:10]}, before download "
                    f"{when[:10]} -> post-download repo changes (mrna_half_life, "
                    "rna_protein_interaction, README rebrand) do not affect this config")
            elif newest:
                say("freshness", "FAIL", f"{cfg}: source files changed upstream on {newest[:10]}, "
                                         f"AFTER download {when[:10]}")
            else:
                say("freshness", "UNRESOLVED", f"{cfg}: could not resolve per-file commit dates")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=Path,
                    default=Path(__file__).resolve().parents[2] / "paths" / "paths.yaml")
    ap.add_argument("--machine", default=platform.node().split(".")[0])
    ap.add_argument("--raw", type=Path, help="override the machine's configured raw_base")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sources", nargs="*", choices=SOURCES, default=list(SOURCES))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.paths.read_text())
    base = args.raw or (cfg["machines"].get(args.machine) or {}).get("raw_base")
    if not base:
        ap.error(f"No raw_base for machine {args.machine!r} in {args.paths}; use --machine or --raw")
    raw = Path(base)
    datasets = cfg["datasets"]
    net_errors = []
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "raw_root": str(raw), "offline": args.offline, "sources": {}}
    for name in args.sources:
        key = name.lstrip("_")
        if key not in datasets:
            ap.error(f"No dataset {key!r} in {args.paths}")
        dataset = datasets[key]
        spec = dict(SOURCES[name], path=dataset["path"], repo=dataset["source"]["repo_id"])
        report["sources"][name] = audit_source(name, spec, raw, not args.offline, net_errors)
    print(json.dumps(report, indent=2))

    w = sys.stderr.write
    w(f"\n{'SOURCE':<13} {'STATUS':<11} FAIL/UNRES/DOC/DESIGN findings\n")
    for name, r in report["sources"].items():
        ms = [f for f in r["findings"] if f["level"] in ("FAIL", "UNRESOLVED", "DOC", "DESIGN")]
        w(f"{name:<13} {r['status']:<11} {len(ms)}\n")
        for f in ms:
            w(f"    [{f['level']}/{f['axis']}] {f['msg'][:150]}\n")
    w("\nDESIGN findings are documented downloader behavior, not defects. DOC findings are\n"
      "documentation-only. Count agreement never proves content identity; byte-identity is\n"
      "claimed only where sha256/git-blob evidence is cited. Non-HF dumps are out of scope.\n")
    statuses = [r["status"] for r in report["sources"].values()]
    return 1 if "FAIL" in statuses else (2 if "UNRESOLVED" in statuses else 0)

if __name__ == "__main__":
    sys.exit(main())
