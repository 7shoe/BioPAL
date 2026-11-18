# Raw bio-sequence dataset audit — resolved findings

Audit of `/homes/csiebenschuh/data/bio_sequences/_raw/` (HF sources only), 2026-11-17.
Produced by `audit_raw_deep.py` (rewritten; formerly `audit_raw_bio_datasets.py`); machine-readable report in
[reports/raw_audit_resolved_v2.json](reports/raw_audit_resolved_v2.json) (earlier `raw_audit.json` / `raw_audit_resolved.json` iterations deleted).
Downloader code: `/home/csiebenschuh/Projects/BinarySignature/dsets/pipelines/`.

## Per-source summary

| Source | Local integrity | Selected-task completeness | Sampling / transformation | Freshness | Ready for curation |
|---|---|---|---|---|---|
| `_nt_tasks` | PASS — full read, 38,822 rows, rows/cols match manifest | **Test split only, by design** (downloader.py:12-16); test == upstream 38,822; official train (493,242) intentionally absent. Table spans all 18 benchmark tasks | None; variable native lengths 300–1000 bp, pure ACGT | Repo unchanged since 2025-06-30 | YES — as a test-only download; never present it as the full benchmark |
| `_cds` | PASS — full read of all 15 parquets (old audit's FAIL was a parser defect: `configs:{}` manifest schema unsupported; manifest n_rows nulls stem from a `skipped_existing` rerun) | **Complete**: 15/15 splits equal upstream counts (datasets-server); `ssp` config skipped by design (downloader.py:20-23) | None; CDS DNA (pure ACGT); fixed 714 bp (fluorescence) / 858 bp (β-lactamase), variable others | Repo unchanged since 2026-01-13 (predates download manifest check — no newer change) | YES |
| `_rfam` | PASS — `data.parquet` (726 MB, 20,051,822 rows) sha256 verified vs manifest **and equal to upstream LFS sha256** → byte-identical to upstream; full batched read clean | **Complete**: rows == upstream total | None; RNA families, lengths 19–10,799 (mean 121); alphabet ACGT/U + IUPAC ambiguity codes (KMNRSWY), all legitimate | Repo unchanged since 2025-05-20 | YES. The old FAIL was `README.md` only: manifest sha equals *current upstream* README → the local README was edited locally (planning notes prepended) ~10 min after download; documentation-only |
| `_fish_gue` | PASS — all 21 CSVs full-read, rows/cols match manifests | **Complete and byte-identical to upstream**: 19 files match git-blob sha1, 2 LFS files (CTCF/train, promoter/train) match LFS sha256. Task↔repo-subdir mapping from downloader.py:26-34 (e.g. CTCF→CTCF_TF, promoter→prom_1k_all); the HF viewer "default" config is an unrelated concatenation | None; fixed windows 300/800/1000 bp, ACGT+N | Repo unchanged since 2026-04-15 (before download) | YES |
| `_gen_lrb` | PASS — full read; n_rows = min(n_source, cap) exactly as coded | **Deliberate subsets** (downloader.py:104-114: shuffle seed=17, caps 7000/1500/1500 of `--limit 10000`). Loader-built full-split sizes (`n_source`) match the official README table for 5 of 6 known counts; enhancer/train differs by 10 (1,914,565 loader vs 1,914,575 README) — UNRESOLVED, likely README rounding/typo, unarbitrable without rebuilding | **Two transformations**: (1) train/val/test subsampling as above; (2) array labels (cage (len/128,50) bins; bulk 218-tissue vector) reduced to scalar nanmean (downloader.py:120-126) — raw labels are NOT upstream labels for cage/bulk. Sequences: every row exactly the requested window (2048/6144/16384 bp) → by construction, no truncation | Repo unchanged since 2024-06-21 | YES for the scalar-proxy contract; NOT a source for full-split or multi-target work. One minor UNRESOLVED (10-row README delta) |
| `_plant_gen` | PASS — full read of 6 parquets (5.1M rows), rows/cols match manifests | **Complete for the loader's official splits**: upstream ships only train+test FASTAs for the 3 selected configs (no `*_validation.fa` exists; loader adds validation only if present). FASTA bytes/record agree across splits (420.0 / 423.5±0.1) corroborating counts | None; fixed 400 bp (poly-A) / 398 bp (splicing), pure ACGT | Repo unchanged since 2024-06-02 | YES |
| `_rna_dst` | PASS — full read of 24 parquets, rows/cols match manifests | **Complete**: all 24 splits (incl. species tests test_danio/fly/thaliana/worm, preserved distinctly) equal current upstream counts | None; splice 400 bp fixed; MRL UTRs 25–100; ncRNA variable; TE 45–100; pure ACGT(+N in species tests) | Repo changed 2026-07-25 & 2026-09-09 **after** download, but per-file commit history proves every selected source file last changed 2024-11-14/22: changes were additive (`mrna_half_life/`, `rna_protein_interaction/`) + README rebrand. Selected data verified current | YES |

## Sequence preservation (global)

Full data-page reads of **26,139,671 rows across 67 payload files**: 0 null sequences,
0 empty sequences, 0 rows with symbols outside IUPAC nucleotide codes, 0 null/NaN targets.
Fixed-length distributions match documented windows (gen_lrb loader kwarg; AgroNT 400/398 bp;
splice 400 bp; Fish_GUE 300/800/1000 bp) — windows by construction, not truncation.
Caveat: for sources verified only by counts (nt test, cds, plant, rna_dst), row-count
agreement does not prove content identity; byte identity is proven only for rfam
`data.parquet` and all 21 Fish_GUE CSVs.

## Explicit answers

- **Complete for their selected tasks:** `_cds` (15/15 splits), `_rfam` (byte-identical),
  `_fish_gue` (byte-identical), `_plant_gen` (full loader splits), `_rna_dst` (24/24 splits),
  `_nt_tasks` *test split* (complete in itself).
- **Subsets / lacking official splits:** `_nt_tasks` (test only; no train — by design);
  `_gen_lrb` (train/val subsampled to ≤7000/1500/1500 with seed 17; test full for bulk only;
  labels mean-reduced for cage/bulk). `_plant_gen` has no validation split because upstream
  has none.
- **Sequence shortening:** no evidence anywhere; all fixed lengths are documented
  construction windows; variable-length sources retain native ranges.
- **Apparent failures that were audit defects:** `_cds` FAIL (unsupported `configs`
  manifest schema); `_rfam` FAIL (audit stopped at a documentation file and never checked
  the payload); Fish_GUE/gen_lrb/plant "mapping unverified" REVIEWs (audit relied on the
  dataset-viewer config model that doesn't apply); first re-run also mis-hashed 2 LFS files
  (fixed: LFS content hash is `lfs.oid` sha256, not git-blob sha1).
- **Unresolved before re-curation:** (1) gen_lrb enhancer/train 10-row delta between
  loader-built count and README table — cosmetic unless exact-official-split claims are
  needed; (2) content identity (vs count identity) for the count-verified sources —
  unobtainable without re-downloading; (3) nothing else. No blocker for re-curation under
  the documented design.

## Audit validation

`python3 audit_raw_deep.py > raw_audit_resolved_v2.json` run on the live raw tree:
exit 2 (= no FAILs, one UNRESOLVED: the enhancer 10-row delta). Runtime ≈4 min: sha256 of
726 MB rfam + 21 Fish_GUE files, 26.1M-row scan, metadata-only network (HF hub API,
datasets-server, raw README/loader text). `--offline` skips network and downgrades the
affected checks to UNRESOLVED. Supports all four manifest schemas, continues past individual
failures, checks freshness per manifest with per-file commit resolution, and separates
integrity / coverage / preservation / freshness with DESIGN/DOC/UNRESOLVED levels.
