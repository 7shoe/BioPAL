# Deduplication

In response to the [PR](https://github.com/7shoe/BioPAL/issues/3), a pragmatic check on duplicates.

Checks every task in `data/processed/tasks.yaml` for input sequences that are duplicated, exactly or nearly, between
`train`, `val` and `test`. Labels are ignored.

```bash
conda run -n biopal-v2 python duplicate.py                 # all registry tasks, writes the two files below
conda run -n biopal-v2 python duplicate.py --tasks sirna   # a subset, prints only
```

| File | Content |
| --- | --- |
| `removed.tsv` | `task, split, series_id, via`: the rows that training leaves out |
| `report.tsv` | per task: split sizes, rows removed, whether the task is exempt and why |

## Rule

- `test` is never touched. A `val` row is removed when it duplicates a `test` sequence; a `train` row when it
  duplicates a `val` or `test` sequence.
- Near-duplicate: a window covering half of one sequence (start, middle or end; at least 16 nucleotides or
  8 residues) occurs letter for letter in the other. This catches shifted or contained windows and variants
  with a few substitutions. For DNA the reverse complement counts as the same sequence.
- Tasks whose sequences resemble each other by construction (mutational scans, RNA families, short oligos;
  the `EXEMPT` table in `duplicate.py`) get the exact check only.

## Examption

Exempt (35 tasks, exact check only)
- 22 ProteinGym tasks, cds_beta_lactamase_*, cds_fluorescence, cds_stability, rfam_*.
- Added by my judgment: rna_dst_ncrna_family_* (family task, like rfam), cripsrCas9 and sirna (short sequences whose curation already applied its own distance-based control).

## Effect

The curated parquet files are not modified. `finetune/finetune.py` reads `removed.tsv` and skips the listed
rows; `config.json` of a run records how many (`removed_duplicates`). Rerun `duplicate.py` after a task is
re-curated or added to the registry, otherwise `finetune.py` stops with a message.

## Result


Around <6K train rows and <200 val rows for removal across 22 tasks, and the trainer now skips them.
**The curated parquet files themselves are untouched.**

| Tasks | Train removed | Val removed |
|---|---:|---:|
| `rna_dst_translation_efficiency_*` (3) | 9.5–12.2% | 4–7% |
| `fish_gue_pou5f1` | 5.2% | 2.7% |
| `rna_dst_splice_site_*` (2) | 4.1–4.7% | 0% |
| `plant_gen_*` (3) | 2.1–3.0% | 0% |
| `fish_gue_promoter` | 2.4% | 1.5% |
| Other 15 filtered tasks | ≤ 1% | ≤ 0.4% |