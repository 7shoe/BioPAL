
# Raw Data

Raw datasets are sourced from HuggingFace or open web corpora. Examples include [ProteinGym](https://huggingface.co/datasets/ICML2022/ProteinGym), [siRNAEfficacyDB](https://cellknowledge.com.cn/siRNAEfficacy/), [FishGUE](https://huggingface.co/datasets/bioinfoihb/Fish_GUE), [plant_gen](https://huggingface.co/datasets/InstaDeepAI/plant-genomic-benchmark), among others. 

Code in this directory scans datases on disk to check availability and audit.

## Data
The data paths depend on the machien and file system. See [Path Management](#path-management) below. 
 
<details>
<summary><b>lambda13</b></summary>
The CELS machine was primarily used for the experiments.

```bash
# raw datasets
(/homes/csiebenschuh/data/bio_sequences/_raw/

# processed
/homes/csiebenschuh/data/bio_sequences/
```
 
</details>
<details>
<summary><b>Aurora</b></summary>
Datasets not present on the Lustre file system as of now.
 
</details>


## Overview

Check sample size overview by raw dataset.

```bash
python overview.py
```

## Suppl. Downloads
Downloads primarily through `~/Projects/BinarySignature` as these bio sequence datasets are still up to date and audited.
Additional download of [Nucleotide Transformer Downstream Tasks](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks) as previous work exclusively relied on held-out test set. See `~./downloads`

## Routine (quic) audit of raw datasets
Meta data audit

```bash
python audit_raw.py.py --machine lambda13
```
- fast health check of every raw dataset listed in `paths.yaml`
- verifies presence of each dataset (incl. expected files, matches manifests w/ recorded size, and that parquet row counts match the manifest)
- checks metadata only — no data is read
- run before curation or training runs

## In-depth (slow) audit of raw datasets

```bash
python audit_raw_deep.py > reports/raw_audit.json
```
- forensic, one-shot verification
- full batched reads of all ~26M rows
- run sparingly: after (re-)downloading a source, adding a new one
- human-readable conclusions of the 2026-11-17 audit live in `RAW_AUDIT_FINDINGS.md`


## Path Management
Point to raw datasets, curated tasks and model caches. Currently lambda13.
Shared config at repo root:
```
paths/paths.yaml
```
Both `overview.py` and `audit_raw_deep.py` use the short hostname to select
`machines.<machine>.raw_base` and append each dataset's configured `path`.
Use `--machine NAME` to select another machine or `--paths FILE` to load a
different registry. For a relocated copy of the data, the deep audit also
accepts `--raw DIR` to override the raw root.
