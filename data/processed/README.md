# Task Curation
Data tasks were curated with `BinarySignature` pipelines on lambda13. Regardless, validation and verification of the curated tasks are given through audit scripts etc.

## Current
Checks the curated downstream data
```bash
python audit_tasks.py --yaml tasks.yaml --machine lambda13
```

## Previous Task Curation 
See the `BinarySignature` repository in 

```bash
~./Projects/BinarySignature/dsets/pipelines/<source>/curation.py
```

## Task Registry
Downstream tasks curated from the `raw` data that pass the test, are listed in the task registry YAML.

```bash
~/processed/tasks.yaml
```


## Supplementary Task Curation
In line with the dedicated download of NT tasks, in `raw`, the task curation pipeline runs in 
```bash
curate_nt_official.py
```

