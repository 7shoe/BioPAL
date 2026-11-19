# Fine-tuning

## TL;DR: 
Fine-tune a registered foundation model (bio LM) on a curated task via
- fixed hyperparameters, or 
- HPO model (hyper-parameter optimized; `hpo.py`), or
- OPAL (reactive optimizer), or
- OPAL + bio features (`--mode biopal --bio-policy fixed`), or
- BiOPAL (`--mode biopal --bio-policy expert`; see `biopal/README.md`).


## Run 
```bash
nohup finetune/samples.sh 1 > ~/logs/out.log 2>&1 &
```

## `finetune.py` — unified fine-tuning (model $\times$ task)

Registry-driven: models from `models/backbones/models.yaml` + the driving
flags in `backbone_flags.py`, tasks from `data/processed/tasks.yaml`, machine
paths from `paths/paths.yaml`. 

Backbone loading reuses the `models/backbones/smoke_backbone.py` helpers, so every quirk validated there
(NTv2 custom auto_map, encoder unwrapping, prot_t5 spaced residues, NTv3
hidden selection + pad-to-128, DNABERT-2 triton shim) applies unchanged.

Architecture: full-backbone fine-tuning (or `--freeze-backbone` for linear probe)
through a linear head on mask-aware mean-pooled hidden states. 

Classification via cross-entropy (accuracy/MCC/AUROC; multiclass AUROC is macro-OvR).
Regression through MSE on train-standardized targets (RMSE/MAE/R²/Spearman/Pearson
in original units). 

Fine-tuning recipe is generic
- AdamW at 2e-5, weight decay 0.01 on weight matrices only (`--optimizer sgd [--momentum 0.9]` for SGD)
- 10% linear warmup, then cosine decay
- gradient clipping at 1.0, FP32
- batch 16, 5 epochs 
Validation selects the best epoch.
Test runs once on that checkpoint after closing OPAL.


### Usage 
```bash
# list runnable models (models.yaml & backbone_flags.py)
conda run -n biopal-v2 python finetune.py --list
```

### Training
Each model requires a specific conda environment. 
```bash
# Plain AdamW baseline (no OPAL)
conda run -n biopal-v2 python finetune.py \
    --model ntv2_50m \
    --task nt_tasks_official_splice_sites_donors \
    --objective classification \
    --mode plain \
    --det strict \
    --out runs/ntv2_donors_plain_s42

# Tailored fine-tune 
conda run -n biopal-v2 python finetune.py \
      --model ntv2_50m \
      --task nt_tasks_official_splice_sites_donors \
      --objective classification \
      --mode plain \
      --freeze-backbone \
      --optimizer sgd \
      --momentum 0.9 \
      --lr 1e-2 \
      --epochs 1 \
      --det strict \
      --out runs/ntv2_donors_probe_sgd_s42

# OPAL "keep": the optimizer is wrapped by OPAL but OPAL never intervenes
conda run -n biopal-v2 python finetune.py \
  --model ntv2_50m \
  --task nt_tasks_official_splice_sites_donors \
  --objective classification \
  --mode keep \
  --det strict \
  --out runs/ntv2_donors_keep_s42

# OPAL: actual reactive optimizer
# - performance-oriented: OPAL_MODEL=claude-opus-5-5 OPAL_EFFORT=high, --budget 1
# - budget-friendly (below): OPAL_MODEL=claude-haiku-5-5 OPAL_EFFORT=low, --budget 0.2
# - cost scales with --decisions (default 5, hard cap; OPAL alone would ask ~50)
OPAL_MODEL=claude-haiku-5-5 OPAL_EFFORT=low \
  conda run -n biopal-v2 python finetune.py \
    --model ntv2_50m \
    --task nt_tasks_official_splice_sites_donors \
    --objective classification \
    --mode opal \
    --budget 0.2 \
    --out runs/ntv2_donors_opal_s42
```


Run under the env that serves the model (`backbone_flags.py` / `envs/README.md`);
a mismatch warns. Diagnostics: `--max-steps N` (no val/checkpoint/test),
`--limit N` (subsample splits), `--cpu`. `--det strict` is required for
`--compare` parity. Outputs per run dir: `config.json`, `train.jsonl`,
`validation.jsonl`, `best.pt` (backbone+head weights), `tokenizer/`,
`result.json` (recipe, versions, data/parameter hashes, metrics).

Validated 2026-11-18 (CPU smoke): NTv2 classification (diagnostic + tiny full
run with val/test), ESM2 regression, NTv3 pad-128 path, rnafm in bio-rna,
DNABERT-2 triton shim, keep-mode opal.wrap, and plain↔keep `--compare`
parity PASS under `--det strict`. Needs `scikit-learn`/`scipy` (added to both
env ymls 2026-11-18).

## `hpo.py` — hyperparameter optimization (RS or BO)

Optuna over `finetune.py --no-test` runs, scored by best validation loss. Searches
lr (log 1e-5..1e-3), weight decay (0..0.1), warmup fraction (0..0.2); trial 0 is
the default recipe. `--mode RS` = random search, `--mode BO` = Gaussian-process
BO (after max(3, n_trials // 4) random startup trials). Unknown flags pass
through to `finetune.py`. Rerunning the same command resumes `<out>/study.db`.

```bash
conda run -n biopal-v2 python hpo.py \
    --model ntv2_50m --task nt_tasks_official_splice_sites_donors \
    --mode BO --n_trials 12 --out runs/hpo_bo_ntv2_donors \
    --epochs 2
```

Outputs: `trials/trial_NNN/` (finetune runs), `logs/`, `hpo_result.json` (all
trials, best, and the `finetune.py` command that scores the winner once on test).

## To do

- OPAL with bio features; BiOPAL (agentic wrapper around the bilevel loop)
- `results/` aggregation across runs

Results land in each run's `--out` directory; cross-run aggregation (tables, figures) belongs to a future top-level `experiments/`.
