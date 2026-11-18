# Models
Various biological language models (bio-FMs) are supported through several conda envs (`biopal-v2`, `bio-rna`).

## Supported models

The `lambda13` audit found local weights for 16 of 19 backbones.

| Modality | With local weights | Backbones with local weights | Missing local weights |
|---|---:|---|---|
| DNA | 8 of 9 | NTv2-50M Multi-Species, `NTv3_pretrained`, `NTv3_100M_pre`, GENA-LM BERT Base, HyenaDNA Tiny 1K, Caduceus-PS 131K, PlantCaduceus L20, DNABERT-2-117M | `InstaDeepAI/ntv3_base_model` |
| RNA | 2 of 3 | RNA-FM, SpliceBERT | `multimolecule/rinalmo-micro` (local snapshot also behind upstream) |
| Protein | 6 of 7 | ESM-2 (8M, 35M, 150M, 650M), Ankh Base, ProtT5-XL UniRef50 half-precision encoder | `ElnaggarLab/ankh-base-encoder` |

## Download directories

Models sit in the Huggingface cache. They were downloaded locally and matches its pinned version.

```bash
# lambda13
/homes/csiebenschuh/data/bio_sequences/ in three HF-style caches
```

The model paths are set in 
```bash
paths/paths.yaml
```

- cache also carries its own metadata: `lm_model_registry.json`, `lm_model_groups.json`, and smoke-test reports 
- smoke ran on 05-22-2026 (host lambda13, torch 2.8.0+cu126, CUDA verified)


## Audit of models
The foundation models are audited. 

```bash
python models/backbones/audit_models.py \
      --machine lambda13 \
      --yaml models/backbones/models.yaml
```


## Smoke test

### Smoke-test all models
Smoke test all models. The script automatically identified and activates the correct conda environment.

```bash
# use GPU 1 only
models/backbones/smoke_sweep.sh 1
```
Below the output 
```bash
=== summary -> models/backbones/model_smoke.yaml ===
  ntv2_50m PASS
  ntv3_pretrained PASS
  ntv3_100m_pre PASS
  hyenadna_tiny_1k PASS
  dnabert2_117m PASS
  esm2_t6_8m PASS
  esm2_t12_35m PASS
  esm2_t30_150m PASS
  esm2_t33_650m PASS
  ankh_base PASS
  prot_t5_xl_half PASS
  rnafm PASS
  splicebert PASS
  caduceus_ps_131k SKIP
  plant_caduceus_l20 SKIP
```

### Smoke-test a single model

  Offline load at the pinned revision, then one forward + backward pass on GPU.
  Prints `PASS {json}` (hidden size, params, token counts, peak GPU memory);
  any failed check raises instead. The script always uses `cuda:0`, so choose
  the physical GPU with `CUDA_VISIBLE_DEVICES`:

  ```bash
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 conda run -n biopal-v2 --no-capture-output \
    python models/backbones/smoke_backbone.py \
    --repo facebook/esm2_t6_8M_UR50D --revision c731040fcd8d73dceaa04b0a8e6329b345b0f5df \
    --modality protein
  ```