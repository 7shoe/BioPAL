# Tests, split by conda env

Each subdirectory holds the tests that must pass inside one conda env (a test
run mixes envs never — models only import in the env that serves them, see
`envs/README.md`).

| Directory | Env | Models |
|---|---|---|
| `biopal_v2/` | `biopal-v2` | ntv2_50m, ntv3 ×2, hyenadna, esm2 ×4, ankh_base, prot_t5, dnabert2 |
| `bio_rna/` | `bio-rna` | rnafm, splicebert (skips cleanly outside the env) |
| `bio_mamba/` | `bio-mamba` | caduceus_ps_131k, plant_caduceus_l20 (skips cleanly outside the env) |

Run: `conda run -n <env> python -m pytest tests/<dir> --cpu -v`. Shared
fixtures live in `tests/conftest.py`, the per-model checks in
`tests/backbone_checks.py`; each env dir only has its `model_specs.py` and two
thin test files.

Two test families per env, both parametrized over the env's model set
(`biopal_v2/model_specs.py`); models missing from the registry skip, so the
tests track `audit_models.py` output. Model loading/encoding goes through the
same helpers as the smoke tool (`models/backbones/smoke_backbone.py`), so
tests and smoke can't drift apart.

- **`test_backbones.py` — load + forward**: every model the env is supposed
  to serve loads **offline** from its pinned cache snapshot (registry:
  `models/backbones/models.yaml`, cache roots: `paths/paths.yaml`) and
  forwards a batch of synthetic sequences to finite hidden states. The
  synthetic batch lives in `sample_data/` (deterministic, seed 17): `dna.txt`
  is 4×384 nt (384 is a multiple of 128, as NTv3's U-Net requires),
  `protein.txt` is 4×192 aa starting with M, like the translated-CDS tasks.
- **`test_training.py` — trainability**: 3 AdamW steps (lr 5e-5) on a
  balanced 4-sample batch of a real curated task — splice-site donors (DNA,
  truncated to 128 nt) and CDS fluorescence (protein, 96 aa), falling back to
  `sample_data/` where the data store is absent — through a linear probe on
  mean-pooled hidden states, fine-tuning the whole backbone. Asserts finite
  losses, non-zero backbone gradients, parameters moving, and the loss
  decreasing while overfitting the fixed batch.

## Run Tests

```bash
# GPU if available, CPU otherwise:
conda run -n biopal-v2 python -m pytest tests/biopal_v2 -v
# force CPU (e.g. GPUs busy or absent):
conda run -n biopal-v2 python -m pytest tests/biopal_v2 --cpu -v
# subset by model name:
conda run -n biopal-v2 python -m pytest tests/biopal_v2 --cpu -v -k "ntv2 or esm2_t6"
```

On CPU, half-precision checkpoints are cast to float32 automatically; the
large models (prot_t5 1.2B, NTv3 703M, esm2 650M) dominate the runtime.