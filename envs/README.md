# Environment plan (backbone serving)

Each model in `models/backbones/models.yaml` is served by one conda env.
Smoke-test any model with `models/backbones/smoke_backbone.py` (offline, pinned
revision, GPU forward + backward + recorded facts).

| Env | Status | Serves | Key pins |
|---|---|---|---|
| `biopal-v2` | ready | ntv2_50m, ntv3_pretrained, ntv3_100m_pre, hyenadna_tiny_1k, dnabert2_117m (triton shim), esm2 t6/t12/t30/t33, ankh_base, prot_t5_xl_half | torch 2.9.0, transformers 4.57.1, sentencepiece + protobuf (T5 tokenizers), einops, opal-optimizer 0.3.10 |
| `bio-rna` | ready | rnafm, splicebert | torch 2.9.0, **transformers 5.9.0**, multimolecule 0.2.3, opal-optimizer 0.3.10 |
| `bio-mamba` | **deferred** — not created; yml + `tests/bio_mamba` staged | caduceus_ps_131k, plant_caduceus_l20 | torch 2.9.0, transformers 4.57.1, mamba-ssm + causal-conv1d (CUDA-only, stage-2 install), opal-optimizer 0.3.10 |

```bash
conda env create -f envs/biopal-v2.yml
conda env create -f envs/bio-rna.yml
conda env create -f envs/bio-mamba.yml   # deferred; see stage 2 below
# verify OPAL + torch/claude readiness in each:
conda run -n biopal-v2 opal doctor
conda run -n bio-rna opal doctor
```

## Models 

| Conda env | Model | Params |
  |---|---|---|
  | `biopal-v2` | [InstaDeepAI/nucleotide-transformer-v2-50m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species) | 53.5M |
  | `biopal-v2` | [LongSafari/hyenadna-tiny-1k-seqlen-hf](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-hf) | 0.44M |
  | `biopal-v2` | [facebook/esm2_t6_8M_UR50D](https://huggingface.co/facebook/esm2_t6_8M_UR50D) | 7.5M |
  | `biopal-v2` | [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D) | 33.5M |
  | `biopal-v2` | [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D) | 148M |
  | `biopal-v2` | [facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) | 651M |
  | `biopal-v2` | [ElnaggarLab/ankh-base](https://huggingface.co/ElnaggarLab/ankh-base) | 453M (encoder) |
  | `biopal-v2` | [Rostlab/prot_t5_xl_half_uniref50-enc](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc) | 1.21B (encoder, fp16) |

## Notes

- **OPAL runtime, every env**: `opal-optimizer` (pip) plus `bubblewrap` +
  `socat` (conda-forge; no sudo on lambda13). Without them opal's agent falls
  back to file tools only. Check with `opal doctor`.
- **Keep opal in the torch resolve**: opal declares only `torch>=2.5`, so a
  standalone install can silently upgrade torch. Upgrade with the pins restated:
  `pip install -U opal-optimizer torch==2.9.0 transformers==<env pin>`.
- **mamba-ssm (stage 2)**: its build imports torch, so it is not in the yml.
  Inside bio-mamba, run
  `pip install --no-build-isolation causal-conv1d==1.7.0 mamba-ssm==2.3.2.post1`
  (fetches prebuilt cu12/torch2.9/cp311 wheels; falls back to nvcc 12.4).
- **DNABERT-2 vs triton**: its `flash_attn_triton.py` breaks on modern triton.
  Run it with `PYTHONPATH=envs/shims/no_triton` (standard attention fallback).
  Apply per process, not per env (mamba-ssm needs triton).
- **transformers pins differ**: 4.57.1 in biopal-v2/bio-mamba (remote-code
  vintage), 5.9.0 in bio-rna (multimolecule 0.2.3 imports the v5-only
  `transformers.initialization`; pip won't catch this at install time).
- **Licenses**: multimolecule models are AGPL-3.0; Agro/NT models CC-BY-NC-SA.

## biopal-v2 validation sweep

All models at their registry revisions under transformers 4.57.1.
`smoke_backbone.py` handles the per-model quirks: NTv2 loads via its custom
AutoModelForMaskedLM, ankh_base via encoder unwrap, prot_t5_xl_half via the
slow tokenizer with space-separated residues, and dnabert2_117m needs the
triton shim. `gena_lm_bert_base` is retired (its frozen custom code needs
transformers ≤ 4.44); dnabert2 covers its niche.

```bash
for spec in \
  "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species|81b29e5786726d891dbf929404ef20adca5b36f1|dna|-t" \
  "LongSafari/hyenadna-tiny-1k-seqlen-hf|e8c1effa8673814e257e627d2e1eda9ea5a373f6|dna|-t" \
  "facebook/esm2_t6_8M_UR50D|c731040fcd8d73dceaa04b0a8e6329b345b0f5df|protein|" \
  "facebook/esm2_t12_35M_UR50D|6fbf070e65b0b7291e7bbcd451118c216cff79d8|protein|" \
  "facebook/esm2_t30_150M_UR50D|a695f6045e2e32885fa60af20c13cb35398ce30c|protein|" \
  "facebook/esm2_t33_650M_UR50D|08e4846e537177426273712802403f7ba8261b6c|protein|" \
  "ElnaggarLab/ankh-base|d99cb6b966530dfc2ae96bc69d9255c2a07308b0|protein|" \
  "Rostlab/prot_t5_xl_half_uniref50-enc|94a6abc029ae13029317b140b7424e012bf8dfbf|protein|--cache /homes/csiebenschuh/data/bio_sequences/_protein_lm_cache" \
; do
  IFS='|' read -r repo rev mod extra <<< "$spec"
  [ "$extra" = "-t" ] && extra="--trust-remote-code"
  echo "=== $repo ==="
  HF_HUB_OFFLINE=1 conda run -n biopal-v2 --no-capture-output \
    python models/backbones/smoke_backbone.py \
    --repo "$repo" --revision "$rev" --modality "$mod" $extra
done

# DNABERT-2 (triton shim):
HF_HUB_OFFLINE=1 PYTHONPATH=envs/shims/no_triton conda run -n biopal-v2 --no-capture-output \
  python models/backbones/smoke_backbone.py \
  --repo zhihan1996/DNABERT-2-117M --revision 7bce263b15377fc15361f52cfab88f8b586abda0 \
  --modality dna --trust-remote-code
```

prot_t5_xl_half lives only in `_protein_lm_cache`, hence the `--cache` override.

## NTv3

Three repos pinned into `_hf_lm_cache` (jax weights excluded; ~3.1 GB), all
code-reviewed at the pinned shas (pure model/tokenizer definitions, no
network/subprocess/writes) and registered in `paths/paths.yaml`:

- `ntv3_pretrained` = InstaDeepAI/NTv3_pretrained @ fd58207244d8 — flagship,
  703M params, hidden 1024. Self-contained custom code.
- `ntv3_100m_pre` = InstaDeepAI/NTv3_100M_pre @ fee14f6adbba — 106M params,
  hidden 768. Its auto_map resolves classes **cross-repo** from
  ntv3_base_model, which must therefore stay cached for offline loads.
- `ntv3_base_model` @ 27682ed2f537 — code-only companion (expect_weights:
  false; WARNs in the audit by design).

Both serve from **biopal-v2** (plain torch/transformers imports); smoke-test
with `--lengths 128 256`. Quirks for fine-tuning: char-level
tokenizer with NO special tokens (1 nt = 1 token); **input length must be a
multiple of 128** (U-Net downsampling), so curated-task sequences need padding
to the next 128 multiple; the forward returns logits first — take
`out.embedding` (pretrained) or `hidden_states[-1]` with
`output_hidden_states=True` (100M), as smoke_backbone.py now does.

## bio-mamba smoke tests (once the env exists)

```bash
HF_HUB_OFFLINE=1 conda run -n bio-mamba python models/backbones/smoke_backbone.py \
  --repo kuleshov-group/PlantCaduceus_l20 --revision d1b3c7d42abacce3125f0119adcec161fd8d59bd \
  --modality dna --trust-remote-code
HF_HUB_OFFLINE=1 conda run -n bio-mamba python models/backbones/smoke_backbone.py \
  --repo kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 \
  --revision d89eeb853136ea64da7feb3d0c8e909771b17ae6 --modality dna --trust-remote-code --long 16384
```
