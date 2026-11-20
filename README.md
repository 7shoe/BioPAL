# $\textsf{BiOPAL}$: $\textcolor{blue}{\textsf{Bi}}\textsf{level }\textcolor{blue}{\textsf{OP}}\textsf{timizer with an }\textcolor{blue}{\textsf{A}}\textsf{gent in the }\textcolor{blue}{\textsf{L}}\textsf{oop}$

Bilevel Optimizer with an Agent in the Loop for Sequence Prediction. 
Based on OPAL in [Reactive Optimizers](https://arxiv.org/abs/2610.33826) ([Bai et al., 2026](#references)).

## Status
<img src="./assets/under_construction.png" alt="Under construction" width="40%"> 

## Conda envs

Each model runs in exactly one env. Every env pins torch 2.9.0 and ships the
OPAL runtime (`opal-optimizer==0.3.10` + `bubblewrap`/`socat` from conda-forge);
check with `conda run -n <env> opal doctor`. Details and rationale: [`envs/README.md`](envs/README.md).

## Usage 
One small model on one task, five ways: `ntv2_50m` (50M parameters) on `nt_tasks_official_splice_sites_donors`
(does a 600-nt window hold a splice donor site?). Run from `finetune/` in the `biopal-v2` env.

```bash
cd finetune && conda activate biopal-v2
RUN="--model ntv2_50m --task nt_tasks_official_splice_sites_donors --epochs 2 --seed 42 --det strict"

# fine-tune only (AdamW, fixed recipe)
python finetune.py $RUN --mode plain --out runs/donors_plain

# HPO (via optuna): 12 trials of Bayesian optimization over lr, weight decay and warmup, picked on validation
python hpo.py --model ntv2_50m --task nt_tasks_official_splice_sites_donors --mode BO --n_trials 12 \
  --epochs 2 --det strict --out runs/donors_hpo

# OPAL (w/ haiku): an agent rewrites the update rule during training; --budget caps its spend in USD
# (A.) low effort
OPAL_MODEL=claude-haiku-5-5 OPAL_EFFORT=low python finetune.py $RUN --mode opal --budget 0.20 \
  --out runs/donors_opal_lowEff

# (B.) more eefort
OPAL_MODEL=claude-sonnet-5-5 OPAL_EFFORT=medium python finetune.py $RUN --mode opal --budget 0.20 \
  --out runs/donors_opal_medEff

# BiOPAL (w/ haiku): OPAL inside; at 5 gates a Claude expert proposes a cohort to weight or a layer to read,
# and a matched trial on private rows decides whether the proposal is adopted
# (A.) Low effort
OPAL_MODEL=claude-haiku-5-5 OPAL_EFFORT=low python finetune.py $RUN --mode biopal --bio-policy expert \
  --outer both --budget 0.50 --expert-model claude-haiku-5-5 --expert-effort low --expert-budget 0.20 \
  --out runs/donors_biopal_lowEff

# (B.) Medium effort
OPAL_MODEL=claude-sonnet-5-5 OPAL_EFFORT=medium python finetune.py $RUN --mode biopal --bio-policy expert \
  --outer both --budget 0.50 --expert-model claude-sonnet-5-5 --expert-effort medium --expert-budget 0.50 \
  --out runs/donors_biopal_medEff

```

(C.) BiOPAL with a local expert: the same, with the expert's seat taken by Qwen3-8B on this machine. It may read
a note from the task owner that no hosted model sees, and it needs no expert budget. This takes two terminals.
The server stays in the foreground of the first; start the run in the second once the server answers (about a
minute). A run started too early stops with `is the server up?` and leaves nothing behind.

```bash
# terminal 1: an env that has vLLM; the weights are already on disk
REV=b968826d9c46dd6066d109eabc6255188de91218
QWEN=/path/to/models--Qwen--Qwen3-8B/snapshots/$REV     # set this: the snapshot folder in a Hugging Face cache
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 vllm serve "$QWEN" \
  --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --max-model-len 8192 --dtype bfloat16 \
  --seed 0 --gpu-memory-utilization 0.25 --kv-cache-memory-bytes 2G --max-num-seqs 4   # 19 GB of a 96 GB card
```

```bash
# terminal 2: in finetune/, biopal-v2 env, RUN set as above
REV=b968826d9c46dd6066d109eabc6255188de91218
OPAL_MODEL=claude-haiku-5-5 OPAL_EFFORT=low python finetune.py $RUN --mode biopal --bio-policy expert \
  --outer both --budget 0.20 --expert-backend local --expert-model Qwen/Qwen3-8B --expert-revision $REV \
  --expert-request '{"chat_template_kwargs": {"enable_thinking": false}}' \
  --expert-note ../biopal/notes/nt_tasks_official_splice_sites_donors.txt --out runs/donors_biopal_local
```

Each run writes `result.json` (metrics) to its `--out` folder; `python -m biopal finetune/runs/donors_biopal`
(from the repo root) prints what happened at a gate. A BiOPAL run holds 15% of train out for its gates during
the first 60% of the run and trains on those rows afterwards: it makes as many updates as the plain run and
visits every train row as often (`--panel-until`). For the matched baseline on the same batches, for
`--mode keep` (no Claude call at all), the
note format and the replay tool, see [`biopal/README.md`](biopal/README.md); for the other flags
[`finetune/README.md`](finetune/README.md).

### `biopal-v2` — the main env

```bash
conda env create -f envs/biopal-v2.yml
conda run -n biopal-v2 python -m pytest tests/biopal_v2 --cpu -v
```

transformers 4.57.1. Models:

- DNA: `ntv2_50m`, `ntv3_pretrained`, `ntv3_100m_pre`, `hyenadna_tiny_1k`, `dnabert2_117m`
- Protein: `esm2_t6_8m`, `esm2_t12_35m`, `esm2_t30_150m`, `esm2_t33_650m`, `ankh_base`, `prot_t5_xl_half`

`dnabert2_117m` needs the triton shim (`PYTHONPATH=envs/shims/no_triton`);
NTv3 inputs must be padded to a multiple of 128.

### `bio-rna`

```bash
conda env create -f envs/bio-rna.yml
conda run -n bio-rna python -m pytest tests/bio_rna --cpu -v
```

transformers **5.9.0** (required by multimolecule 0.2.3). Models:

- RNA: `rnafm`, `splicebert`

### `bio-mamba` — deferred (not created)

Recipe staged in `envs/bio-mamba.yml` (two-stage install, CUDA-only) and
`tests/bio_mamba`. Models: `caduceus_ps_131k`, `plant_caduceus_l20`.

### Not served

`gena_lm_bert_base` is retired (needs transformers ≤ 4.44).

## References

```bash
@article{bai2026reactive,
  title={Reactive Optimizers},
  author={Bai, Xingjian and Liang, Jia-Hao and Teoh, Jayden and Sitzmann, Vincent},
  journal={arXiv preprint arXiv:2610.33826},
  year={2026}
}
```
