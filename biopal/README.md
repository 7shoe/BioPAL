# BiOPAL

Fine-tuning on two levels. OPAL decides how the parameters are updated. At a few checkpoints ("gates") a
biological expert reads measurements of the model and may propose one change to what is trained: more loss
weight on one cohort of rows, or another block feeding the task head. A matched trial on private rows decides
whether the change is adopted. Everything runs through `finetune/finetune.py`.

| File | Job |
| --- | --- |
| `panel.py` | split train into fit F, diagnostic D and private A; cohorts; names shown to the expert |
| `measure.py` | baseline, generalization and layer gaps per cohort, with 95% intervals; the layer probes |
| `outer.py` | gate schedule, snapshot and rewind, the two actions, the trial, the adoption rule, `outer.jsonl` |
| `expert.py` | the brief, the Claude call, answer checks, budget; scripted answers for offline runs |
| `ledger.py` | cumulative Claude spend across runs |
| `__main__.py` | `python -m biopal RUN_DIR [GATE]` prints one gate as a readable trace |

## Arms

All arms of a comparison share `--panel-frac 0.15` and the seed, so they train on the same 85% of train.

| Arm | Flags |
| --- | --- |
| A, AdamW | `--mode plain --panel-frac 0.15` |
| B, OPAL | `--mode biopal --bio-policy off` |
| C, OPAL + bio features | `--mode biopal --bio-policy fixed` |
| F, BiOPAL | `--mode biopal --bio-policy expert --outer both` (add `--expert-shadow masked` for the same-checkpoint masked comparison) |

`--mode keep` in place of `--mode biopal` wraps the optimizer in OPAL without an agent: no OPAL cost, same
schedule, same gates. With `--expert-script FILE` or `--outer-audit` it makes no Claude call at all.

## Smoke runs (cents)

Run from `finetune/` in the `biopal-v2` env. Both use the cheapest model at low effort and a shared ledger, so
the two budgets are ceilings over every run that names the same ledger file.

```bash
# offline: gates, trial and rewind with a scripted expert that keeps; no Claude call
echo '{"informed": []}' > /tmp/keep.json
python finetune.py --model ntv2_50m --task nt_tasks_official_splice_sites_donors --limit 6000 --epochs 1 \
  --det strict --mode keep --panel-frac 0.15 --bio-policy expert --expert-script /tmp/keep.json \
  --outer-decisions 1 --outer-steps 32 --outer-audit 16 32 --no-test --out runs/smoke_offline

# live: 3 OPAL decisions, 2 gates with an informed and a masked expert (about $0.012 in total)
OPAL_MODEL=claude-haiku-5-5 OPAL_EFFORT=low python finetune.py --model esm2_t12_35m \
  --task proteingym_gfp_fluorescence --limit 8000 --epochs 1 --det strict --mode biopal --bio-policy expert \
  --outer both --expert-shadow masked --decisions 3 --outer-decisions 2 --outer-steps 32 \
  --expert-model claude-haiku-5-5 --expert-effort low --budget 0.20 --expert-budget 0.20 \
  --ledger runs/smoke_ledger.jsonl --out runs/smoke_live
(cd .. && python -m biopal finetune/runs/smoke_live)   # the trace of the first gate with a trial
```

## Research pilots

Full data, one seed, default gate settings (5 OPAL decisions, 5 gates, 64-update trials). Set the three
placeholders first; the OPAL model and effort are read from the environment (`finetune.py` defaults to
`claude-opus-5-5` at high effort when they are unset).

```bash
export OPAL_MODEL=<model> OPAL_EFFORT=<effort>; OPAL_USD=<cap>; EXPERT_USD=<cap>
D="--model ntv2_50m --task nt_tasks_official_splice_sites_donors --epochs 2 --seed 42 --panel-frac 0.15 --det strict"
G="--model esm2_t12_35m --task proteingym_gfp_fluorescence --epochs 1 --seed 42 --panel-frac 0.15 --det strict"
F="--mode biopal --bio-policy expert --outer both --expert-shadow masked --expert-budget $EXPERT_USD"

python finetune.py $D --mode plain                                         --out runs/pilot/donors_A
python finetune.py $D --mode biopal --bio-policy off   --budget $OPAL_USD  --out runs/pilot/donors_B
python finetune.py $D --mode biopal --bio-policy fixed --budget $OPAL_USD  --out runs/pilot/donors_C
python finetune.py $D $F                               --budget $OPAL_USD  --out runs/pilot/donors_F
python finetune.py $G --mode biopal --bio-policy off   --budget $OPAL_USD  --out runs/pilot/gfp_B
python finetune.py $G $F                               --budget $OPAL_USD  --out runs/pilot/gfp_F

# free, before spending: how large are trial differences against their noise, and which horizon is enough?
python finetune.py $D --mode keep --outer-audit 16 32 64 128               --out runs/pilot/donors_audit
```

Budgets are per run. Add `--ledger runs/pilot/ledger.jsonl` to every command to make them ceilings over the
whole pilot. `--expert-model` and `--expert-effort` default to `claude-haiku-5-5` and `low`.

`--det strict` belongs in every arm. A gate rewinds to a snapshot and trains on from it, and that only repeats bit
for bit when the GPU arithmetic does. With the default `--det warn`, ESM2's attention kernel is not deterministic:
on full GFP the same configuration run twice from one snapshot differed by 1e-7 in the training loss after 9 to 94
updates and by up to 0.15 in the private loss after 128. NTv2 repeated exactly in both modes.

## What a run writes

- `outer.jsonl`: one line per event. `schedule` (OPAL decisions and gates as placed), `shown_to_opal` (what OPAL
  was shown before each decision), and per gate `table` (the measurements and the actions on offer), `proposals`
  (written and synced before the private set is read), `trial` (per branch: training losses and, per horizon,
  the mean loss difference on A, its standard error and whether it passes; for a readout branch also
  `install_error`, the largest per-row difference on D between the installed head and the probe it came from),
  `decision`, and `bypassed` (after a readout switch: the tensors that got no gradient did not move).
- `expert/gateN_ROLE_brief.txt` and `_reply.json`: what each expert read and what the CLI returned.
- `result.json`, key `biopal`: adopted changes, final configuration, Claude spend by role, OPAL agent calls inside
  trial branches (must be 0), and the readout block `best.pt` was saved with.
- `config.json`, `recipe.panel`: sizes and hashes of F, D and A. Equal hashes mean equal rows across arms.

## Tests

```bash
conda run -n biopal-v2 python -m pytest tests/biopal -q      # CPU, about 40 s, no model download, no Claude call
```

## Known limits

- OPAL copies the training script into its agent's workspace. The agent can therefore read that an outer level
  exists; it is not shown the cohort in force, the expert's text or any trial score.
- The value passed to `optimizer.step(loss=...)` is the unweighted loss in every arm. A program of OPAL's that
  reads `ctx.loss` gets that value while its gradients come from the weighted loss; `loss_aware_updater` in each
  `decision` record flags such runs.
- The rewind clears OPAL's resume note through a private attribute of OPAL 0.3.10 (`_resume_note`); a test fails
  if a later version changes that.
- After a readout switch the optimizer keeps its statistics from before the switch.
- A is read at every gate, so it becomes tuning data over a run. Validation and test are never touched by a gate.
- The adoption rule's standard error counts only the sampling of rows in A, not how much a short training
  window varies from one run to the next. In the full-data audits (`--outer-audit 16 32 64 128`, keep backend) 25
  to 60% of all cohort options passed at any one horizon, the sign of the difference often flipped between
  horizons, and the differences at two horizons were uncorrelated. A pass in a single trial is therefore weak
  evidence on these two tasks; see the audit before trusting an adoption.
