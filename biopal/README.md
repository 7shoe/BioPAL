# BiOPAL: Core Piece of Bilevel Optimizer Agent in the Loop

**TL;DR**: BiOPAL is an agentic fine-tuning framework for biological sequence tasks. It builds on [OPAL/reactive optimizers](https://arxiv.org/abs/2610.33826v2) for the inner optimization loop and a domain-expert for the outer loop.

Fine-tuning occurs on two levels. OPAL decides how the parameters are updated. At a few checkpoints (so-called "gates") a
biological expert reads measurements of the model and may propose one change to what is trained: more loss
weight on one cohort of rows, or another read-out layer feeding the task head. A matched trial on private rows decides
whether the change is adopted. Everything runs through `finetune/finetune.py`.

| File | Job |
| --- | --- |
| `panel.py` | split train into fit F, diagnostic D and private A; the batches of a run in which D and A return to training; cohorts; names shown to the expert |
| `measure.py` | baseline, generalization and layer gaps per cohort, with 95% intervals; the layer probes |
| `outer.py` | gate schedule, snapshot and rewind, the two actions, the trial, the adoption rule, `outer.jsonl` |
| `expert.py` | the brief, answer checks; the Claude call and its budget, the local model call and its note, scripted answers |
| `ledger.py` | cumulative Claude spend across runs |
| `replay.py` | `python -m biopal.replay RUN_DIR ... --out DIR` asks other proposers at a run's saved gates and joins their choices to the recorded trials |
| `__main__.py` | `python -m biopal RUN_DIR [GATE]` prints one gate as a readable trace |

## Training Modes

All modes of a comparison share `--panel-frac 0.15`, `--panel-until` and the seed, so they train on the same
rows in the same batches.

| Mode | Flags |
| --- | --- |
| A, AdamW | `--mode plain --panel-frac 0.15` |
| B, OPAL | `--mode biopal --bio-policy off` |
| C, OPAL + bio features | `--mode biopal --bio-policy fixed` |
| F, BiOPAL | `--mode biopal --bio-policy expert --outer both` (add `--expert-shadow masked` for the same-checkpoint masked comparison) |

`--mode keep` in place of `--mode biopal` wraps the optimizer in OPAL without an agent: no OPAL cost, same
schedule, same gates. With `--expert-script FILE` or `--outer-audit` it makes no Claude call at all.

### The held-out rows return to training

The gates need rows that give no gradient: D for the measurements, A to score a trial. They are needed only
while gates are open. With `--panel-until 0.6` (the default) a run with a panel has as many updates as one
without, `epochs * ceil(train rows / batch size)`, in three parts:

1. Updates 1 to R train on F only. R is 60% of the run, moved up to an OPAL check; it depends on nothing a gate
   does. All gates and their trials lie here: the last trial ends by update R.
2. After update R every row is visited as often as it still lacks to `epochs` visits: D and A rows `epochs`
   times, F rows the rest of theirs, in one shuffled order.
3. At the last update every train row has had `epochs` visits, as in a run without a panel.

Both agents are told: the expert's brief and OPAL's goal name update R. OPAL's decisions stay spread over 15 to
85% of the run; at a decision after R it is shown no held-out loss, since none is left. `--mode plain
--panel-frac 0.15` gets the same batches without OPAL or gates and is the matched baseline. `--panel-until 1`
keeps the panel out to the end, with gates up to 85% of a run that is as short as F: the behaviour before
this flag, bit for bit.

Measured without any agent (`--mode plain`, ntv2_50m on `nt_tasks_official_splice_sites_donors`, 2 epochs,
`--det strict`, 2,957 test rows), test loss and accuracy of the checkpoint validation selected:

| Seed | no panel | `--panel-frac 0.15 --panel-until 1` | `--panel-frac 0.15` (returns at update 1550 of 2508) |
|---|---|---|---|
| 42 | 0.1016, 0.9726 | 0.1255, 0.9692 | 0.1112, 0.9679 |
| 43 | 0.0969, 0.9770 | 0.1523, 0.9608 | 0.1020, 0.9729 |
| 44 | 0.0891, 0.9770 | 0.1539, 0.9574 | 0.1149, 0.9702 |
| mean | 0.0958, 0.9755 | 0.1439, 0.9625 | 0.1094, 0.9704 |

Returning the rows closed 72% of the loss gap and 60% of the accuracy gap to training without a panel; it stayed
behind on all three seeds (+0.014 loss, −0.005 accuracy on average; the sampling of test rows alone puts a
standard error of 0.003 on one accuracy). On seed 42 validation selected the epoch-1 checkpoint, saved at
update 1254, before the return: that score is of a model that never trained on the returned rows. With
`--mode keep` and an expert that keeps, the run ends with the weights of the plain run on the same batches.

## Smoke runs (costs: ¢)
Experiments can grow costly. As of November 2026, the most cost-efficient (and fastest) choice is `claude-haiku-5-5`.
This makes it an ideal candidate for debugging.

Run from `finetune/` in the `biopal-v2` env. Both use the cheap `haiku` model at low effort and a shared ledger, so
the two budgets are ceilings over every run that names the same ledger file.

```bash
# no Claude call (offline): gates, trial and rewind with a scripted expert
echo '{"informed": []}' > /tmp/keep.json
python finetune.py --model ntv2_50m --task nt_tasks_official_splice_sites_donors --limit 6000 --epochs 1 \
  --det strict --mode keep --panel-frac 0.15 --bio-policy expert --expert-script /tmp/keep.json \
  --outer-decisions 1 --outer-steps 32 --outer-audit 16 32 --no-test --out runs/smoke_offline

# online: 3 OPAL decisions, 2 gates with an informed and a masked expert (about $0.012 in total)
OPAL_MODEL=claude-haiku-5-5 OPAL_EFFORT=low python finetune.py --model esm2_t12_35m \
  --task proteingym_gfp_fluorescence --limit 8000 --epochs 1 --det strict --mode biopal --bio-policy expert \
  --outer both --expert-shadow masked --decisions 3 --outer-decisions 2 --outer-steps 32 \
  --expert-model claude-haiku-5-5 --expert-effort low --budget 0.20 --expert-budget 0.20 \
  --ledger runs/smoke_ledger.jsonl --out runs/smoke_live
(cd .. && python -m biopal finetune/runs/smoke_live)   # the trace of the first gate with a trial
```

Note, however, that `haiku` appears inactive for most OPAL updates. It just does not appear capable for the bilevel optimization.

## Research pilots (costs: $)

Full data, one seed, default gate settings (5 OPAL decisions, 5 gates, 64-update trials). Set the three
placeholders first; the OPAL model and effort Vare read from the environment (`finetune.py` defaults to
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

## Local expert (open weights on this machine)

`--expert-backend local` puts a locally served open-weights model in the expert's seat. It reads the same brief
as Claude would, answers under the same schema and menu checks, and the matched trial still decides what is
adopted. It can also read a note from the task owner (`--expert-note`), which no hosted call ever sees. The
question the tooling answers first: on the same checkpoints, does the same local model choose better with the
note than without it? A comparison with Claude on the public brief is possible but secondary: it mixes model
size with everything else.

```bash
# 0. serve the model, in a terminal of its own (it stays in the foreground): any env with vLLM, weights
#    already on disk, pinned by their commit. Set REV in the other terminal too.
REV=b968826d9c46dd6066d109eabc6255188de91218
QWEN=/path/to/models--Qwen--Qwen3-8B/snapshots/$REV     # set this: the snapshot folder in a Hugging Face cache
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 vllm serve "$QWEN" \
  --served-model-name Qwen/Qwen3-8B --host 127.0.0.1 --port 8000 --max-model-len 8192 --dtype bfloat16 --seed 0
#    on a GPU that other jobs use, add: --gpu-memory-utilization 0.25 --kv-cache-memory-bytes 2G --max-num-seqs 4
Q='{"chat_template_kwargs": {"enable_thinking": false}}'     # Qwen3 on vLLM: answer without a thinking block
NOTE=../biopal/notes/nt_tasks_official_splice_sites_donors.txt

# 1. audit (no model call): one fixed trajectory; at every gate each legal action is trained from the same
#    snapshot and scored on the private rows; --expert-shadow masked saves both briefs
echo '{}' > keep.json
python finetune.py --model ntv2_50m --task nt_tasks_official_splice_sites_donors --epochs 2 \
  --mode keep --panel-frac 0.15 --det strict --bio-policy expert --expert-script keep.json \
  --expert-shadow masked --outer both --outer-steps 64 --outer-audit 16 32 64 128 --no-test \
  --out runs/audit_donors

# 2. replay: the local model answers at the audit's gates, masked, informed, and informed with the note
(cd .. && python -m biopal.replay finetune/runs/audit_donors --out finetune/runs/replay_donors \
  --local-model Qwen/Qwen3-8B --local-revision $REV --local-request "$Q" --note finetune/$NOTE)

# 3. live: the local model is the outer expert of a training run (smoke size; OPAL without an agent)
python finetune.py --model ntv2_50m --task nt_tasks_official_splice_sites_donors --limit 6000 --epochs 1 \
  --mode keep --panel-frac 0.15 --det strict --bio-policy expert --outer both --outer-decisions 1 \
  --outer-steps 32 --expert-backend local --expert-model Qwen/Qwen3-8B --expert-revision $REV \
  --expert-request "$Q" --expert-note $NOTE --no-test --out runs/smoke_local
```

Step 3 with `--mode biopal --budget ...` in place of `--mode keep` is the full method: OPAL's agent inside, the
local expert outside (run once at this size with `claude-haiku-5-5` as OPAL's agent: $0.010 for OPAL, nothing
for the expert). `--expert-url` (default `http://127.0.0.1:8000/v1`) names another port.

**The note** is a text file: header lines, a blank line, the text.

```
task: nt_tasks_official_splice_sites_donors
label: simulated institutional knowledge
visibility: private

GC-rich windows are where ...
```

`task` binds the note: the trainer refuses a note written for another task, and a replay over several tasks
uses each note only for the runs of its task. Only the informed local expert reads it, between the background
line and the checkpoint; measurements, menu and their order are the same bytes with and without it. With
`visibility: private` (the default) the text exists in the note file and in the request to the local server,
nowhere else: the saved brief is the one without the note, `outer.jsonl` and `result.json` carry the note's
hash, and the expert's free-text answers and error texts are replaced there by a marker. The raw reply in
`expert/gateN_informed_reply.json` can repeat the note and is the one file to keep on the machine.
`visibility: public` lifts that. The example note in `biopal/notes/` is public and author-written:
simulated institutional knowledge, not a lab's.

**What the client refuses.** The address must be plain http on a loopback IP (`127.0.0.1`, not `localhost`).
Proxy settings of the environment are not used, a redirect is an error, and on a shared machine the port must be
served by you or by root. There is one request per proposal. A timeout, an HTTP error, a reply that is not
plain JSON under the schema, or a target the role was not shown counts as a failure (`failure` in the log:
`timeout`, `transport`, `malformed`, `off_menu`), keeps the incumbent, and is not asked again or elsewhere; a
chosen `keep` is logged as `ok` with outcome `kept`. A local call needs no budget and writes no ledger line. The
server is asked for its model list and version while the flags are checked, so a wrong port or a server that
is not up yet stops the run with one line, before the model loads and before `--out` exists.
`--expert-revision` is recorded, and checked when the server's model path names a commit (a Hugging Face snapshot
folder does). `result.json` then holds model, revision, vLLM version, decoding settings, calls, failures and
latency per call under `biopal.expert`.

**Local expert is not yet an offline run.** With a local expert no brief and no note goes to a hosted model.
The run as a whole is offline only if also: `--mode keep` (with `--mode biopal` OPAL's agent is a hosted model
that gets training losses and can read files of the run), weights and data staged beforehand
(`HF_HUB_OFFLINE=1`), the server's telemetry off (`VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`), and outbound
network blocked for both the trainer and the server. The code cannot check the last point.

**Replay** (`python -m biopal.replay -h`) reads `outer.jsonl` and the saved briefs of audit runs. Each proposer
gets the brief without the earlier-gates section, all answers are written to `proposals.jsonl`, and only then
are the recorded trials read. Always there: `always_keep`, `uniform` (the mean over every legal action, keep
included) and `worst_gap` (the offered cohort with the largest baseline gap among those whose interval lies above
0, else keep). With `--local-model`: the local model masked, informed, and informed with `--note`. With
`--hosted-model ... --hosted-budget USD --exportable`: Claude on the informed brief the local model read, never
on a note; the budget is one cap over all gates and runs, and `--exportable` is your statement that these briefs
may leave the machine. `result.json` holds per gate each choice and its loss difference against keep at every
audited horizon (keep is 0), the choice's standing among the gate's legal actions, coverage (a gate enters the
comparisons of a horizon only if every legal action was scored there; missing and failed branches are listed),
the drift of the incumbent repeat, what a readout switch adds over refitting the head in place, and the paired
difference note minus no note per run and task. Gates of one run share a trajectory, so runs are the unit and
no pooled test over gates is given. A replay measures the quality of proposals over the audited horizons. What a
note does for a trained model takes matched live runs with adoption on and a test set nobody looked at.

**First run** (2026-11-19: ntv2_50m on `nt_tasks_official_splice_sites_donors`, one audit run; Qwen/Qwen3-8B at
`b968826d` on vLLM 0.11.0, temperature 0, thinking off). The audit placed 5 gates (after updates 350, 700, 1100,
1450, 1850; none dropped). All 11 legal actions were scored at every gate and horizon, and the incumbent repeat
reproduced the incumbent exactly. These runs held the panel out to the end; add `--panel-until 1` to steps 1 and
3 to repeat them (the default now places the gates in the first 60% of a longer run). Mean loss difference
against keep after 64 updates:

| Proposer | mean_d | Choices at the 5 gates |
|---|---|---|
| always keep, worst-gap rule | 0 | keep ×5 (no cohort's baseline gap lay above 0) |
| uniform over the menu | +0.020 | |
| Qwen, masked | +0.257 | block_6 ×5 |
| Qwen, informed | +0.257 | block_6 ×5 |
| Qwen, informed with the note | +0.153 | block_9 ×2, block_6 ×3 |
| `claude-haiku-5-5` (effort low), same informed brief | −0.003 | keep ×3, block_9 ×2 |

Note minus no note: −0.104 after 64 updates (−0.116, −0.076, −0.156 after 16, 32, 128). The choice differed at 2
of the 5 gates (−0.278, −0.240); the other three are ties. This is one run, so one trajectory, and no test is
given. What happened behind the number: without the note Qwen read the layer rows with the wrong sign and chose
the worst action on the menu at every gate. With the note it chose block_9 at two gates, which there did what
refitting the head in place does (−0.000 and −0.002 beyond the refit). It never chose what the note advises,
more weight on the GC-rich third, which scored +0.000 on average. So the note made this model's choices less
bad, not better than keep, and not by the route the note describes. Gate 3 was named as the example before any
score was read: +0.263 (SE 0.021) without the note, +0.023 (SE 0.006) with it.

The live run (step 3, both roles): two schema-valid, on-menu answers in 2.2 s and 1.3 s (0.7 s per call in the
replay; about 2,600 prompt tokens each), both for block_6; the 32-update trial measured +0.093 (SE 0.026) and the
run kept the incumbent. The server held 18.8 GB of GPU memory after loading and 19.1 GB after serving, with the
three flags of the comment in step 0. The 5 Claude calls of the replay cost $0.005. The note was hashed before
the audit started; its author had seen aggregate results of an earlier audit of this task (not per cohort).

## What a run writes

- `outer.jsonl`: one line per event. `schedule` (OPAL decisions and gates as placed, and `panel_release`, the
  last update before D and A are trained on), `shown_to_opal` (what OPAL
  was shown before each decision), and per gate `table` (the measurements and the actions on offer), `proposals`
  (written and synced before the private set is read), `trial` (per branch: training losses and, per horizon,
  the mean loss difference on A, its standard error and whether it passes; for a readout branch also
  `install_error`, the largest per-row difference on D between the installed head and the probe it came from),
  `decision`, and `bypassed` (after a readout switch: the tensors that got no gradient did not move).
- `outer.jsonl`, `schedule` also holds the names each role reads and which expert answers; a failed proposal
  has `failure` (its kind) next to `error`.
- `expert/gateN_ROLE_brief.txt` and `_reply.json`: what each expert read (without a private note) and what the
  CLI or the local server returned.
- `result.json`, key `biopal`: adopted changes, final configuration, Claude spend by role, OPAL agent calls inside
  trial branches (must be 0), and the readout block `best.pt` was saved with.
- `config.json`, `recipe.panel`: sizes and hashes of F, D and A. Equal hashes mean equal rows across modes.
  `recipe.panel_until` and `recipe.panel_release` say when the held-out rows returned (`null`: never).

## Tests

```bash
# CPU-only, no Claude call, no model server (a fake one on a loopback port); 133 tests, ~1.5 min
conda run -n biopal-v2 python -m pytest tests/biopal -q
```

## Known limits

- A loopback port is private to the machine, not to you. The client checks who serves the port (Linux), but on a
  shared host start the server yourself and give it a port of its own.
- Replay has the mean loss difference and standard error of each action against keep, not the per-row losses.
  The difference between two proposers' choices therefore comes without an interval.
- *outer loop awareness**: OPAL copies the training script into its agent's workspace. The agent can therefore read that an outer level
  exists; it is not shown the cohort in force, the expert's text or any trial score.
- The value passed to `optimizer.step(loss=...)` is the unweighted loss in every mode. A program of OPAL's that
  reads `ctx.loss` gets that value while its gradients come from the weighted loss; `loss_aware_updater` in each
  `decision` record flags such runs.
- The rewind clears OPAL's resume note through a private attribute of OPAL 0.3.10 (`_resume_note`); a test fails
  if a later version changes that.
- After a readout switch the optimizer keeps its statistics from before the switch.
- `A` is read at every gate, so it becomes tuning data over a run. Validation and test are never touched by a gate.
- The returned rows get their visits late, at a low learning rate (2 epochs, `--panel-until 0.6`: on average
  15% of the peak rate), and are over a third of every batch then. No gate can open after the return, and OPAL's
  later decisions come without a held-out loss.
- Checkpoints are saved at the end of an epoch. One saved before the return (2 epochs: the first, at 50% of
  the run) can win on validation; the test score is then of a model that saw F only.
- The adoption rule's standard error counts only the sampling of rows in A, not how much a short training
  window varies from one run to the next. In the full-data audits (`--outer-audit 16 32 64 128`, keep backend) 25
  to 60% of all cohort options passed at any one horizon, the sign of the difference often flipped between
  horizons, and the differences at two horizons were uncorrelated. A pass in a single trial is therefore weak
  evidence on these two tasks; see the audit before trusting an adoption.


## First Test Run: BiOPAL, OPAL, Fine-Tuning

```bash
cd /home/csiebenschuh/Projects/biopal/finetune && conda activate biopal-v2

MODEL=claude-opus-5-5
EFFORT=medium
OPAL_USD=3
EXPERT_USD=1
RUN=runs/live_opus_medium

CUDA_VISIBLE_DEVICES=0 OPAL_MODEL="$MODEL" OPAL_EFFORT="$EFFORT" \
  python finetune.py \
    --model esm2_t12_35m \
    --task proteingym_gfp_fluorescence \
    --limit 8000 \
    --epochs 1 \
    --det strict \
    --mode biopal \
    --bio-policy expert \
    --outer both \
    --expert-shadow masked \
    --decisions 3 \
    --outer-decisions 2 \
    --outer-steps 32 \
    --expert-model "$MODEL" \
    --expert-effort "$EFFORT" \
    --expert-timeout 300 \
    --budget "$OPAL_USD" \
    --expert-budget "$EXPERT_USD" \
    --ledger "${RUN}.ledger.jsonl" \
    --out "$RUN"
```