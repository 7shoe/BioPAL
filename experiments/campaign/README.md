# BiOPAL campaign: protocol

Written 2026-11-19. Nothing has been launched. This file is the single description of what the campaign runs,
in which order, and what happens when something fails while nobody is watching. The stage scripts and the run
manifest are written from it; if they disagree with this file, this file is right.

State on 2026-11-20: ready to launch (section 15). Stage 0 was run once as a trial: all 7 checks and all 16
tiny runs passed, for $0.07 of Claude spend. The launch repeats stage 0 on the machine as it is then. No run
of full size has been made yet.

## 1. What the campaign shows

| Wave | Question | Runs that answer it |
|---|---|---|
| 0 | Does every kind of run work here, and do the agents act at the chosen Claude settings? | readiness checks, pilot |
| 1 | Are plain fine-tuning, OPAL and BiOPAL viable on every pair? What do their trajectories look like? | seed 42, all arms, six pairs |
| 2 | How do plain, OPAL, BiOPAL, BiOPAL with a local expert, and HPO compare? | seeds 42, 43, 44, all arms, six pairs |
| 3a | Does the choice of expert matter (Opus, Sonnet, Qwen3-8B with and without a note)? | audits on three pairs, then replays on their saved gates |
| 3b | Does the Claude model behind both agents matter? | Sonnet at high effort, three pairs, three seeds |

Wave 1 is not a separate set of runs: its tables and trajectory plots come from the seed-42 runs of wave 2.
No experiment is added beyond this list.

## 2. Fixed settings

### Six pairs

| Short name | Task | Model | Conda env | Objective | Train rows | Updates | Rows return at | Plain fit (min) | Free GPU memory needed |
|---|---|---|---|---|---|---|---|---|---|
| `ctcf` | `fish_gue_ctcf` | `dnabert2_117m` | `biopal-v2` | classification | 20,000 | 2,500 | 1,500 | 11 | 12 GB |
| `h3k27ac` | `nt_tasks_official_h3k27ac` | `ntv2_50m` | `biopal-v2` | classification | 20,619 | 2,578 | 1,550 | 8 | 12 GB |
| `splice` | `rna_dst_splice_site_acceptor` | `splicebert` | `bio-rna` | classification | 17,723 | 2,216 | 1,350 | 8 | 12 GB |
| `mrl` | `rna_dst_mean_ribosome_load` | `rnafm` | `bio-rna` | regression | 20,000 | 2,500 | 1,500 | 7 | 12 GB |
| `melt` | `cds_melting_point` | `esm2_t12_35m` | `biopal-v2` | regression | 9,432 | 1,180 | 750 | 14 | 30 GB |
| `gfp` | `proteingym_gfp_fluorescence` | `esm2_t12_35m` | `biopal-v2` | regression | 41,372 | 5,172 | 3,150 | about 20 | 12 GB |

- **Minutes** are one 2-epoch plain fit, measured on 2026-11-19 with every card of lambda13 at 100% use by other
  jobs (about 2.7 times slower than an idle card). `gfp` with `esm2_t12_35m` was not measured; 20 is an estimate.
- **Updates and the return update** are computed from the row counts: 2 epochs at batch 16, and 60% of the run
  moved up to a multiple of 50. The first run of each pair confirms them.
- **Memory** is what a card must have free when a job starts. `melt` reserved 21 GB in a plain fit; a gate adds
  a copy of the model and optimizer on top. The 12 GB for `gfp` is an estimate.
- **Objective:** `ctcf`, `splice` and `melt` are registered for both objectives, so every command passes
  `--objective` as in the table.

### Training settings, the same in every run

`--epochs 2 --batch-size 16 --det strict`, learning rate 2e-5, weight decay 0.01, warm-up 0.1 (trainer defaults;
only HPO changes these three). Panel runs use `--panel-frac 0.15 --panel-until 0.6`: 15% of the train rows give
no gradient for the first 60% of the run, then join training, so every run of a pair has the same number of
updates. Agents: `--check-every 50 --decisions 5 --outer-decisions 5 --outer-steps 64` (defaults).

### Arms

| Run name | What it is | Flags on top of the common ones | Claude cost |
|---|---|---|---|
| `plain` | fine-tuning on all train rows | `--mode plain` | none |
| `matched` | plain fine-tuning with the same held-out rows and the same batches as the agent arms | `--mode plain --panel-frac 0.15` | none |
| `opal` | OPAL's agent may rewrite the update rule; no expert | `--mode biopal --bio-policy off` | agent |
| `biopal` | OPAL's agent plus a Claude expert at five gates | `--mode biopal --bio-policy expert --outer both` | agent and expert |
| `biopal_qwen` | as `biopal`, expert is Qwen3-8B served on this machine, reading the task's note | `--mode biopal --bio-policy expert --outer both --expert-backend local --expert-note FILE` | agent |
| `hpo` | 6 trials of Bayesian search over learning rate, weight decay and warm-up, then the winner trained and scored once | `hpo.py --mode BO --n_trials 6`, then its `final_command` | none |

- `matched`, `opal`, `biopal` and `biopal_qwen` of one pair and seed see the same batches. Their weights are
  identical until an agent changes something, so any difference between them is the agents' doing.
- `opal` is the matched variant: it holds the same rows out as BiOPAL, and its agent is shown the loss on the
  diagnostic rows. OPAL without a panel (`--mode opal`) is not run.
- `plain` against `matched` shows what holding rows out for 60% of the run costs on its own.
- HPO trials are plain fits on all rows, selected by validation loss, never scored on test. The first trial is
  the trainer's default setting (seen in the trial of stage 0), the first three trials are the start-up of the
  search, and the last three are guided by its model. The study runs at seed 42 only; seeds 43 and 44 retrain
  the winning setting.

### Claude settings

| Role | Model | Effort | How it is set |
|---|---|---|---|
| OPAL's agent | `claude-opus-5-5` | medium | `OPAL_MODEL`, `OPAL_EFFORT` in the environment of every agent run |
| Expert | `claude-opus-5-5` | high | `--expert-model`, `--expert-effort` |

These stay the same for all three seeds. The trainer falls back to Opus at **high** effort when `OPAL_EFFORT`
is missing, so every script sets both variables and stage 0 checks what the run recorded. Haiku is not used
for results (it did nothing useful in earlier runs). Fable is not used (never tried with OPAL, no cost figure).

**OPAL's agent has no shell.** It reads and edits its two programs with Read, Edit and Write and cannot run
code. The trainer sets this itself (`sandbox=False` in `finetune/finetune.py`), so every run is in the same
state. The reason: OPAL only gives a shell inside an OS sandbox, and lambda13 does not allow one (its
administrators switched network namespaces off in June 2025, `user.max_net_namespaces=0`). All earlier agent
runs on this machine were in this state too. OPAL still tries every rewritten rule for 20 updates and goes back
to the previous one on an error.

### Local expert

Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218`, served by vLLM 0.11.0 from conda env `agent` on
`127.0.0.1:8000` with the flags in `biopal/README.md`, thinking switched off in the request. One server runs
from stage 0 to the end of stage 4. The weights stay where they were served from before, in another user's
cache; if that folder disappears, the `biopal_qwen` arm and the local replays stop and everything else goes on.

Each task gets one note in `notes/<task>.txt`. The notes are written from public knowledge and from statistics
of the public training rows, and are labelled simulated institutional knowledge. Each has two parts: what
drives the labels, and which group of rows the authors would give more weight if the measurements show a
weakness there. They say nothing about which block the task head should read. They are fixed, and their
SHA-256 values recorded here, before any gate of the campaign has run. A note is never sent to a hosted model
(the trainer refuses) and never appears in a command line.

The drafts are marked `visibility: public`, like the example note in `biopal/notes/`: the expert's reasons then
stay readable in `outer.jsonl`. With `private` the logs carry only the note's hash and a marker in place of
the expert's free text.

One exception to "before any gate": `proteingym_gfp_fluorescence` with `esm2_t12_35m` already had gates in two
runs on 2026-11-18 (8,000 rows, one epoch). The GFP note was written without reading the gate records of
those runs; only their totals were known (2 of 2 proposals adopted).

## 3. Stages

One script per stage, launched in this order. Hours count from the start of stage 0.

| Stage | Hours | Content | Runs | GPU-hours | Claude estimate | Starts without a person if |
|---|---|---|---|---|---|---|
| 0. Readiness | 0 to 0.5 | checks, one tiny run per kind of run | about 8 tiny | 0.3 | under $0.50 | always |
| 1. Seed 42, free | 0.5 on | `plain`, `matched`, HPO trials | 48 | 9.1 | none | always |
| 1b. Pilot | 0.5 to 2 | three Claude settings, `opal` and `biopal` | 10 | 1.8 | $13 | Claude check passed |
| 2. Seed 42, paid | 2 to 5.5 | `opal`, `biopal`, `biopal_qwen`, HPO winners; first table | 24 | 6.4 | $19 | Claude check passed |
| 3. Seeds 43, 44 | 5.5 to 10 | all arms; HPO winner retrained | 72 | 17.3 | $38 | stage 2 queue is empty or its deadline passed |
| 4. Expert ablation | 10 to 13.5 | three audits, then replays | 3 and replays | 6.3 | under $2 | always (replays need the server) |
| 5. Agent ablation | 10 to 14 | Sonnet at high effort, `opal` and `biopal` | 18 | 4.2 | $10 | its cap is still free |
| 6. Report | by 15.5 | tables, plots, spend, list of missing runs | none | none | none | always |

Total: about 175 runs, 45 GPU-hours, about $82. Sixteen hours on four cards are 64 card-hours, so about 19
are spare. Stage 3 is the tight one: 17.3 GPU-hours in 18 card-hours.

### Stage 0: readiness

A failed check switches off only the runs that depend on it.

1. Repository: record the commit and `git status`. The trainer is not edited while the campaign runs.
2. Environments `biopal-v2`, `bio-rna`, `agent` exist; OPAL 0.3.10 imports in the first two.
3. The six tasks load and the model weights are in the local caches.
4. Disk: at least 150 GB free.
5. GPUs: at least four cards have 12 GB free and at least one has 30 GB free.
6. Claude: `opal doctor --live` passes, and the tiny agent runs of check 9 report "file tools only".
7. The Qwen server answers and its port belongs to this user.
8. The six notes exist and match their recorded hashes.
9. One tiny run of every kind on `mrl` (few rows, Haiku at low effort): `plain`, `matched`, `opal`, `biopal`,
   `biopal_qwen`, an HPO study of one trial with its winner, an audit, a replay. Each must leave its result
   file, and each paid one its ledger rows and the model and effort it was told to use.
10. One tiny paid run with a cap of one cent, to record what the trainer does when a cap is reached. Seen in
    the trial: the agent was asked once instead of twice, and the run went on to its end without it.
11. Every pair is loaded and trained for 30 updates. A pair that fails here, or a kind of run whose tiny run
    fails, is switched off for the later stages.

### Stage 1b: pilot

Seed 42, arms `opal` and `biopal`. Pilot runs never enter a results table.

| Setting | Agent | Expert | Pairs |
|---|---|---|---|
| default | Opus, medium | Opus, high | `h3k27ac`, `mrl` |
| Sonnet | Sonnet, high | Sonnet, high | `h3k27ac`, `mrl` |
| higher effort | Opus, high | Opus, high | `mrl` |

It answers three things: what a run really costs, whether the agents act on these pairs, and whether Sonnet's
inaction on the donor task was the model or the task.

The setting for stages 2 and 3 is chosen by behaviour and cost, never by a test metric. The default stays
unless a person changes it before stage 2 starts. The one change foreseen: OPAL's agent moves to high effort
if the higher-effort runs cost at most $1.50 each and the agent visibly does more than at medium.

### Stages 2 and 3: the comparison

- Stage 2 ends with the seed-42 table and the trajectory plots. That is wave 1, and it is a reportable result
  on its own.
- Paid runs of seed 43 all start before any paid run of seed 44, so a budget stop leaves whole seeds.
- An arm that failed twice on a pair at seed 42 is not run on that pair at seeds 43 and 44.
- A paid run that has a result is never run again. The agents are not deterministic, and a second run would
  invite choosing between two outcomes.

### Stage 4: expert ablation

Audits on `h3k27ac`, `mrl` and `melt` (one per modality), seed 42: no agent, a scripted expert that keeps, and
at each of the five gates every option on offer is tried and the incumbent repeated, scored after 16, 32, 64
and 128 updates. An audit costs about 13 plain fits and no Claude call. Flags: `--mode keep --panel-frac 0.15
--bio-policy expert --expert-script keep.json --expert-shadow masked --outer both --outer-audit 16 32 64 128
--no-test`.

Replays then ask each proposer at the 15 saved gates and score the answer against the recorded trials:
Qwen3-8B with and without the note, Opus at high effort, Sonnet at high effort. Hosted models get the same
brief as the local model and never a note. Sending these briefs to a hosted model is allowed because all six
tasks are public data. A second, smaller local model was dropped from the plan on 2026-11-19.

### Stage 5: agent ablation

Sonnet at high effort for both agents, arms `opal` and `biopal`, pairs `h3k27ac`, `mrl`, `melt`, seeds 42, 43,
44. Compared with the Opus runs of the same pairs and seeds from stages 2 and 3.

### Stage 6: report

Runs at its hour whatever is finished. It never trains. It writes the tables, the plots, the spend per stage,
and a list of every run that is missing with the reason.

## 4. Claude budget

The limit is **$150 for the whole campaign**, with no time window.

| Stage | Estimate | Cap |
|---|---|---|
| 0 and 1b, readiness and pilot | $13 | $20 |
| 2, seed 42 | $19 | $30 |
| 3, seeds 43 and 44 | $38 | $60 |
| 4, replays | under $2 | $5 |
| 5, agent ablation | $10 | $20 |
| Held back | | $15 |
| Total | about $82 | $150 |

Where the estimates come from: OPAL's agent at Opus-medium cost about $0.90 per run in earlier runs here
(extrapolated from three decisions to five). The expert at Opus-medium cost $0.036 per call. Expert at high
effort, Sonnet at high effort and Opus at high effort for the agent were never run; their figures are guesses.

How the caps are enforced:

- **Every paid run has its own ledger file and its own cap.** $2.00 for OPAL's agent and $1.00 for the expert;
  $4.00 for the agent in the pilot's higher-effort runs. The trainer reads the agent's remaining budget once,
  when the run starts, and books the agent's spend when the run ends. A ledger shared by parallel runs would
  let them overshoot together, so they do not share one.
- **Before every paid launch the script adds up all ledger files.** A run that is still going, or that failed
  without booking its spend, counts at its full cap. The run starts only if the stage stays under its cap and
  the campaign under $135. Otherwise it waits.
- **The $15 held back** covers what the per-run caps cannot stop exactly.
- **When a cap is reached, paid runs stop and free runs go on.** Nobody raises a cap but the user.

## 5. GPUs

- Before each launch the script reads the free memory of all eight cards and gives the job to the card with
  the most room that meets the job's need (section 2). No card is reserved or excluded in advance.
- At least four jobs run at a time when four cards qualify, and up to eight. Two of our jobs share a card only
  when fewer than four cards have room.
- Jobs come from one queue, longest first: the HPO studies of `gfp` (about 140 minutes) and `melt` (about 100)
  start first, because a study runs its trials one after another on one card.
- The Qwen server needs about 24 GB on its card and may share that card with a small job.
- A job that runs out of memory is retried once on the card with the most free memory.

## 6. Disk and checkpoints

The trainer saves the full model (`best.pt`, 0.1 to 0.5 GB) and reloads it once for the test evaluation. Nobody
uses it afterwards. Everything is on one shared filesystem that was 98% full on 2026-11-19 with 389 GB free.

| Tier | Runs | Their `best.pt` |
|---|---|---|
| 1 | HPO trials, audits, pilot, readiness, failed attempts | deleted as soon as the run or study has finished |
| 2 | every run whose test metrics enter a table: 90 main-arm runs, 18 HPO winners, 18 agent-ablation runs | kept until the user signs off on the results, then deleted in one step |

- Tier 2 is about 30 GB (estimated from the parameter counts). While it exists, per-row predictions or a new
  metric can still be computed without retraining.
- **Safety floor:** if less than 100 GB is free before a launch, all tier-2 checkpoints of finished runs are
  deleted and every later one is deleted when its run ends.
- **Hard stop:** if less than 40 GB is free, nothing new is launched.
- Only `best.pt` inside the campaign's own output folder is ever deleted. Logs and result files are never
  deleted, also not for failed attempts.

## 7. What every run leaves behind

A few MB per run, in `finetune/runs/campaign/<group>/<pair>/<run name>_s<seed>/` (not in git). The groups are
`ready` (stage 0), `main` (stages 1 to 3), `pilot_<setting>` (stage 1b), `ablation_expert` (stage 4) and
`ablation_agent` (stage 5). This path without its first part is the run's id in `runs.tsv` and in the status
table, for example `main/h3k27ac/biopal_s42`. What a run printed is in `finetune/runs/campaign/_logs/`, one
file per attempt.

| File | Content |
|---|---|
| `config.json` | the full configuration, including the panel split and the return update |
| `result.json` | test metrics, best validation metrics, OPAL's cost, gates and adoptions |
| `train.jsonl` | loss of every update |
| `validation.jsonl` | validation metrics at the end of each epoch (two points) |
| `opal/` | each decision of OPAL's agent and each version of the update rule |
| `outer.jsonl` | each gate: measurements, proposal, trial, decision |
| `expert/` | briefs and raw replies of the expert; local only |
| the run's log | everything the run printed |

Trajectory plots use the training loss per update, with marks at OPAL's decisions, at the gates and at the
return update. A held-out loss every 50 updates should also be there for the agent arms until the return;
stage 0 confirms where it is written.

## 8. Running without supervision

- **A run is done when its result file exists.** Stages can be launched again at any time and skip what is done.
- **A failed attempt is moved aside** to `<run>.failed1`, never deleted. The trainer refuses an existing folder.
- **Time limit:** three times the expected duration of the run. Then it is stopped and counts as failed.
- **One retry,** on another card. After the second failure the run is skipped and written to the failure log.
- **Deadlines:** stage 2 at hour 6.5, stage 3 at hour 11.5, stages 4 and 5 at hour 15. Runs not started by
  then are deferred and the next stage begins. Deferred runs use spare card-hours in the cut order below.
- **Paid runs fall back to free ones.** If Claude, a cap or the server fails, the affected arms stop and
  everything else continues.
- **Everything runs in a detached session** and survives a closed terminal.

Cut order when time runs out: stage 5 first, then the `melt` audit, then seed 44 of `biopal_qwen`. The seed-42
table and the `matched` runs are never cut.

Status files in `experiments/campaign/status/` (written by the scripts, not in git):

| File | Content |
|---|---|
| `summary.txt` | counts per stage, free disk, Claude spend per budget line; the first thing to read |
| `status.tsv` | every run with its state (done, running, pending, failed), the reason it waits or was skipped, card, minutes, USD |
| `failures.log` | every failed attempt with its reason and the path of its log |
| `events.log` | every start and end of a run |
| `readiness.txt` | the checks and the tiny runs of stage 0, PASS or FAIL |
| `report.md`, `results.tsv`, `plots/` | the report of stage 6 |
| `campaign.log` | everything `campaign.sh` printed |
| `ledgers/` | one spend file per paid run |
| `interventions.log` | every manual or agent intervention, written by whoever intervenes |

### What a monitoring agent may do

Allowed, each time with a line in the intervention log: launch a stage again, move a job to another card,
restart the Qwen server, correct a path or a flag in a campaign script.

Not allowed without the user: change arms, pairs, seeds, epochs or Claude settings; raise a cap; delete
anything but tier-1 checkpoints; edit `finetune/` or `biopal/`; commit or push. A bug in the trainer means:
stop the paid arms, let the free ones finish, write down what was seen.

## 9. How the results are read

Fixed now, before any result exists.

- **Primary metric:** MCC for the three classification pairs, Spearman correlation for the three regression
  pairs, on the test split, from the checkpoint with the lowest validation loss. Accuracy, AUROC, RMSE, R² and
  the test loss are reported next to them. The trainer has no average precision and no macro-F1.
- **Comparisons:** `plain` against `matched` (cost of the held-out rows); `opal` against `matched` (OPAL's
  agent); `biopal` against `opal` (the expert); `biopal_qwen` against `biopal` (local against hosted expert);
  `hpo` against `plain` and against `biopal`.
- **Per pair:** mean and standard deviation over three seeds, and the three paired differences. One arm is
  called better than another on a pair only if the difference has the same sign on all three seeds. Anything
  else is reported as not distinguishable.
- **Activity is reported with the metrics:** how many decisions changed the rule, how many proposals were made
  and adopted. A run in which no agent acted equals its `matched` run and is reported as such.

## 10. Known limits

- Three seeds per cell. On the donor task the seed-to-seed standard deviation of plain test accuracy was 0.0025.
- With two epochs, the checkpoint saved after epoch 1 lies before the held-out rows return. If validation
  picks it, the scored model never trained on those rows. The report states the selected epoch for every run.
- The adoption rule (`mean_d < -SE`) is used as it is. It failed the noise audit on the donor task, and
  whether to change it is the user's open decision. The audits of stage 4 measure it on three more pairs.
- `biopal_qwen` differs from `biopal` in the model and in the note. Only the replays separate the two.
- The HPO setting is selected on seed 42 and reused for seeds 43 and 44.
- The durations were measured on one evening's load. The cards are shared and the load will change.
- OPAL's agent cannot run code (section 2). It cannot try a rewritten rule before handing it in; a faulty one
  costs one of the five decisions.
- The notes are author-written from public knowledge, not a lab's. Two of their statements could not be checked
  in the data, which has no species column: that the proteins melting above 70 degrees C are mostly from
  heat-loving organisms, and that AT-rich acceptor windows come from plants, nematodes and insects.

## 11. Not in this campaign

Caduceus and GENA-LM models (not runnable here), the remaining tasks of
`experiments/panel/BioPAL_Experiment_Panel.md`, Haiku and Fable as agents, a second local model, calibration of
the adoption rule, TxGemma, runs of more than two epochs.

## 12. Open before launch

| Item | State | Who |
|---|---|---|
| Start time | Not set. | user |

Settled on 2026-11-19 and no longer open: the sandbox (off, section 2), the six notes (accepted by the user as
drafted and fixed below), and the task descriptions (five one-sentence lines added to `ABOUT` in
`biopal/panel.py`, so every expert's brief now has a line of public background for all six tasks).

The notes were fixed on 2026-11-19, before any gate of the campaign. The values are the SHA-256 of the note's
text after its header, which is what the trainer writes into `result.json` and `outer.jsonl`. A run whose
recorded value differs did not read the fixed note.

| Task | File | SHA-256 |
|---|---|---|
| `fish_gue_ctcf` | `notes/fish_gue_ctcf.txt` | `368e216f3f951cd95f9afc9c396dee3cfebdf4e77372ef4f2d14254f538665d0` |
| `nt_tasks_official_h3k27ac` | `notes/nt_tasks_official_h3k27ac.txt` | `8367ec97eaf4c81e3f76deb89161b9e646e85f2dd76aa1f04c7aa64458efea6f` |
| `rna_dst_splice_site_acceptor` | `notes/rna_dst_splice_site_acceptor.txt` | `13e68aab9fefc795445568f0dec4820621f06d78bde350a155ece28acab6dc76` |
| `rna_dst_mean_ribosome_load` | `notes/rna_dst_mean_ribosome_load.txt` | `bed3543229664e6739c0d537733391f111cf3c6ab62b85a642d3b4e06597aba0` |
| `cds_melting_point` | `notes/cds_melting_point.txt` | `a253b5b948124ec478ec04e8c99e712ad0786b1030d056902f57dbd6236d9548` |
| `proteingym_gfp_fluorescence` | `notes/proteingym_gfp_fluorescence.txt` | `472caee30a3b914b2ffb264b4766725cd86587229f04b3ca6900c24fdcc98055` |

## 13. Decisions and proposals

Decided by the user on 2026-11-19:

- The waves, the six pairs, two epochs, three seeds, and no further experiments.
- Opus for both agents, agent at medium effort, expert at high. Haiku is out.
- $150 of Claude spend for the whole campaign.
- Any of the eight cards that has room; four at least.
- `opal` and `biopal` are the matched variants.
- Checkpoints in two tiers, with the 100 GB safety floor.
- Qwen3-8B is served as before. No second local model.
- The campaign runs without OPAL's sandbox: the agent has no shell, set in the trainer.
- The six notes as drafted, marked public, and a line of public background for every task in the brief.

Proposed in this file and not yet confirmed by the user:


- The primary metrics and the rule for calling an arm better (section 9).
- The pilot pairs, and `h3k27ac`, `mrl`, `melt` as the three pairs of both ablations.
- Tier 2 also holds the 18 HPO winners and the 18 agent-ablation runs.
- The hard stop at 40 GB free.
- Per-run caps of $2.00 and $1.00, and Haiku at low effort for the tiny readiness runs.

## 14. Files of this folder

| Path | Content | State |
|---|---|---|
| `README.md` | this protocol | written |
| `runs.tsv` | one row per run: id, stage, pair, run name, seed, Claude setting, kind, env, memory need, expected minutes, paid or free, what it needs first | written by `campaign.py manifest` |
| `campaign.py` | the runner: picks cards, enforces budget, disk and time limits, keeps the status files | written |
| `stage_*.sh`, `campaign.sh` | one script per stage, and one that chains them | written |
| `qwen_server.sh` | starts, stops and checks the local expert (tmux session `qwen`) | written |
| `report.py` | stage 6 | written |
| `test_campaign.py` | checks the runner's handling of failures, time limits, skipped runs and the budget with fake commands; 10 seconds, no GPU, no Claude call. Run it after any change to `campaign.py` | written |
| `notes/` | the six task notes and `hashes.tsv` | fixed on 2026-11-19; do not edit (the hashes in section 12 would no longer match) |
| `keep.json` | the scripted expert of the audits: it always keeps | written |
| `status/` | written by the scripts while the campaign runs | not in git |

## 15. How to launch and watch

```bash
cd /home/csiebenschuh/Projects/biopal
tmux new-session -d -s campaign 'experiments/campaign/campaign.sh'   # the whole campaign, detached
cat experiments/campaign/status/summary.txt                          # where it stands
tail -f experiments/campaign/status/campaign.log                     # what it is doing
```

- `campaign.sh` sets the clock (`status/t0`) on its first start, repeats stage 0 on the machine as it is then,
  and runs the stages in the order of section 3. Stage 1 runs beside stages 1b and 2, and stage 4 beside
  stage 5. The report is written after seed 42 and again at the end. Before the last report, runs that were
  deferred or failed once get another chance, most important first.
- Started again after an interruption, it continues: runs with a result are skipped, the clock stays. Jobs of
  a runner that was stopped go on by themselves and are booked by the next runner.
- One stage by hand: `experiments/campaign/stage_3_seeds43_44.sh`. One run by hand:
  `python3 experiments/campaign/campaign.py run --stage 2 --only main/h3k27ac/biopal_s42`. Add `--dry-run` to
  either to see the commands without starting anything.
- Stage 0 also loads every pair and trains it for 30 updates. A pair that does not load, or a kind of run
  whose tiny run fails, is switched off for the later stages (`status/readiness.txt` says which).
- After the user signed off on the results: `python3 experiments/campaign/campaign.py cleanup --yes` deletes the
  kept checkpoints, and `experiments/campaign/qwen_server.sh stop` ends the local expert.
