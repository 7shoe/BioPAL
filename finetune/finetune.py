#!/usr/bin/env python3
"""Unified fine-tuning: any registered backbone on any curated task.

Models come from models/backbones/models.yaml (audit-generated, OK-only) plus
the driving flags in finetune/backbone_flags.py; tasks come from
data/processed/tasks.yaml; machine paths from paths/paths.yaml. Loading reuses
the smoke_backbone helpers, so every model quirk learned there (custom
auto_map, encoder unwrapping, slow/spaced tokenizers, NTv3 hidden-state
selection) applies here unchanged.

Architecture: the full backbone is fine-tuned through a linear head on
mask-aware mean-pooled hidden states (the mask is dropped when the backbone
changes sequence resolution, e.g. NTv3's U-Net). Classification uses
cross-entropy over the task's classes; regression standardizes y on train
statistics, trains on MSE in z-space, and reports metrics in original units.

Defaults are a generic fine-tuning recipe: FP32,
batch 16, five epochs, AdamW 2e-5, weight decay .01 on matrix weights only,
10% linear warmup then cosine decay, gradient clipping at 1.0 in every mode.
Validation selects the best epoch; test is evaluated once after restoring that
checkpoint (and after closing OPAL). Agent feedback is training-only.

Modes: plain AdamW, OPAL keep (wrapped, no interventions - parity target),
opal (live Claude; requires --budget), or biopal (live OPAL as the inner level
of BiOPAL; see biopal/__init__.py. --mode keep with --panel-frac is its offline
twin). --max-steps is a diagnostic run (no
validation/checkpoint/test; plain/keep only). --limit subsamples each split
for pipeline smoke tests. --compare checks two run dirs for exact parity.
best.pt holds backbone+head weights, not an optimizer/resume checkpoint.

Run under the conda env that serves the model (see finetune/backbone_flags.py
and envs/README.md), e.g.:
  conda run -n biopal-v2 python finetune.py \
    --model ntv2_50m --task nt_tasks_official_splice_sites_donors \
    --objective classification --mode plain --out runs/ntv2_donors_plain_s42
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
import types

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models" / "backbones"))  # smoke_backbone helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backbone_flags import FLAGS


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def machine_paths():
    """(curated_base, {cache key: absolute dir}) for the current machine."""
    cfg = load_yaml(REPO / "paths" / "paths.yaml")
    for machine in cfg["machines"].values():
        base = (machine or {}).get("models_base")
        if base and Path(base).is_dir():
            return (Path(machine["curated_base"]),
                    {k: Path(base) / v for k, v in cfg["caches"].items()})
    raise SystemExit("No machine in paths/paths.yaml has an existing models_base")


def registries():
    models = load_yaml(REPO / "models" / "backbones" / "models.yaml")["models"]
    tasks = load_yaml(REPO / "data" / "processed" / "tasks.yaml")["tasks"]
    return models, tasks


def compare_runs(paths):
    left, right = [Path(p) for p in paths]
    a, b = [json.loads((p / "result.json").read_text()) for p in (left, right)]
    for key in ("initial_sha256", "final_sha256", "steps", "data_sha256", "recipe"):
        if a[key] != b[key]:
            raise RuntimeError(f"Parity failed: {key} differs")
    losses = []
    for p in (left, right):
        losses.append([json.loads(line)["loss"] for line in
                       (p / "train.jsonl").read_text().splitlines()])
    if len(losses[0]) != a["steps"] or len(losses[1]) != b["steps"]:
        raise RuntimeError("Training log length does not match step count")
    if losses[0] != losses[1]:
        raise RuntimeError("Parity failed: training loss sequences differ")
    print(f"PASS: identical initial/final parameters and all {a['steps']} losses.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true",
                   help="list runnable models and registered tasks, then exit")
    p.add_argument("--compare", nargs=2, metavar=("PLAIN_DIR", "KEEP_DIR"))
    p.add_argument("--model", help="name from models/backbones/models.yaml")
    p.add_argument("--task", help="name from data/processed/tasks.yaml")
    p.add_argument("--objective", choices=["classification", "regression"],
                   help="required when the task registers both")
    p.add_argument("--mode", choices=["plain", "keep", "opal", "biopal"], default="plain")
    p.add_argument("--out", type=Path)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--momentum", type=float, default=.9,
                   help="SGD momentum (ignored for adamw)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup-frac", type=float, default=.1)
    p.add_argument("--clip-norm", type=float, default=1.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-chars", type=int, default=0,
                   help="truncate raw sequences to this many characters (0 = none)")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="linear probe: train the head only")
    p.add_argument("--max-steps", type=int, default=0,
                   help="diagnostic run: no validation, checkpoint, or test")
    p.add_argument("--limit", type=int, default=0,
                   help="diagnostic: cap each split at N rows (stratified for classification)")
    p.add_argument("--budget", type=float,
                   help="required positive USD cap for live Claude (--mode opal)")
    p.add_argument("--decisions", type=int, default=5,
                   help="OPAL agent decisions per run, also the hard cap incl. "
                        "agent-requested wake-ups (keep/opal; OPAL's own default is 50)")
    p.add_argument("--det", choices=["strict", "warn", "off"], default="warn",
                   help="deterministic algorithms: strict errors on non-deterministic "
                        "ops (required for --compare parity), warn only warns")
    p.add_argument("--cpu", action="store_true",
                   help="run on CPU (smoke tests; real runs need CUDA)")
    p.add_argument("--no-test", action="store_true",
                   help="skip the test evaluation (HPO candidates select on validation only)")
    b = p.add_argument_group("BiOPAL (biopal/)")
    b.add_argument("--panel-frac", type=float,
                   help="hold this fraction of train out of the gradient, half as diagnostic "
                        "set D and half as private set A (default 0; 0.15 with --mode biopal)")
    b.add_argument("--bio-policy", choices=["off", "fixed", "expert"], default="off",
                   help="off: OPAL sees the loss on D. fixed: also the whole-panel baseline and "
                        "generalization gaps. expert: OPAL sees what off shows, and an expert "
                        "proposes changes to the training configuration at gates")
    b.add_argument("--outer", choices=["off", "cohort", "readout", "both"], default="both",
                   help="what the expert may change: the cohort whose rows weigh more in the "
                        "loss, the block that feeds the task head, or either (one per gate)")
    b.add_argument("--outer-decisions", type=int, default=5, help="gates per run")
    b.add_argument("--outer-steps", type=int, default=64,
                   help="updates of the matched trial at a gate")
    b.add_argument("--outer-alpha", type=float, default=.2,
                   help="share of the loss weight moved to the chosen cohort")
    b.add_argument("--outer-audit", type=int, nargs="*", metavar="UPDATES",
                   help="at every gate also try each option on offer and repeat the incumbent, "
                        "scored after these many updates (default 16 32 64 128); no extra "
                        "Claude calls")
    b.add_argument("--expert-shadow", choices=["none", "masked"], default="none",
                   help="masked: a second expert without biological names proposes from the "
                        "same checkpoint; its proposal is scored, never adopted")
    b.add_argument("--expert-script", type=Path,
                   help="JSON file of scripted proposals instead of Claude calls")
    b.add_argument("--check-every", type=int, default=50,
                   help="OPAL's check interval in updates; BiOPAL schedules fall on checks")
    b.add_argument("--ledger", type=Path,
                   help="JSON-lines file of Claude spend shared by several runs; with it, "
                        "--budget is a ceiling on their sum")
    args = p.parse_args()

    models, tasks = registries()
    if args.list:
        print("Runnable models (registry ∩ backbone_flags):")
        for name in sorted(set(models) & set(FLAGS)):
            f = FLAGS[name]
            print(f"  {name:20s} {models[name]['modality']:8s} env={f.get('env')}")
        print("Tasks:")
        for name, t in sorted(tasks.items()):
            print(f"  {name:40s} {t['modality']:8s} {','.join(t['tasks'])}")
        return
    if args.compare:
        compare_runs(args.compare)
        return

    # ------------------------------------------------------------- validation
    if not args.model or not args.task or args.out is None:
        p.error("--model, --task, and --out are required (see --list)")
    if args.model not in models:
        p.error(f"Unknown model {args.model!r}; audit registry has: {sorted(models)}")
    if args.model not in FLAGS:
        p.error(f"{args.model} is registered but has no entry in "
                f"finetune/backbone_flags.py (not runnable yet)")
    if args.task not in tasks:
        p.error(f"Unknown task {args.task!r}; see --list")
    entry, flags, task = models[args.model], FLAGS[args.model], tasks[args.task]
    if entry["modality"] != task["modality"]:
        p.error(f"Modality mismatch: {args.model} is {entry['modality']}, "
                f"{args.task} is {task['modality']}")
    if args.objective is None:
        if len(task["tasks"]) != 1:
            p.error(f"{args.task} registers {task['tasks']}; pass --objective")
        args.objective = task["tasks"][0]
    if args.objective not in task["tasks"]:
        p.error(f"{args.task} does not register {args.objective}")
    live = args.mode in ("opal", "biopal")
    if args.panel_frac is None:
        args.panel_frac = .15 if args.mode == "biopal" else 0.
    use_bio = args.mode in ("keep", "biopal") and args.panel_frac > 0
    if args.outer_audit == []:
        args.outer_audit = [16, 32, 64, 128]
    if (args.mode == "biopal" or args.bio_policy != "off" or args.outer_audit) and not use_bio:
        p.error("BiOPAL needs --panel-frac > 0 and --mode biopal (or keep, its offline twin)")
    if args.bio_policy == "expert" and (args.outer == "off" or not args.expert_script):
        p.error("--bio-policy expert needs --outer and --expert-script")
    if args.outer_steps < 1 or args.outer_decisions < 1 or not 0 < args.outer_alpha < 1 \
            or min(args.outer_audit or [1]) < 1:
        p.error("Invalid --outer-steps, --outer-decisions, --outer-alpha or --outer-audit")
    if args.epochs < 1 or args.batch_size < 1 or args.max_steps < 0 \
            or args.limit < 0 or args.lr <= 0 or args.decisions < 1 \
            or not 0 <= args.panel_frac < 1:
        p.error("Invalid epochs, batch size, max steps, limit, learning rate, decisions, "
                "or panel fraction")
    if live and (args.budget is None or args.budget <= 0):
        p.error("Live mode requires an explicit --budget, e.g. --budget 5")
    if live and args.max_steps:
        p.error("Use plain/keep for --max-steps diagnostics")
    budget_left = args.budget
    if live and args.ledger:
        sys.path.append(str(REPO))
        from biopal import ledger
        budget_left = args.budget - ledger.spent(args.ledger, "opal")
        if budget_left < .01:
            p.error(f"OPAL ceiling of ${args.budget} is used up in {args.ledger}; "
                    "continue offline with --mode keep")
    if args.out.exists():
        p.error(f"Output already exists; choose a new directory: {args.out}")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if flags.get("env") and conda_env and conda_env != flags["env"]:
        print(f"WARNING: {args.model} is served by conda env {flags['env']!r}, "
              f"running under {conda_env!r}", flush=True)

    if flags.get("block_triton"):
        # DNABERT-2's bundled flash_attn_triton only guards its ImportError
        # path; a None entry makes `import triton` raise -> fallback attention
        sys.modules["triton"] = None

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import (accuracy_score, matthews_corrcoef, r2_score,
                                 roc_auc_score)
    from smoke_backbone import encode_batch, forward_hidden, load_backbone

    classification = args.objective == "classification"
    label = "y_cls" if classification else "y_reg"
    device = "cpu" if args.cpu else "cuda:0"
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select an allocated GPU or pass --cpu")
    if args.det != "off":
        torch.use_deterministic_algorithms(True, warn_only=args.det == "warn")
    torch.backends.cudnn.benchmark = False
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def seed_all():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device != "cpu":
            torch.cuda.manual_seed_all(args.seed)

    def parameter_hash(module):
        h = hashlib.sha256()
        for name, param in sorted(module.named_parameters()):
            x = param.detach().cpu().contiguous()
            header = json.dumps([name, str(x.dtype), list(x.shape)]).encode()
            h.update(len(header).to_bytes(8, "little") + header)
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------------- data
    curated_base, cache_dirs = machine_paths()
    parquet = curated_base / args.task / f"task_{args.objective}.parquet"
    seed_all()
    df = pd.read_parquet(parquet, columns=["sequence", label, "split"])
    if df.isna().any().any():
        raise ValueError(f"Nulls in {parquet}")
    df = df.copy()
    df["sequence"] = df["sequence"].str.upper()
    if not df["sequence"].map(bool).all():
        raise ValueError("Empty sequences in the task data")
    if set(df["split"]) != {"train", "val", "test"}:
        raise ValueError("Expected train/val/test splits")
    if (df.groupby("sequence")["split"].nunique() > 1).any():
        raise ValueError("Exact sequence overlap across splits; fix leakage first")
    if classification:
        classes = sorted(df[label].unique())
        if classes != list(range(len(classes))) or len(classes) < 2:
            raise ValueError(f"Expected contiguous integer classes from 0, got {classes}")
        num_out = len(classes)
    else:
        if not np.isfinite(df[label]).all():
            raise ValueError("Non-finite regression targets")
        num_out = 1
    if args.max_chars:
        df["sequence"] = df["sequence"].str[:args.max_chars]
    frames = {}
    for s in ("train", "val", "test"):
        f = df[df["split"].eq(s)].reset_index(drop=True)
        if args.limit and len(f) > args.limit:
            if classification:
                per = max(1, args.limit // num_out)
                f = pd.concat([f[f[label] == c].head(per) for c in range(num_out)])
            else:
                f = f.head(args.limit)
            f = f.reset_index(drop=True)
        frames[s] = f
    if classification and any(f[label].nunique() < 2 for f in frames.values()):
        raise ValueError("Each split needs at least two classes for AUROC")
    fit, roles = np.arange(len(frames["train"])), None  # train rows that give gradients
    if args.panel_frac:
        sys.path.append(str(REPO))
        from biopal import panel as bio_panel
        roles = bio_panel.split(frames["train"], label, classification,
                                args.panel_frac, args.seed)
        fit = roles["F"]
    if classification:
        y_mean, y_std = 0.0, 1.0
    else:
        fit_y = frames["train"][label] if roles is None else frames["train"][label].iloc[fit]
        y_mean = float(fit_y.mean())
        y_std = float(fit_y.std())
        if not y_std > 0:
            raise ValueError("Constant regression targets in train")
    print("Split sizes:", {s: len(f) for s, f in frames.items()}, flush=True)
    if roles is not None:
        print("Train roles (fit F, diagnostic D, private A):",
              {k: len(v) for k, v in roles.items()}, flush=True)

    # ------------------------------------------------------------------ model
    cache = str(cache_dirs[entry["caches"][0]])
    tok, model, base = load_backbone(entry["repo_id"], entry["revision"], cache,
                                     trust_remote_code=flags.get("trust", False))
    model = model.float().to(device)  # FP32 everywhere (half ckpts upcast)
    if args.freeze_backbone:
        model.requires_grad_(False)
        model.eval()

    # spaced-residue probe (e.g. prot_t5 wants "M K T ..."), decided once
    _, spaced = encode_batch(tok, frames["train"]["sequence"].head(2).tolist(),
                             device)
    pad_multiple = flags.get("pad_multiple")
    if pad_multiple and tok.pad_token_id is None:
        raise RuntimeError(f"{args.model} needs pad_to_multiple_of={pad_multiple} "
                           "but its tokenizer has no pad token")
    hidden_size = None

    def encode(ss):
        xs = [" ".join(s) for s in ss] if spaced else list(ss)
        b = tok(xs, return_tensors="pt", padding=True, truncation=False,
                pad_to_multiple_of=pad_multiple)
        return {k: v.to(device) for k, v in b.items()
                if k in ("input_ids", "attention_mask")}

    def pool(hidden, batch):
        mask = batch.get("attention_mask")
        if mask is not None and mask.shape[1] != hidden.shape[1]:
            mask = None  # backbone changes sequence resolution (NTv3 U-Net)
        return ((hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                if mask is not None else hidden.mean(1))

    def predict(batch, layers=None):
        """Task output. With `layers` (BiOPAL layer probes) also the pooled output of those
        hidden states, from the same forward pass. After a BiOPAL readout switch the head
        reads the chosen block instead of the backbone's default output."""
        route = int(net["readout"].layer) if "readout" in net else -1
        if route < 0 and not layers:
            return head(pool(forward_hidden(base, batch), batch))
        hs = base(**batch, output_hidden_states=True).hidden_states
        out = head(net["readout"](pool(hs[route], batch)) if route >= 0
                   else pool(hs[-1], batch))
        return (out, {k: pool(hs[k], batch) for k in layers}) if layers else out

    with torch.no_grad():  # re-encode the probe: encode() honors pad_multiple
        hidden_size = forward_hidden(
            base, encode(frames["train"]["sequence"].head(2).tolist())).shape[-1]
    head = torch.nn.Linear(hidden_size, num_out).to(device)
    net = torch.nn.ModuleDict(dict(backbone=model, head=head))
    initial_hash = parameter_hash(net)
    blocks = None  # BiOPAL: number of blocks, if every layer's output can be read
    if use_bio:
        from biopal import outer as bio_outer
        blocks = bio_outer.blocks(
            base, encode(frames["train"]["sequence"].head(2).tolist()), forward_hidden)
        if blocks:  # the route of a readout switch travels with the weights (best.pt)
            net["readout"] = bio_outer.Readout(hidden_size).to(device)

    def batch(split, indices):
        rows = frames[split].iloc[np.asarray(indices)]
        x = encode(rows["sequence"].tolist())
        if classification:
            y = torch.tensor(rows[label].to_numpy(), dtype=torch.long, device=device)
        else:
            z = (rows[label].to_numpy(dtype=np.float64) - y_mean) / y_std
            y = torch.tensor(z, dtype=torch.float32, device=device)
        return x, y

    def loss_fn(out, y):
        return (F.cross_entropy(out, y) if classification
                else F.mse_loss(out.squeeze(-1), y))

    @torch.no_grad()
    def rows(indices, layers=None):
        """BiOPAL: per-row loss on train rows in evaluation mode (and the pooled output of the
        given hidden states); module modes are left as they were."""
        modes = [(m, m.training) for m in net.modules()]
        net.eval()
        losses, feats = [], {k: [] for k in layers or ()}
        for start in range(0, len(indices), 4 * args.batch_size):
            x, y = batch("train", indices[start:start + 4 * args.batch_size])
            out = predict(x, layers)
            if layers:
                out, hidden = out
                for k in layers:
                    feats[k].append(hidden[k].cpu())
            losses.append((F.cross_entropy(out, y, reduction="none") if classification
                           else F.mse_loss(out.squeeze(-1), y, reduction="none")).cpu())
        for m, was in modes:
            m.training = was
        return (torch.cat(losses).double().numpy(),
                {k: torch.cat(v).double().numpy() for k, v in feats.items()})

    @torch.no_grad()
    def evaluate(split):
        was_training = net.training
        net.eval()
        targets, outputs = [], []
        loss_sum = 0.0
        for start in range(0, len(frames[split]), args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, len(frames[split])))
            x, y = batch(split, indices)
            out = predict(x)
            if not torch.isfinite(out).all().item():
                raise RuntimeError("Non-finite evaluation outputs")
            if classification:
                loss_sum += F.cross_entropy(out, y, reduction="sum").item()
            else:
                loss_sum += F.mse_loss(out.squeeze(-1), y, reduction="sum").item()
            targets.extend(y.cpu().tolist())
            outputs.append(out.cpu())
        net.train(was_training)
        out = torch.cat(outputs)
        n = len(targets)
        if classification:
            probs = out.softmax(-1).numpy()
            preds = out.argmax(-1).tolist()
            auroc = (roc_auc_score(targets, probs[:, 1]) if num_out == 2 else
                     roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
            return dict(loss=loss_sum / n,
                        accuracy=accuracy_score(targets, preds),
                        mcc=matthews_corrcoef(targets, preds), auroc=auroc)
        z = out.squeeze(-1).numpy()
        yhat = z * y_std + y_mean
        yraw = np.asarray(targets) * y_std + y_mean
        return dict(loss=loss_sum / n,  # MSE in z-space (selection metric)
                    rmse=float(np.sqrt(np.mean((yhat - yraw) ** 2))),
                    mae=float(np.mean(np.abs(yhat - yraw))),
                    r2=float(r2_score(yraw, yhat)),
                    spearman=float(spearmanr(yraw, yhat).statistic),
                    pearson=float(pearsonr(yraw, yhat).statistic))

    # -------------------------------------------------------------- optimizer
    trainable = [v for v in net.parameters() if v.requires_grad]
    full_steps = args.epochs * math.ceil(len(fit) / args.batch_size)
    total_steps = min(args.max_steps, full_steps) if args.max_steps else full_steps
    groups = [dict(params=[v for v in trainable if v.ndim > 1],
                   weight_decay=args.weight_decay),
              dict(params=[v for v in trainable if v.ndim <= 1], weight_decay=0.)]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=args.lr, foreach=False)
    else:
        optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum,
                                    foreach=False)
    warmup = max(1, int(args.warmup_frac * full_steps))

    def lr_lambda(step):  # linear warmup then cosine decay over the FULL run
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, full_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    args.out.mkdir(parents=True, exist_ok=False)
    recipe = dict(model=args.model, repo_id=entry["repo_id"],
                  revision=entry["revision"], task=args.task,
                  objective=args.objective, seed=args.seed, epochs=args.epochs,
                  batch_size=args.batch_size, optimizer=args.optimizer,
                  momentum=args.momentum if args.optimizer == "sgd" else None,
                  lr=args.lr,
                  weight_decay=args.weight_decay, warmup_fraction=args.warmup_frac,
                  clip_norm=args.clip_norm, precision="float32",
                  scheduler="cosine", max_chars=args.max_chars, limit=args.limit,
                  freeze_backbone=args.freeze_backbone, det=args.det,
                  y_mean=y_mean, y_std=y_std, num_out=num_out,
                  spaced_input=spaced, pad_multiple=pad_multiple,
                  hidden_size=hidden_size)
    if roles is not None:
        recipe.update(panel_frac=args.panel_frac, panel=bio_panel.manifest(roles))
    data_hash = hashlib.sha256(parquet.read_bytes()).hexdigest()
    versions = {}
    for k in ("torch", "transformers", "numpy", "opal-optimizer", "pandas",
              "pyarrow", "scikit-learn", "scipy", "multimolecule"):
        try:
            versions[k] = importlib.metadata.version(k)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata = dict(recipe=recipe, versions=versions, mode=args.mode,
                    budget=args.budget, no_test=args.no_test,
                    decisions=args.decisions if args.mode != "plain" else None,
                    data_sha256=data_hash,
                    initial_sha256=initial_hash, hostname=socket.gethostname(),
                    conda_env=conda_env, device=device,
                    gpu=torch.cuda.get_device_name(0) if device != "cpu" else None,
                    n_params_trainable=sum(v.numel() for v in trainable),
                    full_steps=full_steps,
                    split_sizes={k: len(v) for k, v in frames.items()})
    (args.out / "config.json").write_text(json.dumps(metadata, indent=2))

    wrapped = args.mode != "plain"
    if wrapped:
        import opal
        extra = {} if args.mode == "keep" else dict(
            max_cost_usd=budget_left,
            backend_options=dict(
                model=os.environ.get("OPAL_MODEL", "claude-opus-5-5"),
                effort=os.environ.get("OPAL_EFFORT", "high"),
                unavailable_timeout=120,
                timeout=300,
            ))
        objective_metric = ("cross-entropy" if classification
                            else "mean-squared error")
        plan = dict(decisions=args.decisions, max_decisions=args.decisions)
        if use_bio:  # decisions evenly over 15-85% of the run, on OPAL's checks
            at = bio_outer.spaced(total_steps, args.decisions, args.check_every)
            plan = dict(schedule=at, max_decisions=len(at), every=args.check_every)
        optimizer = opal.wrap(
            optimizer, net, scheduler=scheduler, total_steps=total_steps,
            goal=f"Fine-tune {args.model} ({entry['repo_id']}) on the "
                 f"{entry['modality']} task {args.task} ({args.objective}); "
                 f"judged by held-out test {objective_metric} of the checkpoint "
                 f"selected by lowest end-of-epoch validation {objective_metric}. "
                 "Validation and test metrics are not provided to the agent.",
            backend="keep" if args.mode == "keep" else "claude",
            blocking=True, summarize=False, run_dir=str(args.out / "opal"),
            **plan, **extra)
    def update(indices, weights=None):
        """One optimizer update on train rows `indices`. The training loop and BiOPAL's trial
        branches both call it. `weights` (BiOPAL cohort weighting) reweights the rows for the
        gradient only; the loss that is logged and passed to OPAL stays the plain mean."""
        x, y = batch("train", indices)
        optimizer.zero_grad(set_to_none=True)
        out = predict(x)
        loss = loss_fn(out, y)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite training loss at step {scheduler.last_epoch + 1}")
        if weights is None:
            loss.backward()
        else:
            each = (F.cross_entropy(out, y, reduction="none") if classification
                    else F.mse_loss(out.squeeze(-1), y, reduction="none"))
            (each * weights).mean().backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.clip_norm,
                                       error_if_nonfinite=True)
        if wrapped:
            optimizer.step(loss=loss)
        else:
            optimizer.step()
        scheduler.step()
        return loss

    bio = None
    if use_bio:
        bio = bio_outer.Outer(args, types.SimpleNamespace(
            net=net, optimizer=optimizer, scheduler=scheduler, update=update, rows=rows,
            frame=frames["train"], label=label, roles=roles, classification=classification,
            y=(frames["train"][label].to_numpy() if classification else
               (frames["train"][label].to_numpy(dtype=np.float64) - y_mean) / y_std),
            num_out=num_out, modality=entry["modality"], task=args.task, model=args.model,
            device=device, per_epoch=math.ceil(len(fit) / args.batch_size),
            total=total_steps, blocks=blocks, out=args.out))

    # ------------------------------------------------------------------ train
    # Reset after wrapping; use an independent RNG for batch permutations.
    seed_all()
    order_rng = np.random.default_rng(args.seed)
    step, best_epoch, best_val = 0, None, None
    training_seconds = 0.0
    started = time.perf_counter()
    print(f"model={args.model} task={args.task} objective={args.objective} "
          f"mode={args.mode}; updates={total_steps}; output={args.out}", flush=True)
    try:
        with (args.out / "train.jsonl").open("w", buffering=1) as train_log, \
             (args.out / "validation.jsonl").open("w", buffering=1) as val_log:
            for epoch in range(1, args.epochs + 1):
                net.train()
                if args.freeze_backbone:
                    model.eval()  # stable features: no dropout in the backbone
                order = fit[order_rng.permutation(len(fit))]
                for start in range(0, len(order), args.batch_size):
                    if device != "cpu":
                        torch.cuda.synchronize()
                    indices = order[start:start + args.batch_size]
                    if bio:
                        bio.before(step + 1)
                    tick = time.perf_counter()
                    loss = update(indices, bio.weights(indices) if bio else None)
                    if device != "cpu":
                        torch.cuda.synchronize()
                    training_seconds += time.perf_counter() - tick
                    step += 1
                    row = dict(step=step, epoch=epoch, loss=loss.item())
                    train_log.write(json.dumps(row) + "\n")
                    if step == 1 or step % 25 == 0 or step == total_steps:
                        print(f"epoch={epoch} step={step}/{total_steps} "
                              f"loss={loss.item():.6f}", flush=True)
                    if bio and step in bio.gates:  # trial on the next updates, then go on
                        bio.gate(step, [order[s:s + args.batch_size] for s in range(
                            start + args.batch_size, len(order), args.batch_size)])
                    if step >= total_steps:
                        break
                if not args.max_steps:
                    metrics = evaluate("val")
                    val_log.write(json.dumps(dict(epoch=epoch, step=step, **metrics)) + "\n")
                    print("Validation:", json.dumps(metrics), flush=True)
                    if best_val is None or metrics["loss"] < best_val["loss"]:
                        best_val, best_epoch = metrics, epoch
                        torch.save(net.state_dict(), args.out / "best.pt")
                if step >= total_steps:
                    break
    finally:
        if wrapped:
            optimizer.close()
        if bio:
            bio_summary = bio.close()
    elapsed = time.perf_counter() - started
    final_hash = parameter_hash(net)
    test = None
    if not args.max_steps:
        net.load_state_dict(torch.load(args.out / "best.pt", map_location=device,
                                       weights_only=True))
        if bio:
            bio_summary["best_readout_block"] = bio.block()  # the route best.pt was saved with
        if not args.no_test:
            test = evaluate("test")
        tok.save_pretrained(args.out / "tokenizer")
    result = dict(metadata, steps=step, final_sha256=final_hash,
                  best_epoch=best_epoch, best_validation=best_val, test=test,
                  training_wall_seconds=training_seconds,
                  seconds_per_update=training_seconds / max(step, 1),
                  training_and_validation_wall_seconds=elapsed)
    if bio:
        result["biopal"] = bio_summary
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
