#!/usr/bin/env python3
"""Hyperparameter optimization (HPO) over finetune.py

Strategies are random search (RS) and Bayesian optimization (BO)

Each trial is one `finetune.py --no-test` run (subprocess, same conda env, same
training seed), scored by its best validation loss. Both modes search the same
space (lr log-uniform, weight decay, warmup fraction) and both start with the
default recipe as trial 0, so neither can end worse than untuned defaults.

  --mode RS   Optuna RandomSampler(seed)
  --mode BO   Optuna GPSampler(seed): Gaussian-process BO after
              max(3, n_trials // 4) random startup trials

The study lives in <out>/study.db; rerunning the same command resumes it up to
--n_trials finished trials. Test data are never evaluated here: score the
winner once with the command printed at the end (also in hpo_result.json).

Any unrecognized flags are forwarded to finetune.py, e.g.
  python hpo.py --model ntv2_50m --task nt_tasks_official_splice_sites_donors \
      --mode BO --n_trials 12 --out runs/hpo_bo_ntv2_donors --epochs 2
"""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import optuna

HERE = Path(__file__).resolve().parent
DEFAULTS = dict(lr=2e-5, weight_decay=.01, warmup_frac=.1)   # finetune.py defaults
SEARCHED = {"--lr", "--weight-decay", "--warmup-frac"}
OWNED = SEARCHED | {"--model", "--task", "--objective", "--seed", "--out",
                    "--no-test", "--mode", "--budget"}


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--objective", choices=["classification", "regression"])
    p.add_argument("--mode", choices=["RS", "BO"], required=True)
    p.add_argument("--n_trials", type=int, required=True)
    p.add_argument("--seed", type=int, default=42,
                   help="training seed for every trial and the sampler seed")
    p.add_argument("--lr-range", type=float, nargs=2, default=(1e-5, 1e-3))
    p.add_argument("--wd-range", type=float, nargs=2, default=(0.0, 0.1))
    p.add_argument("--warmup-range", type=float, nargs=2, default=(0.0, 0.2))
    p.add_argument("--out", type=Path, required=True)
    args, passthrough = p.parse_known_args()
    clash = OWNED & {a.split("=")[0] for a in passthrough if a.startswith("--")}
    if clash:
        p.error(f"hpo.py controls {sorted(clash)}; do not pass them through")
    if args.n_trials < 1:
        p.error("--n_trials must be >= 1")

    config = dict(model=args.model, task=args.task, objective=args.objective,
                  mode=args.mode, seed=args.seed, lr_range=list(args.lr_range),
                  wd_range=list(args.wd_range), warmup_range=list(args.warmup_range),
                  passthrough=passthrough)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg_file = args.out / "hpo_config.json"
    if cfg_file.exists():
        old = json.loads(cfg_file.read_text())
        if old != config:
            sys.exit(f"{args.out} holds a different study: {old}")
    else:
        cfg_file.write_text(json.dumps(config, indent=2))

    if args.mode == "RS":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.GPSampler(seed=args.seed,
                                            n_startup_trials=max(3, args.n_trials // 4))
    study = optuna.create_study(study_name="hpo", direction="minimize", sampler=sampler,
                                storage=f"sqlite:///{args.out / 'study.db'}",
                                load_if_exists=True)
    if not study.trials:
        study.enqueue_trial(DEFAULTS)
    (args.out / "trials").mkdir(exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)

    def base_cmd(hp, out):
        cmd = [sys.executable, str(HERE / "finetune.py"), "--model", args.model,
               "--task", args.task, "--seed", str(args.seed), "--out", str(out),
               "--lr", repr(hp["lr"]), "--weight-decay", repr(hp["weight_decay"]),
               "--warmup-frac", repr(hp["warmup_frac"])]
        if args.objective:
            cmd += ["--objective", args.objective]
        return cmd + passthrough

    def objective(trial):
        hp = dict(lr=trial.suggest_float("lr", *args.lr_range, log=True),
                  weight_decay=trial.suggest_float("weight_decay", *args.wd_range),
                  warmup_frac=trial.suggest_float("warmup_frac", *args.warmup_range))
        out = args.out / "trials" / f"trial_{trial.number:03d}"
        trial.set_user_attr("dir", str(out))
        log = args.out / "logs" / f"trial_{trial.number:03d}.log"
        print(f"[hpo] trial {trial.number} {hp}", flush=True)
        with log.open("w") as f:
            rc = subprocess.run(base_cmd(hp, out) + ["--no-test"],
                                stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"finetune.py exited {rc}; see {log}")
        value = json.loads((out / "result.json").read_text())["best_validation"]["loss"]
        print(f"[hpo] trial {trial.number} val_loss={value:.6f}", flush=True)
        return value

    done = sum(t.state.is_finished() for t in study.trials)
    if done < args.n_trials:
        study.optimize(objective, n_trials=args.n_trials - done, catch=(RuntimeError,))

    trials = [dict(number=t.number, state=t.state.name, params=t.params,
                   val_loss=t.value, dir=t.user_attrs.get("dir")) for t in study.trials]
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    result = dict(config, n_trials=len(trials),
                  n_failed=sum(t["state"] == "FAIL" for t in trials), trials=trials)
    if complete:
        best = study.best_trial
        final = args.out / "final"
        cmd = base_cmd(best.params, final)
        cmd[:2] = ["python", "finetune.py"]
        result.update(best=dict(number=best.number, params=best.params,
                                val_loss=best.value, dir=best.user_attrs.get("dir")),
                      final_command=shlex.join(cmd))
    (args.out / "hpo_result.json").write_text(json.dumps(result, indent=2))

    print(f"\n[hpo] {args.mode}: {len(complete)} complete, {result['n_failed']} failed "
          f"-> {args.out / 'hpo_result.json'}")
    if complete:
        print(f"[hpo] best trial {best.number}: val_loss={best.value:.6f} {best.params}")
        print(f"[hpo] score it once on test:\n  {result['final_command']}")


if __name__ == "__main__":
    main()
