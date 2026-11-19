"""Print one gate of a BiOPAL run as a readable trace, from checkpoint diagnostics to the next committed update:

    python -m biopal RUN_DIR [GATE]
    
Gate numbers start at 1. By default, show the first gate that ran a trial or gate 1 if none did
"""
import json
import sys
from pathlib import Path

def lines(path):
    """Read JSONL records from fiele & skip blank lines"""
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def show(action, target):
    """Format an action for display, using block labels for readout."""
    
    return action + (
        f" block_{target}" if "readout" in action or "refit" in action
        else f" {target}" if target else ""
    )

def trace(run, gate=None):
    """Print the diagnostics, proposals, trial outcome, and decision for one gate.

    An explicit gate index is zero-based. If a trial ran, compare its selected
    branch with the training losses recorded after the decision.
    """
    run = Path(run)
    log, config = lines(run / "outer.jsonl"), json.loads((run / "config.json").read_text())
    losses = {r["step"]: r["loss"] for r in lines(run / "train.jsonl")}
    of = lambda kind, g: next((r for r in log if r["event"] == kind and r.get("gate") == g), None)
    
    # use the 1st gate that ran a trial, or gate 0 if none did
    if gate is None:
        gate = min([r["gate"] for r in log if r["event"] == "trial"] or [0])
    table, proposals, trial, decision = (of(k, gate) for k in ("table", "proposals", "trial", "decision"))
    if table is None:
        sys.exit(f"no gate {gate + 1} in {run}")
    
    # track which expert cited each diagnostic row
    before, cited = table["config"], {}
    for role in ("informed", "masked"):
        for row in (proposals.get(role) or {}).get("evidence_rows") or []:
            cited.setdefault(row, []).append(role)
    print(
        f"Gate {gate + 1} after update {table['update']} "
        f"({config['recipe']['model']} on {config['recipe']['task']})"
    )
    print(
        f"\n1. Checkpoint. Loss weights: {before['cohort'] or 'ordinary'}; "
        f"the head reads block {before['readout_block']}. "
        f"Loss on the diagnostic set {table['panel_loss']:.4f}, "
        f"on the fit sample {table['fit_loss']:.4f}."
    )

    # show whole-panel rows, cited rows, and the four largest cohort gaps by magnitude
    whole = [r for r in table["table"] if r["cohort"] == "all"]
    extreme = sorted((r for r in table["table"] if r["cohort"] != "all"), key=lambda r: -abs(r["gap"]))[:4]
    
    print(
        f"   Diagnostics ({len(table['table'])} rows; shown: the whole panel, "
        f"rows an expert cited, the largest gaps):"
    )
    
    for r in sorted({r["id"]: r for r in whole + extreme + [r for r in table["table"] if r["id"] in cited]}.values(),
                    key=lambda r: r["id"]):
        kind = r["measurement"] + (f" block_{r['layer']}" if r["layer"] is not None else "")
        note = f"   <- cited by {' and '.join(cited[r['id']])}" if r["id"] in cited else ""
        print(f"     {r['id']}  {kind:15s} {r['cohort']:22s} n={r['n']:<5d} {r['gap']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{note}")
    print("\n2. Proposals, saved before the private set was read:")
    
    # role-dependent: show what each expert proposed before seeing the trial result
    for role in ("informed", "masked"):
        p = proposals.get(role)
        if p:
            said = f"predicted the private loss to be {p.get('predicted_sign')}. \"{p.get('hypothesis')}\"" if p["ok"] \
                else f"failed: {p['error']}"
            print(f"     {role:9s} {show(p['action'], p['target'])}; {said}")
    if not any(proposals.get(r) for r in ("informed", "masked")):
        print("     none (audit only)")
    
    if trial:
        # compare branches at the final trial update.
        k = str(trial["trial_updates"])
        print(f"\n3. Matched trial: {k} updates from one snapshot per branch, scored on the "
              f"{config['recipe']['panel']['A']['n']} private rows.")
        print(f"     incumbent: private loss {trial['incumbent']['loss_A'][k]:.4f}")
        for b in trial["branches"]:
            who = "+".join(b["by"])
            if "failed" in b:
                print(f"     {show(b['action'], b['target'])} [{who}]: failed ({b['failed']})")
                continue
            v = b["horizons"][k]
            print(f"     {show(b['action'], b['target'])} [{who}]: private loss differs by {v['mean_d']:+.4f} "
                  f"(SE {v['se']:.4f}) -> {'passes' if v['passes'] else 'does not pass'} (needs below {-v['se']:.4f})")
    else:
        print("\n3. No trial: no proposal differed from the configuration in force.")
    
    # show which loss weights and readout block training continued with
    after = decision["config_after"]
    what = f"adopted {show(decision['action'], decision['target'])}" + (" (forced by a test script)" if decision["forced"] else "")
    print(
        f"\n4. Decision: {what if decision['adopted'] else 'kept the incumbent'}. "
        f"Loss weights: {after['cohort'] or 'ordinary'}; "
        f"the head reads block {after['readout_block']}."
    )
    
    # report whether each expert predicted the direction of the private loss change
    for role, p in decision["proposals"].items():
        if p.get("sign_correct") is not None:
            print(f"     {role} predicted {p['predicted_sign']}: {'right' if p['sign_correct'] else 'wrong'}"
                  + ("" if role == "informed" else f" (its proposal {p['outcome']}; a shadow is never adopted)"))
    print(f"     OPAL: {decision['opal_calls_in_trial']} agent calls inside trial branches so far; program changed by the "
          f"trial: {decision['opal_program_changed']}; its updater reads ctx.loss: {decision['loss_aware_updater']}")
    
    # check if training reproduced the losses from the selected branch
    step = decision["next_update"]
    
    # ??? YYY
    if trial and step in losses:
        won = next((b for b in trial["branches"] if decision["adopted"] and (b["action"], b["target"]) == (
            decision["action"], decision["target"])), trial["incumbent"])
        redone = [losses[step + i] for i in range(len(won["losses"])) if step + i in losses]
        
        # count matching losses up to the 1st mismatch
        equal = next((i for i, (a, b) in enumerate(zip(redone, won["losses"])) if a != b), len(redone))
        same = (
            f"all {equal} recorded updates of the window repeat"
            if equal == len(redone) else (
                f"only the first {equal} of {len(redone)} updates repeat (largest loss difference "
                f"{max(abs(a - b) for a, b in zip(redone, won['losses'])):.2g}; "
                f"the GPU arithmetic was not reproducible)"
            )
        )
        print(
            f"\n5. Next committed update {step}: training loss {losses[step]:.6f}. "
            f"Training went on from the snapshot: {same} "
            f"the {'adopted' if decision['adopted'] else 'incumbent'} branch exactly."
        )
    elif step in losses:
        print(f"\n5. Next committed update {step}: training loss {losses[step]:.6f}.")

# entry
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    
    # convert the optional gate number to the zero-based index used in the log
    trace(sys.argv[1], int(sys.argv[2]) - 1 if len(sys.argv) > 2 else None)
