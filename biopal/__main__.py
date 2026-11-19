"""Print one gate of a BiOPAL run as a readable trace, from checkpoint diagnostics to the next committed update:

    python -m biopal RUN_DIR [GATE]        (GATE counts from 1; default: the first gate that ran a trial)
"""
import json
import sys
from pathlib import Path


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def show(action, target):
    return action + (f" block_{target}" if "readout" in action or "refit" in action else f" {target}" if target else "")


def trace(run, gate=None):
    run = Path(run)
    log, config = lines(run / "outer.jsonl"), json.loads((run / "config.json").read_text())
    losses = {r["step"]: r["loss"] for r in lines(run / "train.jsonl")}
    of = lambda kind, g: next((r for r in log if r["event"] == kind and r.get("gate") == g), None)   # noqa: E731
    if gate is None:
        gate = min([r["gate"] for r in log if r["event"] == "trial"] or [0])
    table, proposals, trial, decision = (of(k, gate) for k in ("table", "proposals", "trial", "decision"))
    if table is None:
        sys.exit(f"no gate {gate + 1} in {run}")
    before, cited = table["config"], {}
    for role in ("informed", "masked"):
        for row in (proposals.get(role) or {}).get("evidence_rows") or []:
            cited.setdefault(row, []).append(role)
    print(f"Gate {gate + 1} after update {table['update']} ({config['recipe']['model']} on {config['recipe']['task']})")
    print(f"\n1. Checkpoint. Loss weights: {before['cohort'] or 'ordinary'}; the head reads block {before['readout_block']}."
          f" Loss on the diagnostic set {table['panel_loss']:.4f}, on the fit sample {table['fit_loss']:.4f}.")
    whole = [r for r in table["table"] if r["cohort"] == "all"]
    extreme = sorted((r for r in table["table"] if r["cohort"] != "all"), key=lambda r: -abs(r["gap"]))[:4]
    print(f"   Diagnostics ({len(table['table'])} rows; shown: the whole panel, rows an expert cited, the largest gaps):")
    for r in sorted({r["id"]: r for r in whole + extreme + [r for r in table["table"] if r["id"] in cited]}.values(),
                    key=lambda r: r["id"]):
        kind = r["measurement"] + (f" block_{r['layer']}" if r["layer"] is not None else "")
        note = f"   <- cited by {' and '.join(cited[r['id']])}" if r["id"] in cited else ""
        print(f"     {r['id']}  {kind:15s} {r['cohort']:22s} n={r['n']:<5d} {r['gap']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{note}")
    print("\n2. Proposals, saved before the private set was read:")
    for role in ("informed", "masked"):
        p = proposals.get(role)
        if p:
            said = f"predicted the private loss to be {p.get('predicted_sign')}. \"{p.get('hypothesis')}\"" if p["ok"] \
                else f"failed: {p['error']}"
            print(f"     {role:9s} {show(p['action'], p['target'])}; {said}")
    if not any(proposals.get(r) for r in ("informed", "masked")):
        print("     none (audit only)")
    if trial:
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
    after = decision["config_after"]
    what = f"adopted {show(decision['action'], decision['target'])}" + (" (forced by a test script)" if decision["forced"] else "")
    print(f"\n4. Decision: {what if decision['adopted'] else 'kept the incumbent'}. Loss weights: "
          f"{after['cohort'] or 'ordinary'}; the head reads block {after['readout_block']}.")
    for role, p in decision["proposals"].items():
        if p.get("sign_correct") is not None:
            print(f"     {role} predicted {p['predicted_sign']}: {'right' if p['sign_correct'] else 'wrong'}"
                  + ("" if role == "informed" else f" (its proposal {p['outcome']}; a shadow is never adopted)"))
    print(f"     OPAL: {decision['opal_calls_in_trial']} agent calls inside trial branches so far; program changed by the "
          f"trial: {decision['opal_program_changed']}; its updater reads ctx.loss: {decision['loss_aware_updater']}")
    step = decision["next_update"]
    if trial and step in losses:
        won = next((b for b in trial["branches"] if decision["adopted"] and (b["action"], b["target"]) == (
            decision["action"], decision["target"])), trial["incumbent"])
        same = "the same value as" if won["losses"][0] == losses[step] else "close to" if abs(
            won["losses"][0] - losses[step]) < 1e-3 else "NOT equal to"
        print(f"\n5. Next committed update {step}: training loss {losses[step]:.6f}, {same} the first update of the "
              f"{'adopted' if decision['adopted'] else 'incumbent'} branch ({won['losses'][0]:.6f}).")
    elif step in losses:
        print(f"\n5. Next committed update {step}: training loss {losses[step]:.6f}.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    trace(sys.argv[1], int(sys.argv[2]) - 1 if len(sys.argv) > 2 else None)
