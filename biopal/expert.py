"""BiOPAL's biological expert: one proposal per gate.

An answer is a dict with `action` (set_cohort, ordinary_weighting, set_readout or keep) and `target` (the cohort,
or the block; "" otherwise).
Scripted answers stand in for the Claude call in tests and offline runs.
"""
import json
from pathlib import Path


class Scripted:
    """Answers read from a JSON file {"informed": [...], "masked": [...]}: entry i answers gate i, a missing entry
    keeps. An entry may carry "force": true to adopt its change without the trial margin (tests only)."""

    def __init__(self, path):
        self.plan = json.loads(Path(path).read_text())

    def propose(self, role, gate, **context):
        plan = self.plan.get(role, [])
        return dict(plan[gate]) if gate < len(plan) else dict(action="keep", target="")


def check(answer, menu):
    """(action, target) if the answer is one of the actions offered at this gate; ValueError otherwise."""
    action, target = answer.get("action"), answer.get("target") or None
    if action == "keep" or (action == "ordinary_weighting" and menu.get("ordinary_weighting")):
        return action, None
    if action in ("set_cohort", "set_readout") and target in menu.get(action, []):
        return action, target
    raise ValueError(f"unavailable action: {action} {target}")
