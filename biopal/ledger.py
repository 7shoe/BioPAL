"""Cumulative Claude spend across runs: one JSON line per charge, appended as soon as the cost is known. The file
lives outside every run and every training checkpoint, so rewinding a trial branch cannot undo a charge."""
import json
import time
from pathlib import Path


def spent(path, role):
    """USD charged so far to `role` ("opal" or "expert"); 0 without a ledger."""
    if path is None or not Path(path).exists():
        return 0.
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return sum(r["usd"] for r in rows if r["role"] == role)


def charge(path, role, usd, **note):
    if path is not None:
        with Path(path).open("a") as f:
            f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), role=role, usd=round(float(usd), 6),
                                    **note)) + "\n")
