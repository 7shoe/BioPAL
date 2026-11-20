"""Offline CPU tests for biopal/. No model downloads, no Claude calls:
    conda run -n biopal-v2 python -m pytest tests/biopal -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repo root, for `import biopal`
sys.path.insert(0, str(Path(__file__).resolve().parent))       # toy.py
torch.set_num_threads(2)


import pytest   # noqa: E402


@pytest.fixture
def endpoint():
    """A fake local model server (tests/biopal/local_server.py)."""
    from local_server import Endpoint
    server = Endpoint()
    yield server
    server.close()


@pytest.fixture
def note(tmp_path):
    """A private note file bound to the toy task."""
    path = tmp_path / "note.txt"
    path.write_text("task: toy_task\nlabel: simulated institutional knowledge\n\n"
                    "In our hands the GC-rich rows are the noisy ones.\nUpweighting them has not paid off.\n")
    return path
