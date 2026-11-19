"""Offline CPU tests for biopal/. No model downloads, no Claude calls:
    conda run -n biopal-v2 python -m pytest tests/biopal -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the repo root, for `import biopal`
sys.path.insert(0, str(Path(__file__).resolve().parent))       # toy.py
torch.set_num_threads(2)
