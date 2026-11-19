"""Shared fixtures for all per-env test directories (see tests/README.md)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "models" / "backbones"))  # smoke_backbone helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))  # backbone_checks

import pytest
import yaml


def pytest_addoption(parser):
    parser.addoption("--cpu", action="store_true",
                     help="force CPU even when CUDA is available")


@pytest.fixture(scope="session")
def device(request) -> str:
    import torch
    if request.config.getoption("--cpu") or not torch.cuda.is_available():
        return "cpu"
    return "cuda:0"


def _paths_cfg() -> dict:
    with open(REPO / "paths" / "paths.yaml") as f:
        return yaml.safe_load(f)


def _existing_base(cfg: dict, key: str) -> Path | None:
    for machine in cfg["machines"].values():
        base = (machine or {}).get(key)
        if base and Path(base).is_dir():
            return Path(base)
    return None


@pytest.fixture(scope="session")
def registry() -> dict:
    """models/backbones/models.yaml — the generated OK-only model registry."""
    with open(REPO / "models" / "backbones" / "models.yaml") as f:
        return yaml.safe_load(f)["models"]


@pytest.fixture(scope="session")
def cache_dirs() -> dict:
    """cache key (hf/dna_rna/protein) -> absolute dir, for this machine."""
    cfg = _paths_cfg()
    base = _existing_base(cfg, "models_base")
    if base is None:
        pytest.skip("no machine in paths/paths.yaml has an existing models_base")
    return {k: str(base / v) for k, v in cfg["caches"].items()}


@pytest.fixture(scope="session")
def sample() -> dict:
    """Synthetic sequences from tests/sample_data, one per line."""
    d = REPO / "tests" / "sample_data"
    return {m: (d / f"{m}.txt").read_text().split()
            for m in ("dna", "rna", "protein")}


# one representative curated task per modality (all in data/processed/tasks.yaml)
SAMPLE_TASKS = {"dna": "nt_tasks_official_splice_sites_donors",
                "rna": "rfam_trna_easy",
                "protein": "cds_fluorescence"}
# truncation keeps CPU runs fast; 128 is the NTv3 U-Net length granule
TRUNC = {"dna": 128, "rna": 128, "protein": 96}


@pytest.fixture(scope="session")
def task_batch(sample):
    """modality -> (sequences, labels, source): a balanced 4-sample batch from
    the curated task, or from tests/sample_data when the store is absent."""
    curated = _existing_base(_paths_cfg(), "curated_base")

    def get(modality: str):
        name, n_chars = SAMPLE_TASKS[modality], TRUNC[modality]
        pq = curated / name / "task_classification.parquet" if curated else None
        if pq is not None and pq.exists():
            import pandas as pd
            df = pd.read_parquet(pq, columns=["sequence", "split", "y_cls"])
            df = df[df["split"] == "train"]
            rows = pd.concat([df[df["y_cls"] == c].head(2) for c in (0, 1)])
            if len(rows) == 4:
                return ([s[:n_chars] for s in rows["sequence"]],
                        [int(y) for y in rows["y_cls"]], name)
        return [s[:n_chars] for s in sample[modality]], [0, 1, 0, 1], "synthetic"

    return get
