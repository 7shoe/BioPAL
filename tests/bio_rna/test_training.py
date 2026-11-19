"""bio-rna: short-training-run trainability check per model."""
import pytest

pytest.importorskip("multimolecule",
                    reason="not the bio-rna env (multimolecule missing)")

from backbone_checks import training_check
from model_specs import MODELS


@pytest.mark.parametrize("name", MODELS)
def test_short_training_run(name, registry, cache_dirs, task_batch, device):
    training_check(name, MODELS, registry, cache_dirs, task_batch, device)
