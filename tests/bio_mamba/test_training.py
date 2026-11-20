"""bio-mamba: short-training-run trainability check per model."""
import pytest

pytest.importorskip("mamba_ssm",
                    reason="not the bio-mamba env (mamba_ssm missing)")

from backbone_checks import training_check
from model_specs import MODELS


@pytest.mark.parametrize("name", MODELS)
def test_short_training_run(name, registry, cache_dirs, task_batch, device):
    training_check(name, MODELS, registry, cache_dirs, task_batch, device)
