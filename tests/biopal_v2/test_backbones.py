"""biopal-v2: offline load + forward for every model this env serves."""
import pytest
from backbone_checks import forward_check
from model_specs import MODELS


@pytest.mark.parametrize("name", MODELS)
def test_load_and_forward(name, registry, cache_dirs, sample, device):
    forward_check(name, MODELS, registry, cache_dirs, sample, device)
