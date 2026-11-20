"""The models the bio-mamba env serves (both hard-import mamba_ssm from their
remote code)."""

MODELS = {
    "caduceus_ps_131k": dict(modality="dna", trust=True),
    "plant_caduceus_l20": dict(modality="dna", trust=True),
}
