"""The models the bio-rna env serves (multimolecule registers their classes
with transformers on import; no trust_remote_code needed)."""

MODELS = {
    "rnafm": dict(modality="rna"),
    "splicebert": dict(modality="rna"),
}
