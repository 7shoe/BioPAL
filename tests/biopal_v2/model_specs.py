"""The models the biopal-v2 env serves, and how to drive each one.
Flags are explained in envs/README.md."""

MODELS = {
    "ntv2_50m": dict(modality="dna", trust=True),
    "ntv3_pretrained": dict(modality="dna", trust=True),
    "ntv3_100m_pre": dict(modality="dna", trust=True),
    "hyenadna_tiny_1k": dict(modality="dna", trust=True),
    "esm2_t6_8m": dict(modality="protein"),
    "esm2_t12_35m": dict(modality="protein"),
    "esm2_t30_150m": dict(modality="protein"),
    "esm2_t33_650m": dict(modality="protein"),
    "ankh_base": dict(modality="protein"),
    "prot_t5_xl_half": dict(modality="protein"),
    # last: it blocks the triton import for the rest of the process
    "dnabert2_117m": dict(modality="dna", trust=True, block_triton=True),
}
