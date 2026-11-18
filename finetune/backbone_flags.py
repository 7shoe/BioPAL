"""How to run fine-tuning per model

Keys must exist in models/backbones/models.yaml (=OK-only audit registry)

Flags:
  trust         pass trust_remote_code=True when loading
  
  block_triton  hide triton for the whole process (DNABERT-2's bundled
                flash_attn_triton crashes on modern triton)

  pad_multiple  pad tokenized batches to this multiple (NTv3 U-Net needs
                input lengths divisible by 128; its char tokenizer adds no
                special tokens)

  env           conda env that serves the model (informational; finetune.py
                warns on mismatch but does not refuse)
"""

FLAGS = {
    # biopal-v2 (validated 2026-11-17)
    "ntv2_50m": dict(trust=True, env="biopal-v2"),
    "ntv3_pretrained": dict(trust=True, pad_multiple=128, env="biopal-v2"),
    "ntv3_100m_pre": dict(trust=True, pad_multiple=128, env="biopal-v2"),
    "hyenadna_tiny_1k": dict(trust=True, env="biopal-v2"),
    "dnabert2_117m": dict(trust=True, block_triton=True, env="biopal-v2"),
    "esm2_t6_8m": dict(env="biopal-v2"),
    "esm2_t12_35m": dict(env="biopal-v2"),
    "esm2_t30_150m": dict(env="biopal-v2"),
    "esm2_t33_650m": dict(env="biopal-v2"),
    "ankh_base": dict(env="biopal-v2"),
    "prot_t5_xl_half": dict(env="biopal-v2"),
    # bio-rna (validated; multimolecule registers the classes)
    "rnafm": dict(env="bio-rna"),
    "splicebert": dict(env="bio-rna"),
    # deferred: bio-mamba (CUDA-only mamba-ssm)
    "caduceus_ps_131k": dict(trust=True, env="bio-mamba"),
    "plant_caduceus_l20": dict(trust=True, env="bio-mamba"),
}
