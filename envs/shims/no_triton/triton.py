# Import shim: makes `import triton` fail with ImportError so DNABERT-2's
# guarded flash_attn_triton import falls back to standard attention.
# Its bundled triton kernel uses the long-removed `trans_b=` API and raises a
# CompilationError (not ImportError) on modern triton, which the model code
# does not catch. Use per-process: PYTHONPATH=envs/shims/no_triton python ...
raise ImportError("triton hidden by envs/shims/no_triton (DNABERT-2 fallback)")
