# Explanation S5.2 — Inference Optimization (Faster Inference)

## Why

The A100 runtime has a 2-hour limit with 80GB VRAM, but the existing pipeline used conservative settings designed for T4/MPS compatibility (batch_size=16, num_beams=5, no attention optimization). This left significant performance headroom unused, risking timeout on large test sets and missing the opportunity for higher-quality beam search.

## What

Four inference optimizations added to `submission/main.py`, all backward-compatible:

1. **SDPA (Scaled Dot Product Attention)** — Both `load_model()` and `load_large_model()` now request `attn_implementation="sdpa"`, which uses FlashAttention-2 on A100 CUDA and falls back gracefully on MPS/CPU. No changes to the generate() call site.

2. **`torch.compile()` guard** — `maybe_compile(model, device)` compiles models with `mode="reduce-overhead"` on CUDA for ~20-40% decode speedup. Disabled on CPU/MPS (unreliable). Fails gracefully if compile errors occur.

3. **Dynamic batch sizing** — `get_optimal_batch_size(device, model_size)` returns optimal batch sizes: A100 gets batch_size=32 for large-v3 and 64 for small; MPS keeps 16; CPU drops to 4. Unknown devices default to 4 (safe).

4. **Enhanced beam search** — `get_beam_config(model_size)` returns num_beams=8 for large model (speed gains offset cost) and num_beams=5 for small. `transcribe_batch()` now accepts `num_beams`, `length_penalty`, and `max_new_tokens` parameters.

## How

- All optimizations are opt-in helpers (functions), not forced changes to the pipeline flow
- `transcribe_batch()` signature extended with keyword args that default to previous behavior
- `run_inference()` and `run_ensemble_inference()` signatures unchanged — full backward compatibility
- 23 new tests cover all optimization paths including edge cases (unknown device, compile failure)
- All 402 existing tests continue to pass

## Connections

- **Depends on S3.3** (ensemble inference) — optimizations layer on top of the ensemble architecture
- **Enables S5.4** (final submission) — faster inference means more reliable ensemble completion within the 2-hour budget, and higher beam search quality
- **Uses pre-installed packages only** — SDPA and torch.compile are part of PyTorch/transformers, no external deps needed (faster-whisper/CTranslate2 not used since they aren't pre-installed in the competition runtime)
