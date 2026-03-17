# Spec S5.2 — Inference Optimization (Faster Inference)

## Overview

Optimize the ensemble inference pipeline (`submission/main.py`) for speed on the A100 runtime, freeing time budget for larger beam search and more reliable ensemble passes. Uses only pre-installed runtime packages (no faster-whisper/CTranslate2 — not available offline).

## Depends On

- **S3.3** (Ensemble inference pipeline) — done

## Motivation

The current pipeline uses vanilla HuggingFace `model.generate()` with `num_beams=5` and `batch_size=16`. On A100 (80GB VRAM), there's significant headroom:
1. **SDPA (Flash Attention)** is available in transformers >=4.44 but not explicitly enabled
2. **Batch size** can be much larger on 80GB VRAM (32-64 for large-v3, 64-128 for small)
3. **torch.compile()** can give 20-40% decode speedup with PyTorch >=2.1
4. **Dynamic batching** — grouping similar-length utterances reduces padding waste (already sorting by duration, but batch size is static)

The speed gains free time for: larger beam search (num_beams=8), multi-pass inference, or guaranteed ensemble completion.

## Requirements

### R1: SDPA / Flash Attention
- Load models with `attn_implementation="sdpa"` for Scaled Dot Product Attention
- This uses FlashAttention-2 on A100 (CUDA) and falls back gracefully on MPS/CPU
- No code change needed at generate() call site

### R2: torch.compile() Support
- Optionally compile the model with `torch.compile(model, mode="reduce-overhead")`
- Guard behind a flag (default ON for CUDA, OFF for MPS/CPU — MPS compile is unreliable)
- Compile only the encoder (decoder compile has diminishing returns and longer warmup)

### R3: Dynamic Batch Sizing
- Auto-detect available VRAM and set batch size accordingly
- A100 80GB: batch_size=32 for large-v3, batch_size=64 for small
- T4 16GB / MPS: keep batch_size=16 (current default)
- CPU: batch_size=4
- Expose as a function `get_optimal_batch_size(device, model_size)` for testability

### R4: Enhanced Beam Search
- Increase default `num_beams` from 5 to 8 for large-v3 (speed gains offset the cost)
- Keep `num_beams=5` for small model (diminishing returns)
- Add `length_penalty=1.0` explicitly (Whisper default)

### R5: Batch-Level Padding Optimization
- Group utterances into duration buckets before batching (already sorted by duration)
- Within each batch, pad to max length in the batch (not global max)
- This is already handled by the processor, but ensure we're not accidentally padding to 30s

### R6: Backward Compatibility
- All optimizations must be transparent to the rest of the pipeline
- `run_inference()` and `run_ensemble_inference()` signatures unchanged
- MacBook (MPS/CPU) testing still works — optimizations gracefully degrade
- All existing tests continue to pass

## Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Model A (large-v3) throughput | ~X utt/sec | ~2-3X utt/sec |
| Model B (small) throughput | ~Y utt/sec | ~2-3X utt/sec |
| Default beam search (large) | 5 beams | 8 beams |
| Batch size (A100, large) | 16 | 32 |
| Batch size (A100, small) | 16 | 64 |
| Existing tests | pass | pass |

## Files Modified

- `submission/main.py` — SDPA loading, torch.compile, dynamic batch sizing, beam config
- `tests/test_faster_inference.py` — New test file for S5.2

## TDD Notes

### Tests to write first:
1. `test_get_optimal_batch_size_cuda` — returns 32 for large model on CUDA
2. `test_get_optimal_batch_size_cpu` — returns 4 on CPU
3. `test_get_optimal_batch_size_mps` — returns 16 on MPS
4. `test_sdpa_attn_requested` — model loaded with `attn_implementation="sdpa"`
5. `test_compile_enabled_cuda` — torch.compile called on CUDA
6. `test_compile_disabled_cpu` — torch.compile NOT called on CPU/MPS
7. `test_large_model_beam_config` — num_beams=8 for large model
8. `test_small_model_beam_config` — num_beams=5 for small model
9. `test_transcribe_batch_accepts_num_beams` — num_beams parameter forwarded
10. `test_existing_inference_still_works` — backward compatibility
11. `test_existing_ensemble_still_works` — backward compatibility
