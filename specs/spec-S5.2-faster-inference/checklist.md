# Checklist S5.2 — Inference Optimization (Faster Inference)

## Phase 1: Red — Write Tests First
- [x] Create `tests/test_faster_inference.py`
- [x] Test `get_optimal_batch_size()` for CUDA/MPS/CPU and large/small models
- [x] Test SDPA attention implementation requested on model load
- [x] Test torch.compile enabled on CUDA, disabled on CPU/MPS
- [x] Test beam search config: num_beams=8 for large, 5 for small
- [x] Test `transcribe_batch` accepts and forwards `num_beams` parameter
- [x] Test backward compatibility: existing inference and ensemble still work
- [x] Run tests — all new tests FAIL (Red)

## Phase 2: Green — Implement
- [x] Add `get_optimal_batch_size(device, model_size)` to `submission/main.py`
- [x] Update `load_model()` to use `attn_implementation="sdpa"`
- [x] Update `load_large_model()` to use `attn_implementation="sdpa"`
- [x] Add `maybe_compile(model, device)` helper
- [x] Add `get_beam_config(model_size)` helper
- [x] Update `transcribe_batch()` to accept `num_beams` and `length_penalty` parameters
- [x] Run tests — all 23 tests PASS (Green)

## Phase 3: Refactor
- [x] Run ruff linter — clean
- [x] Ensure all 402 existing tests still pass
- [x] Clean up unused imports in test file
