# Spec S2.4 — Inference with Fine-Tuned Whisper-small

## Overview
Update `submission/main.py` to load fine-tuned Whisper-small weights (from HuggingFace Hub or local directory) instead of zero-shot `openai/whisper-small`. The pipeline must remain backward-compatible with zero-shot mode and continue to work on MacBook (MPS/CPU) and A100 (CUDA).

## Depends On
- **S2.2** (done): Whisper-small training script — produces fine-tuned weights saved to HF Hub or local checkpoint dir
- **S1.4** (done): Zero-shot inference pipeline — the base `submission/main.py` to extend

## Location
- `submission/main.py` (modify existing)

## Requirements

### R1: Configurable Model Source
- `load_model()` accepts an optional `model_name_or_path` parameter (default: `"openai/whisper-small"`)
- When given a local directory path, loads model + processor from that directory
- When given an HF Hub model ID, downloads and loads from Hub
- Zero-shot mode still works when no fine-tuned weights are available

### R2: Fine-Tuned Weight Paths
- Default fine-tuned weights location: `./model_weights/whisper_small_ft/` (relative to `submission/`)
- Fallback: if fine-tuned weights dir doesn't exist, fall back to zero-shot `openai/whisper-small`
- `main()` attempts fine-tuned path first, then falls back gracefully

### R3: Processor Consistency
- When loading fine-tuned model, load processor from same path (processor is saved alongside model by `Seq2SeqTrainer.save_model()`)
- If processor not found in model dir, fall back to `openai/whisper-small` processor

### R4: Device & Dtype Handling
- CUDA: load in float16
- MPS/CPU: load in float32
- Same logic as existing S1.4 implementation

### R5: Logging & Status
- Log which model source is being used (fine-tuned vs zero-shot)
- Log fallback when fine-tuned weights not found

## Outcomes
1. `load_model()` supports both zero-shot and fine-tuned weights
2. `main()` auto-detects fine-tuned weights at default path, falls back to zero-shot
3. All existing S1.4 tests continue to pass
4. New tests cover fine-tuned loading, fallback, and processor loading
5. Works on MacBook (MPS/CPU) — no CUDA-only code paths

## TDD Notes
- Mock `WhisperForConditionalGeneration.from_pretrained` and `WhisperProcessor.from_pretrained`
- Test fine-tuned path resolution and fallback logic
- Test processor loading from model dir vs fallback
- Test `main()` with fine-tuned weights dir present/absent
- All tests must pass without real model weights or GPU
