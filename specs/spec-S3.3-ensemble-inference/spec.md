# Spec S3.3 — Ensemble Inference Pipeline

## Overview
Upgrade `submission/main.py` to run both Whisper-large-v3 + LoRA and Whisper-small (fine-tuned) sequentially on A100, with confidence-based prediction merging and time budget management.

## Depends On
- **S3.1**: LoRA configuration & training (provides LoRA adapter loading via PEFT)
- **S2.4**: Inference with fine-tuned small (provides current single-model inference pipeline)

## Location
- `submission/main.py` (primary — extend existing file)

## Outcomes
1. Load Whisper-large-v3 + LoRA adapter in fp16 on A100
2. Load Whisper-small fine-tuned weights in fp16 on A100
3. Run Model A (large) first on all utterances
4. If time budget allows (< 90 min elapsed), free Model A VRAM, load Model B (small), run on all utterances
5. Merge predictions: prefer Model A; if Model A produced empty/whitespace, use Model B's prediction
6. Apply `EnglishTextNormalizer` to all final predictions
7. Time budget management: 2-hour wall clock, configurable safety margin
8. Graceful degradation: if only Model A completes, submit those predictions
9. Works on MacBook (MPS/CPU) for local testing — skip LoRA loading if adapter not found

## Design Details

### Model Loading
- `load_large_model(device, base_model_path, adapter_path)` → loads Whisper-large-v3 base + merges LoRA adapter via `PeftModel.from_pretrained()`
- `load_small_model(device, model_path)` → existing `load_model()` renamed/wrapped
- Both return `(model, processor)` tuples

### Inference Flow
```
1. Start timer
2. Load metadata, sort by duration (longest first)
3. Load Model A (large-v3 + LoRA)
4. Run inference on all utterances → predictions_a
5. Check elapsed time vs TIME_LIMIT - SAFETY_MARGIN
6. If time allows:
   a. Delete Model A, free VRAM
   b. Load Model B (small fine-tuned)
   c. Run inference on all utterances → predictions_b
7. Merge: for each utterance, use predictions_a unless empty → then use predictions_b
8. Write submission.jsonl
```

### Time Budget Constants
- `TIME_LIMIT_SEC = 7200` (2 hours)
- `SAFETY_MARGIN_SEC = 300` (5 minutes for writing output + buffer)
- `MODEL_B_CUTOFF_SEC = 5400` (90 minutes — don't start Model B if past this)

### Merge Strategy
- If predictions_a[uid] is non-empty → use it
- If predictions_a[uid] is empty AND predictions_b exists → use predictions_b[uid]
- If both empty → output empty string

## TDD Notes

### Test Categories
1. **load_large_model**: Mock PeftModel, verify adapter loading, device placement
2. **merge_predictions**: Test all merge cases (both non-empty, A empty B non-empty, both empty, B missing)
3. **check_time_budget**: Test time budget decisions
4. **run_ensemble_inference (integration)**: Mock both models, verify full flow
5. **Graceful degradation**: Model B skipped when time exceeded
6. **Backward compatibility**: When no LoRA adapter exists, fall back to small-only mode

### What to Mock
- `WhisperForConditionalGeneration.from_pretrained`
- `PeftModel.from_pretrained`
- `WhisperProcessor.from_pretrained`
- `load_audio` (use synthetic arrays)
- `time.time()` (for time budget tests)
