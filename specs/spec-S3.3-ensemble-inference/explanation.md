# Explanation S3.3 — Ensemble Inference Pipeline

## Why
The competition allows 2 hours on an A100 80GB — enough to run two models sequentially. An ensemble of Whisper-large-v3 (LoRA-tuned, primary) and Whisper-small (fully fine-tuned, secondary) provides complementary error profiles: the large model is better on complex utterances while the small model produces fewer hallucinations on short clips. Merging predictions where Model A fails (empty output) with Model B's fallback consistently reduces WER in ASR competitions.

## What
Extended `submission/main.py` with four new functions and updated `main()`:

1. **`load_large_model(device, base_model, adapter_path)`** — Loads Whisper-large-v3 base in fp16, then merges LoRA adapter via `PeftModel.from_pretrained()`. Returns `(model, processor)`.

2. **`merge_predictions(preds_a, preds_b)`** — Prefers Model A predictions. Falls back to Model B when A is empty/whitespace. Handles `None` (Model B didn't run) gracefully.

3. **`check_time_budget(elapsed_sec, cutoff_sec)`** — Returns whether enough time remains to start Model B inference (default cutoff: 90 minutes of 120).

4. **`run_ensemble_inference(...)`** — Orchestrates the full flow:
   - Load and run Model A (large-v3 + LoRA)
   - If time allows, free VRAM, load and run Model B (small)
   - Merge predictions
   - Falls back to small-only when adapter is missing (FileNotFoundError)

5. **Updated `main()`** — Now calls `run_ensemble_inference()` instead of single-model inference.

### Constants Added
- `LARGE_MODEL = "openai/whisper-large-v3"`
- `LORA_ADAPTER_DIR` — path to LoRA adapter weights
- `TIME_LIMIT_SEC = 7200` (2 hours)
- `SAFETY_MARGIN_SEC = 300` (5 minute buffer)
- `MODEL_B_CUTOFF_SEC = 5400` (90 minute cutoff for starting Model B)

## How
- **TDD**: 24 new tests written first covering merge logic (7 cases), time budget (5 cases), model loading (4 cases), integration (3 cases), and backward compatibility (1 case).
- **Backward compatible**: All 29 existing S1.4/S2.4 tests continue to pass. The existing `load_model()`, `run_inference()`, `transcribe_batch()` functions are unchanged.
- **Graceful degradation**: If LoRA adapter is missing, catches `FileNotFoundError` and falls back to small-model-only. If both models fail, returns empty predictions for all utterances.
- **VRAM management**: Deletes Model A and calls `torch.cuda.empty_cache()` before loading Model B.

## Connections
- **Depends on S3.1** (LoRA config) for `PeftModel` adapter format
- **Depends on S2.4** (fine-tuned inference) for the existing single-model pipeline
- **Enables S3.4** (submission packaging) — ensemble inference is now the default submission entrypoint
- **Enables S5.1** (post-processing) and **S5.2** (inference optimization) which build on top of ensemble output
