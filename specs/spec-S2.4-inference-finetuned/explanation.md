# Explanation — S2.4 Inference with Fine-Tuned Whisper-small

## Why
The S1.4 inference pipeline was hardcoded to use zero-shot `openai/whisper-small`. After S2.2 produced fine-tuned weights, the submission pipeline needed to load those weights instead. Without this spec, fine-tuned models trained on Kaggle couldn't be used for competition submissions.

## What
Updated `submission/main.py` to support configurable model loading:

1. **`resolve_model_path(finetuned_path, default)`** — checks if a local directory contains valid model weights (has `config.json`). Returns the path if valid, otherwise returns the default HF Hub model ID. This separation of resolution from loading makes fallback logic testable.

2. **`load_model(device, model_name_or_path)`** — extended with optional `model_name_or_path` parameter. Loads both model and processor from the specified source. Processor loading has its own fallback: if the processor can't be loaded from the model path (e.g., only model weights were saved without processor), it falls back to `openai/whisper-small` processor.

3. **`main()`** — now calls `resolve_model_path(FINETUNED_WEIGHTS_DIR)` before loading, automatically preferring fine-tuned weights at `submission/model_weights/whisper_small_ft/` when present.

## How
- **Backward compatible**: All existing S1.4 function signatures still work with default arguments. `load_model("cpu")` still loads zero-shot whisper-small.
- **Graceful degradation**: Missing weights dir → zero-shot. Empty weights dir → zero-shot. Missing processor in weights dir → fallback processor.
- **TDD**: 10 new tests covering `resolve_model_path` (5 tests) and `load_model` fine-tuned extensions (5 tests). All 29 inference tests pass, 197 total tests pass.

## Connections
- **Upstream**: S2.2 (`train_whisper_small.py`) saves model + processor via `Seq2SeqTrainer.save_model()` and `processor.save_pretrained()` — this spec loads those outputs.
- **Downstream**: S3.3 (ensemble inference) will extend this further to load both Whisper-large-v3+LoRA and fine-tuned small together. The `model_name_or_path` pattern established here will be reused for the large model.
- **Submission workflow**: `scripts/download_weights.sh` (S3.4) will pull HF Hub weights into `submission/model_weights/whisper_small_ft/` before packaging the submission zip.
