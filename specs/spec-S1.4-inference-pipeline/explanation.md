# Explanation S1.4 — Zero-Shot Inference Pipeline

## Why
The zero-shot inference pipeline establishes the Phase 1 baseline — a fully functional competition submission using Whisper-small with no fine-tuning. This gives us an immediate WER measurement to beat, validates the end-to-end data flow (metadata → audio → model → normalized text → JSONL), and provides the scaffolding that all future specs (S2.4, S3.3) will extend with fine-tuned models and ensemble logic.

## What
A `submission/main.py` module that:
- Reads `utterance_metadata.jsonl` and sorts utterances by duration (longest first) for efficient GPU batching
- Loads Whisper-small (`openai/whisper-small`) with automatic device selection (CUDA → MPS → CPU)
- Runs batch inference with beam search (num_beams=5, max_new_tokens=225)
- Detects silent audio via RMS energy threshold and returns empty string (prevents hallucination)
- Normalizes all predictions with Whisper's `EnglishTextNormalizer`
- Writes `submission.jsonl` with `{"utterance_id", "orthographic_text"}` per line
- Handles edge cases: missing audio files, empty batches, silent clips

## How

### Architecture
```
utterance_metadata.jsonl
    │
    ▼ load_metadata()
[sorted by duration]
    │
    ▼ load_audio() + is_silence()  ← reuses src/preprocess.py
    │
    ▼ transcribe_batch()           ← Whisper-small, beam=5
    │
    ▼ normalize_text()             ← reuses src/utils.py
    │
    ▼ write_submission()
submission.jsonl
```

### Key Design Decisions
1. **Reuse `src/preprocess.py` and `src/utils.py`** — no duplication of audio loading or text normalization logic. The submission module imports from the shared source.
2. **Duration-sorted batching** — longest utterances first minimizes padding waste and ensures GPU utilization is highest when time budget is freshest.
3. **Silence detection before transcription** — skips the model entirely for silent clips, saving compute and preventing Whisper's tendency to hallucinate on silence.
4. **Device-agnostic** — same code works on A100 (competition), MacBook MPS (local testing), and CPU (CI).
5. **Missing prediction defaults to empty string** — ensures all utterance IDs appear in output even if errors occur during processing.

### Testing
19 tests in `tests/test_inference.py` covering:
- Device detection, metadata loading (valid/empty/missing)
- Model loading (mocked, verifies whisper-small)
- Batch transcription (mocked model, empty batch edge case)
- Full inference pipeline (sorting, silence detection, normalization)
- Submission writing (JSONL format, all IDs present, missing predictions, directory creation)

## Connections
- **Depends on S1.2** — `load_audio()` and `is_silence()` from `src/preprocess.py`
- **Depends on S1.3** — `normalize_text()` from `src/utils.py`
- **Extended by S2.4** — will update `load_model()` to load fine-tuned weights
- **Extended by S3.3** — will add ensemble logic with Whisper-large-v3 + LoRA
- **Extended by S5.2** — may swap to faster-whisper/CTranslate2 for speed
