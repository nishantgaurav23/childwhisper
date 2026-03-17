# Spec S1.4 — Zero-Shot Inference Pipeline

## Feature
Zero-shot inference pipeline using Whisper-small for children's speech transcription. Reads utterance metadata, loads audio, runs batch inference, applies text normalization, and writes JSONL submission output. This is the Phase 1 baseline — no fine-tuning, just `openai/whisper-small` out of the box.

## Location
`submission/main.py`

## Depends On
- S1.2 (Audio preprocessing pipeline) — done
- S1.3 (Text normalization) — done

## Outcomes
1. Load utterance metadata from `utterance_metadata.jsonl`
2. Sort utterances by duration (longest first) for efficient batching
3. Load and preprocess audio (16 kHz mono) using `src/preprocess.py`
4. Run Whisper-small zero-shot inference with beam search (num_beams=5)
5. Detect silence and return empty string for silent clips
6. Apply `EnglishTextNormalizer` to all predictions
7. Write `submission.jsonl` with one `{"utterance_id", "orthographic_text"}` per line
8. Handle edge cases: missing audio files, empty predictions, corrupted files
9. Support both CUDA and CPU/MPS devices for local testing
10. Log progress and timing information
11. All utterance IDs from metadata appear in output (no missing IDs)

## Interface

```python
# submission/main.py

# --- Constants ---
DATA_DIR: Path       # /code_execution/data (competition) or configurable for local
OUTPUT_DIR: Path     # /code_execution/submission (competition) or configurable
SAMPLE_RATE: int     # 16000
BATCH_SIZE: int      # 16 (configurable)

def get_device() -> str:
    """Return best available device: cuda > mps > cpu."""

def load_metadata(data_dir: Path) -> list[dict]:
    """Load utterance_metadata.jsonl, return list of dicts."""

def load_model(device: str) -> tuple:
    """Load Whisper-small model + processor, return (model, processor)."""

def transcribe_batch(
    model, processor, audio_arrays: list[np.ndarray], device: str
) -> list[str]:
    """Transcribe a batch of audio arrays. Returns list of raw text predictions."""

def run_inference(
    model, processor, utterances: list[dict], data_dir: Path, device: str,
    batch_size: int = 16,
) -> dict[str, str]:
    """Run inference on all utterances. Returns {utterance_id: normalized_text}."""

def write_submission(
    predictions: dict[str, str], utterances: list[dict], output_dir: Path
) -> Path:
    """Write submission.jsonl. Returns path to output file."""

def main():
    """Entrypoint: load data, run inference, write output."""
```

## TDD Notes
- Mock `WhisperForConditionalGeneration` and `WhisperProcessor` — no real model loading
- Mock `librosa.load` — no real audio files
- Use `src/preprocess.py` functions for audio loading/silence detection
- Use `src/utils.py` `normalize_text` for text normalization
- Test metadata loading with sample JSONL
- Test batch transcription with mocked model outputs
- Test submission JSONL format compliance
- Test device detection logic
- Test edge cases: empty metadata, silent audio, missing files
- Test that all utterance IDs appear in output
- Test duration-based sorting
- All tests in `tests/test_inference.py`

## Code Standards
- Import `load_audio`, `is_silence` from `src/preprocess`
- Import `normalize_text` from `src/utils`
- Use `torch.no_grad()` for inference
- Use `torch.float16` for model on CUDA, `torch.float32` for CPU/MPS
- Set `language="en"`, `task="transcribe"` for generation
- Beam search with `num_beams=5`, `max_new_tokens=225`
- Handle `sys.path` so `src/` imports work from `submission/`
- Print timing info (model load time, inference time, total time)
