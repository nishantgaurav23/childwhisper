# Spec S1.2 — Audio Preprocessing Pipeline

## Feature
Audio preprocessing pipeline for loading, resampling, trimming, filtering, and silence detection of children's speech audio files.

## Location
`src/preprocess.py`

## Depends On
- S1.1 (Project structure & dependencies) — done

## Outcomes
1. Load FLAC/WAV audio files and resample to 16 kHz mono
2. Trim leading/trailing silence with `librosa.effects.trim(top_db=30)`
3. Filter audio by duration: skip clips < 0.3s or > 30.0s
4. Detect silence: return empty string for clips with RMS energy < -40 dB
5. Provide a batch preprocessing function for processing multiple utterances
6. Load training metadata from JSONL files
7. All config values sourced from `configs/training_config.yaml`

## Interface

```python
def load_audio(audio_path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate (mono)."""

def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Trim leading/trailing silence from audio."""

def is_silence(audio: np.ndarray, threshold_db: float = -40.0) -> bool:
    """Return True if audio RMS energy is below threshold."""

def get_duration(audio: np.ndarray, sr: int = 16000) -> float:
    """Return duration in seconds."""

def is_valid_duration(duration: float, min_dur: float = 0.3, max_dur: float = 30.0) -> bool:
    """Return True if duration is within valid range."""

def preprocess_utterance(
    audio_path: str | Path,
    transcript: str,
    target_sr: int = 16000,
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    trim_top_db: int = 30,
    silence_threshold_db: float = -40.0,
) -> dict | None:
    """
    Full preprocessing for one utterance.
    Returns dict with keys: audio, transcript, duration, sr
    Returns None if utterance should be skipped (invalid duration, empty transcript).
    Returns dict with empty transcript if silence detected.
    """

def load_metadata(jsonl_path: str | Path) -> list[dict]:
    """Load utterance metadata from JSONL file."""
```

## TDD Notes
- Mock `librosa.load` and `soundfile` in unit tests — no real audio files needed
- Use `numpy` arrays with known properties for testing silence detection and duration
- Test edge cases: empty audio, very short clips, very long clips, silent clips
- Test metadata loading with sample JSONL strings
- All tests in `tests/test_preprocess.py`

## Code Standards
- Use `librosa` for loading and resampling (not torchaudio)
- Use `numpy` for RMS computation
- Type hints on all public functions
- Config defaults match `configs/training_config.yaml`
