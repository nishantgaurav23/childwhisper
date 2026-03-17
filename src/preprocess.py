"""Audio preprocessing pipeline for children's speech.

Handles loading, resampling, trimming, duration filtering, and silence detection.
All audio is processed to 16 kHz mono.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np


def load_audio(
    audio_path: str | Path, target_sr: int = 16000
) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate (mono)."""
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio, target_sr


def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Trim leading/trailing silence from audio."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def is_silence(audio: np.ndarray, threshold_db: float = -40.0) -> bool:
    """Return True if audio RMS energy is below threshold."""
    rms = np.sqrt(np.mean(audio**2))
    db = 20 * np.log10(rms + 1e-10)
    return bool(db < threshold_db)


def get_duration(audio: np.ndarray, sr: int = 16000) -> float:
    """Return duration in seconds."""
    return len(audio) / sr


def is_valid_duration(
    duration: float, min_dur: float = 0.3, max_dur: float = 30.0
) -> bool:
    """Return True if duration is within valid range (inclusive)."""
    return min_dur <= duration <= max_dur


def preprocess_utterance(
    audio_path: str | Path,
    transcript: str,
    target_sr: int = 16000,
    min_duration: float = 0.3,
    max_duration: float = 30.0,
    trim_top_db: int = 30,
    silence_threshold_db: float = -40.0,
) -> dict | None:
    """Preprocess a single utterance.

    Returns dict with keys: audio, transcript, duration, sr.
    Returns None if utterance should be skipped (invalid duration, empty transcript).
    Returns dict with empty transcript if silence detected.
    """
    if not transcript or not transcript.strip():
        return None

    audio, sr = load_audio(audio_path, target_sr=target_sr)
    audio = trim_silence(audio, top_db=trim_top_db)

    if is_silence(audio, threshold_db=silence_threshold_db):
        return {"audio": audio, "transcript": "", "duration": get_duration(audio, sr), "sr": sr}

    duration = get_duration(audio, sr)
    if not is_valid_duration(duration, min_dur=min_duration, max_dur=max_duration):
        return None

    return {"audio": audio, "transcript": transcript, "duration": duration, "sr": sr}


def load_metadata(jsonl_path: str | Path) -> list[dict]:
    """Load utterance metadata from JSONL file."""
    path = Path(jsonl_path)
    text = path.read_text().strip()
    if not text:
        return []
    return [json.loads(line) for line in text.split("\n")]
