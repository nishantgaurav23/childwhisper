"""Tests for src/preprocess.py — Audio preprocessing pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.preprocess import (
    get_duration,
    is_silence,
    is_valid_duration,
    load_audio,
    load_metadata,
    preprocess_utterance,
    trim_silence,
)


# --- Fixtures ---


@pytest.fixture
def silent_audio():
    """Audio array with very low energy (silence)."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def loud_audio():
    """Audio array with clearly audible signal."""
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine, amplitude 0.5


@pytest.fixture
def short_audio():
    """Audio shorter than 0.3s at 16kHz."""
    return np.random.randn(3200).astype(np.float32) * 0.1  # 0.2s


@pytest.fixture
def long_audio():
    """Audio longer than 30s at 16kHz."""
    return np.random.randn(16000 * 31).astype(np.float32) * 0.1  # 31s


@pytest.fixture
def valid_audio():
    """Audio with valid duration (~1s) and audible signal."""
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t) * 0.3


@pytest.fixture
def sample_metadata():
    """Sample JSONL metadata lines."""
    return [
        {
            "utterance_id": "utt_001",
            "child_id": "child_01",
            "session_id": "sess_01",
            "audio_path": "audio/utt_001.flac",
            "audio_duration_sec": 2.5,
            "age_bucket": "5-7",
            "orthographic_text": "hello world",
        },
        {
            "utterance_id": "utt_002",
            "child_id": "child_02",
            "session_id": "sess_02",
            "audio_path": "audio/utt_002.flac",
            "audio_duration_sec": 1.1,
            "age_bucket": "3-4",
            "orthographic_text": "i goed to the store",
        },
    ]


# --- Tests for load_audio ---


class TestLoadAudio:
    def test_resamples_to_target_sr(self, valid_audio):
        """load_audio should resample audio to target sample rate."""
        with patch("src.preprocess.librosa") as mock_librosa:
            mock_librosa.load.return_value = (valid_audio, 16000)
            audio, sr = load_audio("fake_path.flac", target_sr=16000)
            mock_librosa.load.assert_called_once_with(
                "fake_path.flac", sr=16000, mono=True
            )
            assert sr == 16000
            np.testing.assert_array_equal(audio, valid_audio)

    def test_accepts_path_object(self, valid_audio):
        """load_audio should accept Path objects."""
        with patch("src.preprocess.librosa") as mock_librosa:
            mock_librosa.load.return_value = (valid_audio, 16000)
            audio, sr = load_audio(Path("fake_path.flac"))
            assert sr == 16000


# --- Tests for trim_silence ---


class TestTrimSilence:
    def test_trims_silence(self):
        """trim_silence should remove leading/trailing silence."""
        # Create audio: 0.5s silence + 0.5s tone + 0.5s silence
        silence = np.zeros(8000, dtype=np.float32)
        tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 8000)).astype(np.float32) * 0.5
        audio = np.concatenate([silence, tone, silence])

        with patch("src.preprocess.librosa") as mock_librosa:
            mock_librosa.effects.trim.return_value = (tone, (8000, 16000))
            trimmed = trim_silence(audio, top_db=30)
            mock_librosa.effects.trim.assert_called_once()
            assert len(trimmed) == len(tone)


# --- Tests for is_silence ---


class TestIsSilence:
    def test_detects_silence(self, silent_audio):
        """is_silence should return True for silent audio."""
        assert is_silence(silent_audio, threshold_db=-40.0) is True

    def test_non_silent_audio(self, loud_audio):
        """is_silence should return False for audible audio."""
        assert is_silence(loud_audio, threshold_db=-40.0) is False

    def test_near_threshold(self):
        """is_silence with audio just below threshold should return True."""
        # RMS at -41 dB: 10^(-41/20) ≈ 0.0089
        amplitude = 10 ** (-41 / 20)
        audio = np.full(16000, amplitude, dtype=np.float32)
        assert is_silence(audio, threshold_db=-40.0) is True

    def test_above_threshold(self):
        """is_silence with audio just above threshold should return False."""
        # RMS at -39 dB: 10^(-39/20) ≈ 0.0112
        amplitude = 10 ** (-39 / 20)
        audio = np.full(16000, amplitude, dtype=np.float32)
        assert is_silence(audio, threshold_db=-40.0) is False


# --- Tests for get_duration ---


class TestGetDuration:
    def test_one_second(self):
        """16000 samples at 16kHz = 1.0 second."""
        audio = np.zeros(16000, dtype=np.float32)
        assert get_duration(audio, sr=16000) == pytest.approx(1.0)

    def test_half_second(self):
        """8000 samples at 16kHz = 0.5 seconds."""
        audio = np.zeros(8000, dtype=np.float32)
        assert get_duration(audio, sr=16000) == pytest.approx(0.5)

    def test_empty_audio(self):
        """Empty audio should have 0 duration."""
        audio = np.array([], dtype=np.float32)
        assert get_duration(audio, sr=16000) == pytest.approx(0.0)


# --- Tests for is_valid_duration ---


class TestIsValidDuration:
    def test_valid_duration(self):
        assert is_valid_duration(1.0) is True
        assert is_valid_duration(15.0) is True
        assert is_valid_duration(0.3) is True
        assert is_valid_duration(30.0) is True

    def test_too_short(self):
        assert is_valid_duration(0.1) is False
        assert is_valid_duration(0.29) is False

    def test_too_long(self):
        assert is_valid_duration(30.1) is False
        assert is_valid_duration(60.0) is False

    def test_custom_bounds(self):
        assert is_valid_duration(0.5, min_dur=1.0) is False
        assert is_valid_duration(5.0, max_dur=3.0) is False


# --- Tests for preprocess_utterance ---


class TestPreprocessUtterance:
    def test_valid_utterance(self, valid_audio):
        """Full pipeline should return dict for valid utterance."""
        with patch("src.preprocess.librosa") as mock_librosa:
            mock_librosa.load.return_value = (valid_audio, 16000)
            mock_librosa.effects.trim.return_value = (valid_audio, (0, len(valid_audio)))
            result = preprocess_utterance("fake.flac", "hello world")
            assert result is not None
            assert "audio" in result
            assert result["transcript"] == "hello world"
            assert result["sr"] == 16000
            assert result["duration"] == pytest.approx(1.0)

    def test_too_short_returns_none(self, short_audio):
        """Should return None for audio shorter than min_duration."""
        with patch("src.preprocess.librosa") as mock_librosa:
            mock_librosa.load.return_value = (short_audio, 16000)
            mock_librosa.effects.trim.return_value = (short_audio, (0, len(short_audio)))
            result = preprocess_utterance("fake.flac", "hello")
            assert result is None

    def test_too_long_returns_none(self, long_audio):
        """Should return None for audio longer than max_duration."""
        with patch("src.preprocess.librosa") as mock_librosa:
            mock_librosa.load.return_value = (long_audio, 16000)
            mock_librosa.effects.trim.return_value = (long_audio, (0, len(long_audio)))
            result = preprocess_utterance("fake.flac", "hello")
            assert result is None

    def test_empty_transcript_returns_none(self, valid_audio):
        """Should return None for empty transcript."""
        result = preprocess_utterance("fake.flac", "")
        assert result is None

    def test_whitespace_transcript_returns_none(self, valid_audio):
        """Should return None for whitespace-only transcript."""
        result = preprocess_utterance("fake.flac", "   ")
        assert result is None

    def test_silence_returns_empty_transcript(self, silent_audio):
        """Should return dict with empty transcript for silent audio."""
        with patch("src.preprocess.librosa") as mock_librosa:
            mock_librosa.load.return_value = (silent_audio, 16000)
            mock_librosa.effects.trim.return_value = (silent_audio, (0, len(silent_audio)))
            result = preprocess_utterance("fake.flac", "hello")
            assert result is not None
            assert result["transcript"] == ""


# --- Tests for load_metadata ---


class TestLoadMetadata:
    def test_loads_jsonl(self, sample_metadata):
        """Should parse JSONL file into list of dicts."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for entry in sample_metadata:
                f.write(json.dumps(entry) + "\n")
            f.flush()
            result = load_metadata(f.name)
            assert len(result) == 2
            assert result[0]["utterance_id"] == "utt_001"
            assert result[1]["orthographic_text"] == "i goed to the store"

    def test_loads_empty_file(self):
        """Should return empty list for empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            f.flush()
            result = load_metadata(f.name)
            assert result == []

    def test_accepts_path_object(self, sample_metadata):
        """Should accept Path objects."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_metadata[0]) + "\n")
            f.flush()
            result = load_metadata(Path(f.name))
            assert len(result) == 1
