"""Tests for noisy validation functions in src/evaluate.py (S4.2)."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.evaluate import (
    apply_noise_to_val,
    combined_validation_summary,
    format_validation_report,
)


def _create_dummy_wav(directory, filename="noise.wav", duration_sec=1.0, sr=16000):
    """Create a dummy WAV file with white noise."""
    audio = np.random.randn(int(sr * duration_sec)).astype(np.float32) * 0.1
    path = directory / filename
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def noise_dir(tmp_path):
    """Create a tmp dir with a dummy noise file."""
    d = tmp_path / "realclass"
    d.mkdir()
    _create_dummy_wav(d, "classroom_001.wav")
    return d


@pytest.fixture
def sample_audio_list():
    """List of 3 clean audio arrays at 16kHz (1 second each)."""
    rng = np.random.RandomState(42)
    return [rng.randn(16000).astype(np.float32) * 0.5 for _ in range(3)]


# ---------------------------------------------------------------------------
# apply_noise_to_val tests
# ---------------------------------------------------------------------------

class TestApplyNoiseToVal:
    def test_returns_list_same_length(self, noise_dir, sample_audio_list):
        """Returns list of same length as input."""
        result = apply_noise_to_val(sample_audio_list, noise_dir)
        assert isinstance(result, list)
        assert len(result) == len(sample_audio_list)

    def test_output_shape_matches_input(self, noise_dir, sample_audio_list):
        """Each output array has same shape as corresponding input."""
        result = apply_noise_to_val(sample_audio_list, noise_dir)
        for orig, noisy in zip(sample_audio_list, result):
            assert noisy.shape == orig.shape

    def test_output_dtype_is_float(self, noise_dir, sample_audio_list):
        """Output arrays are float dtype."""
        result = apply_noise_to_val(sample_audio_list, noise_dir)
        for arr in result:
            assert np.issubdtype(arr.dtype, np.floating)

    def test_modifies_audio(self, noise_dir, sample_audio_list):
        """Noisy audio differs from clean (noise was applied)."""
        originals = [a.copy() for a in sample_audio_list]
        result = apply_noise_to_val(sample_audio_list, noise_dir)
        any_different = any(
            not np.allclose(orig, noisy) for orig, noisy in zip(originals, result)
        )
        assert any_different

    def test_deterministic_with_seed(self, noise_dir, sample_audio_list):
        """Same seed produces same output."""
        r1 = apply_noise_to_val(sample_audio_list, noise_dir, seed=42)
        r2 = apply_noise_to_val(sample_audio_list, noise_dir, seed=42)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)

    def test_raises_for_missing_dir(self, tmp_path, sample_audio_list):
        """FileNotFoundError when noise_dir doesn't exist."""
        with pytest.raises(FileNotFoundError):
            apply_noise_to_val(sample_audio_list, tmp_path / "nonexistent")

    def test_empty_list(self, noise_dir):
        """Empty input returns empty output."""
        result = apply_noise_to_val([], noise_dir)
        assert result == []


# ---------------------------------------------------------------------------
# combined_validation_summary tests
# ---------------------------------------------------------------------------

class TestCombinedValidationSummary:
    def test_structure(self):
        """Result dict has clean, noisy, wer_degradation, relative_degradation."""
        refs = ["hello world", "the cat sat"]
        clean_hyps = ["hello world", "the cat sat"]
        noisy_hyps = ["hello world", "the dog sat"]
        ages = ["3-4", "5-7"]
        result = combined_validation_summary(refs, clean_hyps, noisy_hyps, ages)
        assert "clean" in result
        assert "noisy" in result
        assert "wer_degradation" in result
        assert "relative_degradation" in result

    def test_clean_matches_validation_summary(self):
        """Clean sub-dict matches standalone validation_summary."""
        from src.evaluate import validation_summary

        refs = ["hello world", "the cat"]
        clean_hyps = ["hello world", "the cat"]
        noisy_hyps = ["hello wrong", "the dog"]
        ages = ["3-4", "5-7"]
        result = combined_validation_summary(refs, clean_hyps, noisy_hyps, ages)
        expected = validation_summary(refs, clean_hyps, ages)
        assert result["clean"]["overall_wer"] == expected["overall_wer"]

    def test_degradation_computed_correctly(self):
        """wer_degradation = noisy_wer - clean_wer."""
        refs = ["the cat sat down"]
        clean_hyps = ["the cat sat down"]  # WER = 0.0
        noisy_hyps = ["the dog sat down"]  # WER = 0.25
        ages = ["5-7"]
        result = combined_validation_summary(refs, clean_hyps, noisy_hyps, ages)
        assert abs(result["wer_degradation"] - 0.25) < 1e-6
        assert result["relative_degradation"] == 0.0  # clean WER is 0, so 0.0

    def test_relative_degradation_nonzero_clean(self):
        """relative_degradation = (noisy - clean) / clean when clean > 0."""
        refs = ["the cat sat down"]
        clean_hyps = ["the dog sat down"]  # WER = 0.25
        noisy_hyps = ["the dog ran down"]  # WER = 0.50
        ages = ["5-7"]
        result = combined_validation_summary(refs, clean_hyps, noisy_hyps, ages)
        assert abs(result["wer_degradation"] - 0.25) < 1e-6
        assert abs(result["relative_degradation"] - 1.0) < 1e-6  # 0.25/0.25 = 1.0


# ---------------------------------------------------------------------------
# format_validation_report tests
# ---------------------------------------------------------------------------

class TestFormatValidationReport:
    def _make_combined_summary(self):
        refs = ["hello world", "the cat sat"]
        clean_hyps = ["hello world", "the cat sat"]
        noisy_hyps = ["hello world", "the dog sat"]
        ages = ["3-4", "5-7"]
        return combined_validation_summary(refs, clean_hyps, noisy_hyps, ages)

    def test_contains_clean_and_noisy_headers(self):
        """Report contains 'Clean' and 'Noisy' labels."""
        summary = self._make_combined_summary()
        report = format_validation_report(summary)
        assert "Clean" in report
        assert "Noisy" in report

    def test_contains_age_buckets(self):
        """Report contains age bucket labels from the summary."""
        summary = self._make_combined_summary()
        report = format_validation_report(summary)
        assert "3-4" in report
        assert "5-7" in report

    def test_contains_degradation(self):
        """Report contains degradation info."""
        summary = self._make_combined_summary()
        report = format_validation_report(summary)
        assert "degradation" in report.lower() or "Degradation" in report
