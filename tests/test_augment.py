"""Tests for the noise augmentation pipeline (S4.1)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf


def _create_dummy_wav(directory, filename="noise.wav", duration_sec=1.0, sr=16000):
    """Create a dummy WAV file with white noise in the given directory."""
    audio = np.random.randn(int(sr * duration_sec)).astype(np.float32) * 0.1
    path = directory / filename
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def noise_dirs(tmp_path):
    """Create tmp dirs with dummy audio files for AddBackgroundNoise."""
    noise_dir = tmp_path / "musan"
    noise_dir.mkdir()
    _create_dummy_wav(noise_dir, "babble_001.wav")

    realclass_dir = tmp_path / "realclass"
    realclass_dir.mkdir()
    _create_dummy_wav(realclass_dir, "classroom_001.wav")

    return noise_dir, realclass_dir


class TestCreateAugmentation:
    """Tests for create_augmentation factory function."""

    def test_returns_callable(self, noise_dirs):
        """create_augmentation returns a callable."""
        from src.augment import create_augmentation

        noise_dir, realclass_dir = noise_dirs
        aug_fn = create_augmentation(noise_dir, realclass_dir)
        assert callable(aug_fn)

    def test_accepts_str_paths(self, noise_dirs):
        """Factory accepts string paths, not just Path objects."""
        from src.augment import create_augmentation

        noise_dir, realclass_dir = noise_dirs
        aug_fn = create_augmentation(str(noise_dir), str(realclass_dir))
        assert callable(aug_fn)

    def test_raises_for_missing_noise_dir(self, tmp_path):
        """FileNotFoundError raised when noise_dir doesn't exist."""
        from src.augment import create_augmentation

        realclass_dir = tmp_path / "realclass"
        realclass_dir.mkdir()
        _create_dummy_wav(realclass_dir)

        with pytest.raises(FileNotFoundError):
            create_augmentation(tmp_path / "nonexistent", realclass_dir)

    def test_raises_for_missing_realclass_dir(self, tmp_path):
        """FileNotFoundError raised when realclass_dir doesn't exist."""
        from src.augment import create_augmentation

        noise_dir = tmp_path / "musan"
        noise_dir.mkdir()
        _create_dummy_wav(noise_dir)

        with pytest.raises(FileNotFoundError):
            create_augmentation(noise_dir, tmp_path / "nonexistent")

    @patch("src.augment.Compose")
    @patch("src.augment.AddBackgroundNoise")
    @patch("src.augment.OneOf")
    @patch("src.augment.Gain")
    def test_output_shape_matches_input(
        self, mock_gain, mock_oneof, mock_add_noise, mock_compose, tmp_path
    ):
        """Output array has same shape as input."""
        from src.augment import create_augmentation

        noise_dir = tmp_path / "musan"
        noise_dir.mkdir()
        realclass_dir = tmp_path / "realclass"
        realclass_dir.mkdir()

        mock_instance = MagicMock()
        mock_instance.side_effect = lambda samples, sample_rate: samples
        mock_compose.return_value = mock_instance

        aug_fn = create_augmentation(noise_dir, realclass_dir)

        audio = np.random.randn(16000).astype(np.float32)
        result = aug_fn(audio, sample_rate=16000)

        assert result.shape == audio.shape

    @patch("src.augment.Compose")
    @patch("src.augment.AddBackgroundNoise")
    @patch("src.augment.OneOf")
    @patch("src.augment.Gain")
    def test_output_dtype_is_float(
        self, mock_gain, mock_oneof, mock_add_noise, mock_compose, tmp_path
    ):
        """Output array dtype is float."""
        from src.augment import create_augmentation

        noise_dir = tmp_path / "musan"
        noise_dir.mkdir()
        realclass_dir = tmp_path / "realclass"
        realclass_dir.mkdir()

        mock_instance = MagicMock()
        mock_instance.side_effect = lambda samples, sample_rate: samples
        mock_compose.return_value = mock_instance

        aug_fn = create_augmentation(noise_dir, realclass_dir)

        audio = np.random.randn(16000).astype(np.float32)
        result = aug_fn(audio, sample_rate=16000)

        assert np.issubdtype(result.dtype, np.floating)

    @patch("src.augment.Compose")
    @patch("src.augment.AddBackgroundNoise")
    @patch("src.augment.OneOf")
    @patch("src.augment.Gain")
    def test_output_is_1d(
        self, mock_gain, mock_oneof, mock_add_noise, mock_compose, tmp_path
    ):
        """Output array is 1D (mono)."""
        from src.augment import create_augmentation

        noise_dir = tmp_path / "musan"
        noise_dir.mkdir()
        realclass_dir = tmp_path / "realclass"
        realclass_dir.mkdir()

        mock_instance = MagicMock()
        mock_instance.side_effect = lambda samples, sample_rate: samples
        mock_compose.return_value = mock_instance

        aug_fn = create_augmentation(noise_dir, realclass_dir)

        audio = np.random.randn(16000).astype(np.float32)
        result = aug_fn(audio, sample_rate=16000)

        assert result.ndim == 1

    def test_custom_snr_ranges(self, noise_dirs):
        """Factory accepts custom SNR ranges."""
        from src.augment import create_augmentation

        noise_dir, realclass_dir = noise_dirs
        aug_fn = create_augmentation(
            noise_dir,
            realclass_dir,
            realclass_min_snr=10,
            realclass_max_snr=25,
            musan_min_snr=5,
            musan_max_snr=20,
        )
        assert callable(aug_fn)

    def test_augmented_audio_output(self, noise_dirs):
        """Augmentation produces valid output with real audiomentations."""
        from src.augment import create_augmentation

        noise_dir, realclass_dir = noise_dirs
        aug_fn = create_augmentation(noise_dir, realclass_dir)

        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = aug_fn(audio, sample_rate=16000)

        assert result.shape == audio.shape
        assert result.ndim == 1
        assert np.issubdtype(result.dtype, np.floating)


class TestCreateNoiseOnlyAugmentation:
    """Tests for create_noise_only_augmentation."""

    def test_returns_callable(self, tmp_path):
        """create_noise_only_augmentation returns a callable."""
        from src.augment import create_noise_only_augmentation

        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        _create_dummy_wav(noise_dir)

        aug_fn = create_noise_only_augmentation(noise_dir)
        assert callable(aug_fn)

    def test_raises_for_missing_dir(self, tmp_path):
        """FileNotFoundError raised when noise_dir doesn't exist."""
        from src.augment import create_noise_only_augmentation

        with pytest.raises(FileNotFoundError):
            create_noise_only_augmentation(tmp_path / "nonexistent")

    def test_accepts_str_path(self, tmp_path):
        """Factory accepts string path."""
        from src.augment import create_noise_only_augmentation

        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        _create_dummy_wav(noise_dir)

        aug_fn = create_noise_only_augmentation(str(noise_dir))
        assert callable(aug_fn)

    def test_custom_snr_and_probability(self, tmp_path):
        """Factory accepts custom SNR range and probability."""
        from src.augment import create_noise_only_augmentation

        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        _create_dummy_wav(noise_dir)

        aug_fn = create_noise_only_augmentation(
            noise_dir, min_snr=5, max_snr=15, p=0.8
        )
        assert callable(aug_fn)

    @patch("src.augment.Compose")
    @patch("src.augment.AddBackgroundNoise")
    def test_output_matches_input_shape(
        self, mock_add_noise, mock_compose, tmp_path
    ):
        """Output shape matches input."""
        from src.augment import create_noise_only_augmentation

        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()

        mock_instance = MagicMock()
        mock_instance.side_effect = lambda samples, sample_rate: samples
        mock_compose.return_value = mock_instance

        aug_fn = create_noise_only_augmentation(noise_dir)

        audio = np.random.randn(16000).astype(np.float32)
        result = aug_fn(audio, sample_rate=16000)

        assert result.shape == audio.shape
        assert result.ndim == 1


class TestAugmentFnSignature:
    """Test that augmentation functions match WhisperDataset.augment_fn signature."""

    @patch("src.augment.Compose")
    @patch("src.augment.AddBackgroundNoise")
    @patch("src.augment.OneOf")
    @patch("src.augment.Gain")
    def test_callable_with_audio_and_sample_rate(
        self, mock_gain, mock_oneof, mock_add_noise, mock_compose, tmp_path
    ):
        """augment_fn(audio, sample_rate=sr) works — matches WhisperDataset call."""
        from src.augment import create_augmentation

        noise_dir = tmp_path / "musan"
        noise_dir.mkdir()
        realclass_dir = tmp_path / "realclass"
        realclass_dir.mkdir()

        mock_instance = MagicMock()
        mock_instance.side_effect = lambda samples, sample_rate: samples
        mock_compose.return_value = mock_instance

        aug_fn = create_augmentation(noise_dir, realclass_dir)

        audio = np.random.randn(8000).astype(np.float32)
        # This is exactly how WhisperDataset calls it
        result = aug_fn(audio, sample_rate=16000)

        assert isinstance(result, np.ndarray)
