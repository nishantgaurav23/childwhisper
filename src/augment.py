"""Noise augmentation pipeline for children's speech training.

Provides factory functions that return callables compatible with
WhisperDataset's augment_fn parameter. Uses audiomentations for
mixing classroom noise (RealClass) and babble noise (MUSAN).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from audiomentations import AddBackgroundNoise, Compose, Gain, OneOf


def _validate_dir(path: str | Path, name: str) -> Path:
    """Validate that a directory exists and return as Path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{name} directory not found: {p}")
    return p


def create_augmentation(
    noise_dir: str | Path,
    realclass_dir: str | Path,
    sample_rate: int = 16000,
    realclass_min_snr: float = 5.0,
    realclass_max_snr: float = 20.0,
    musan_min_snr: float = 0.0,
    musan_max_snr: float = 15.0,
) -> Callable:
    """Create a noise augmentation pipeline for training.

    Probability split:
    - 50% RealClass classroom noise at SNR 5-20 dB
    - 20% MUSAN babble noise at SNR 0-15 dB
    - 30% clean (no noise added)
    - All samples: optional Gain variation (+-6 dB, p=0.3)

    Args:
        noise_dir: Path to MUSAN noise directory.
        realclass_dir: Path to RealClass noise directory.
        sample_rate: Audio sample rate (default 16000).
        realclass_min_snr: Minimum SNR for RealClass mixing.
        realclass_max_snr: Maximum SNR for RealClass mixing.
        musan_min_snr: Minimum SNR for MUSAN mixing.
        musan_max_snr: Maximum SNR for MUSAN mixing.

    Returns:
        Callable with signature (audio: np.ndarray, sample_rate: int) -> np.ndarray
    """
    noise_dir = _validate_dir(noise_dir, "noise_dir")
    realclass_dir = _validate_dir(realclass_dir, "realclass_dir")

    transform = Compose([
        OneOf(
            [
                AddBackgroundNoise(
                    sounds_path=str(realclass_dir),
                    min_snr_db=realclass_min_snr,
                    max_snr_db=realclass_max_snr,
                    p=1.0,
                ),
                AddBackgroundNoise(
                    sounds_path=str(noise_dir),
                    min_snr_db=musan_min_snr,
                    max_snr_db=musan_max_snr,
                    p=1.0,
                ),
            ],
            weights=[5, 2],  # 50% RealClass, 20% MUSAN (when noise applied)
            p=0.7,
        ),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
    ])

    def augment_fn(audio: np.ndarray, sample_rate: int = sample_rate) -> np.ndarray:
        augmented = transform(samples=audio, sample_rate=sample_rate)
        return np.asarray(augmented, dtype=audio.dtype).ravel()

    return augment_fn


def create_noise_only_augmentation(
    noise_dir: str | Path,
    sample_rate: int = 16000,
    min_snr: float = 5.0,
    max_snr: float = 20.0,
    p: float = 1.0,
) -> Callable:
    """Create a single-source noise augmentation (for noisy validation).

    Args:
        noise_dir: Path to noise audio directory.
        sample_rate: Audio sample rate (default 16000).
        min_snr: Minimum SNR in dB.
        max_snr: Maximum SNR in dB.
        p: Probability of applying noise.

    Returns:
        Callable with signature (audio: np.ndarray, sample_rate: int) -> np.ndarray
    """
    noise_dir = _validate_dir(noise_dir, "noise_dir")

    transform = Compose([
        AddBackgroundNoise(
            sounds_path=str(noise_dir),
            min_snr_db=min_snr,
            max_snr_db=max_snr,
            p=p,
        ),
    ])

    def augment_fn(audio: np.ndarray, sample_rate: int = sample_rate) -> np.ndarray:
        augmented = transform(samples=audio, sample_rate=sample_rate)
        return np.asarray(augmented, dtype=audio.dtype).ravel()

    return augment_fn
