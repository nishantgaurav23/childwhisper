# Spec S4.1 — Noise Augmentation Pipeline

## Overview
Build a noise augmentation pipeline using `audiomentations` that mixes classroom noise (RealClass) and babble noise (MUSAN) into training audio at controlled SNR levels. The pipeline integrates with `WhisperDataset` via the existing `augment_fn` parameter.

## Dependencies
- S2.1 (PyTorch Dataset for Whisper) — done

## Location
- `src/augment.py`

## Functional Requirements

### AUG-1: Augmentation Compose Pipeline
- Build an `audiomentations.Compose` pipeline with configurable noise directories
- 50% of training samples: add RealClass background noise at SNR 5–20 dB
- 20% of training samples: add MUSAN babble noise at SNR 0–15 dB
- 30% of training samples: clean (no augmentation)
- All samples get optional Gain variation (±6 dB, p=0.3)

### AUG-2: Factory Function
- `create_augmentation(noise_dir, realclass_dir, sample_rate=16000)` → returns a callable `(audio, sample_rate) -> augmented_audio`
- Must accept `Path` or `str` for directories
- Must validate that directories exist (raise `FileNotFoundError` if not)

### AUG-3: Noise-Only Augmentation
- `create_noise_only_augmentation(noise_dir, snr_range, p)` — for creating single-source augmenters (used by noisy validation in S4.2)

### AUG-4: Integration with WhisperDataset
- The returned callable must match the `augment_fn` signature: `(audio: np.ndarray, sample_rate: int) -> np.ndarray`
- Output array must be same dtype and sample rate as input
- Output must be 1D (mono)

### AUG-5: Configurable SNR Ranges
- SNR ranges configurable via parameters with defaults from design doc
- RealClass: min_snr=5, max_snr=20
- MUSAN: min_snr=0, max_snr=15

## Non-Functional Requirements
- All audio stays at 16 kHz mono
- No external API calls
- Pure CPU processing (works on MacBook)
- Deterministic when seeded (for reproducibility)

## TDD Notes

### Test Strategy
- Mock `audiomentations` transforms to avoid needing actual noise files
- Test factory functions with `tmp_path` directories
- Test that output shape and dtype match input
- Test probability distribution (with enough samples, ~50/20/30 split)
- Test error handling for missing directories
- Test integration signature compatibility with `WhisperDataset.augment_fn`

### Key Test Cases
1. `create_augmentation` returns callable with correct signature
2. Output array is 1D, float32, same length as input
3. `FileNotFoundError` raised for nonexistent noise dirs
4. `create_noise_only_augmentation` returns callable
5. Augmented audio differs from input (when augmentation applied)
6. Clean pass-through preserves audio (when no augmentation applied)
7. Factory accepts both `str` and `Path` arguments

## Outcomes
- `src/augment.py` exists with `create_augmentation` and `create_noise_only_augmentation`
- `tests/test_augment.py` passes with >80% coverage
- Lint clean (`ruff`)
- Callable integrates with `WhisperDataset(augment_fn=...)` parameter
