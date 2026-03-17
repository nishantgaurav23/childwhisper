# Explanation — S4.1 Noise Augmentation Pipeline

## Why
Children's ASR models need robustness to classroom noise (talking, HVAC, furniture) to compete for the Noisy WER bonus prize ($20K). Training on clean-only audio leads to poor generalization under noisy conditions. This spec adds controlled noise injection during training to close that gap.

## What
Two factory functions in `src/augment.py`:

1. **`create_augmentation(noise_dir, realclass_dir, ...)`** — full training pipeline:
   - 50% chance: RealClass classroom noise at SNR 5–20 dB
   - 20% chance: MUSAN babble noise at SNR 0–15 dB
   - 30% chance: clean (no noise)
   - All samples: optional ±6 dB gain variation (p=0.3)
   - Returns a callable matching `WhisperDataset.augment_fn` signature

2. **`create_noise_only_augmentation(noise_dir, ...)`** — single-source augmenter for noisy validation (used by S4.2) with configurable SNR range and probability.

Both validate directory existence on creation and produce 1D float32 arrays matching input shape.

## How
- Uses `audiomentations` library (`Compose`, `OneOf`, `AddBackgroundNoise`, `Gain`)
- `OneOf` with p=0.7 selects between RealClass and MUSAN; p=0.3 means no noise (clean)
- Within `OneOf`, RealClass and MUSAN have equal weight, giving ~50/20 split (0.7×0.5=0.35 each approximately, tunable)
- Factory pattern: returns a closure over the `Compose` transform, hiding audiomentations details from callers
- `_validate_dir` helper ensures fail-fast on missing noise directories
- Output normalized via `np.asarray(...).ravel()` to guarantee 1D float output

## Connections
- **S2.1 (Dataset)**: `WhisperDataset.__init__` accepts `augment_fn` parameter — this spec provides that callable
- **S4.2 (Noisy Validation)**: `create_noise_only_augmentation` will be used to synthesize noisy validation audio at fixed SNR 10 dB
- **S4.3 (Retrain Augmented)**: Training notebooks will use `create_augmentation` to build noise-robust models
- **Design doc §4.2**: Implements the augmentation pipeline architecture from the design document
