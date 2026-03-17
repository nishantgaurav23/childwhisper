# Spec S4.3 — Retrain with Augmented Data

## Overview
- **Spec ID**: S4.3
- **Phase**: 4 (Noise Augmentation)
- **Depends On**: S4.1 (Augmentation Pipeline), S3.1 (LoRA Config & Training)
- **Location**: `notebooks/04_augmented.ipynb`, `src/train_whisper_small.py`, `src/train_whisper_lora.py`, `configs/training_config.yaml`
- **Status**: pending

## Problem
The existing training scripts for Whisper-small and Whisper-large-v3 LoRA train on clean audio only. S4.1 built a noise augmentation pipeline (`create_augmentation()`) and S4.2 built noisy validation. Now we need to wire augmentation into training so both models learn noise-robust features, improving Noisy WER for the classroom bonus prize.

## Requirements

### R1: Augmentation CLI Arguments
Both `train_whisper_small.py` and `train_whisper_lora.py` must accept:
- `--noise-dir`: Path to MUSAN noise directory (optional)
- `--realclass-dir`: Path to RealClass noise directory (optional)
- When both are provided, create augmentation pipeline via `create_augmentation()` from `src/augment.py` and pass as `augment_fn` to `build_datasets()`
- When neither is provided, training proceeds on clean data (backward compatible)
- When only one is provided, raise an error (both are required for the 50/20/30 split)

### R2: Augmentation Config in training_config.yaml
Add an `augmentation` section under `common`:
```yaml
augmentation:
  realclass_min_snr_db: 5.0
  realclass_max_snr_db: 20.0
  musan_min_snr_db: 0.0
  musan_max_snr_db: 15.0
```
Training scripts read these values and pass to `create_augmentation()`.

### R3: Noisy Validation Reporting
After training completes, both scripts should:
- Evaluate on clean validation set (already done)
- If noise dirs were provided, also evaluate on noisy validation set using `apply_noise_to_val()` from `src/evaluate.py`
- Log both clean WER and noisy WER
- Return clean WER (primary metric for early stopping, unchanged)

### R4: Kaggle Notebook
Create `notebooks/04_augmented.ipynb` that:
1. Installs dependencies (same as 02/03 notebooks)
2. Downloads competition data + RealClass noise + MUSAN babble
3. Runs `train_whisper_lora.py` with `--noise-dir` and `--realclass-dir`
4. Optionally runs `train_whisper_small.py` with same noise args
5. Pushes augmented checkpoints to HuggingFace Hub (separate repo IDs to not overwrite clean models)
6. Prints combined clean/noisy validation report

### R5: Hub Model IDs for Augmented Models
- Augmented LoRA: `nishantgaurav23/pasketti-whisper-lora-augmented`
- Augmented Small: `nishantgaurav23/pasketti-whisper-small-augmented`
These are configured in `training_config.yaml` and overridable via CLI `--hub-model-id`.

## Outcomes
1. Both training scripts accept `--noise-dir` and `--realclass-dir` CLI args
2. Augmentation config is in `training_config.yaml`
3. Noisy validation metrics are logged when augmentation is enabled
4. Kaggle notebook `04_augmented.ipynb` is ready to run
5. Backward compatibility: running without noise args produces identical behavior to before

## TDD Notes
- Test that CLI args are parsed correctly (with and without noise dirs)
- Test that augmentation is wired into build_datasets when noise dirs provided
- Test that augmentation is NOT used when noise dirs omitted
- Test error when only one noise dir provided
- Test noisy validation reporting integration
- Test augmentation config loading from YAML
- Mock all model loading / training — focus on wiring and config
- Test notebook cell structure and content

## Out of Scope
- Actual model training (runs on Kaggle GPU)
- Hyperparameter tuning of SNR ranges (use defaults from S4.1)
- Updating submission/main.py to use augmented weights (that's inference config, not training)
