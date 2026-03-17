# Explanation — S4.3 Retrain with Augmented Data

## Why
The models trained in Phases 2-3 were trained on clean audio only. The competition includes a Noisy WER bonus for classroom environments with background noise (children talking, HVAC, furniture). By training on noise-augmented data using the pipeline from S4.1 (RealClass + MUSAN noise at various SNR levels), both models learn noise-robust features, improving performance on noisy test data.

## What Was Built

### CLI Extensions
Both `train_whisper_small.py` and `train_whisper_lora.py` now accept:
- `--noise-dir`: Path to MUSAN noise directory
- `--realclass-dir`: Path to RealClass noise directory
- `--hub-model-id`: Override HuggingFace Hub model ID (for saving augmented models separately)

### `create_augment_fn()` (in both training scripts)
A helper that validates CLI args and creates the augmentation callable from S4.1:
- Both dirs provided → creates augmentation pipeline with 50/20/30 split
- Neither provided → returns None (clean training, backward compatible)
- Only one provided → raises `ValueError` (prevents misconfiguration)

Reads SNR ranges from the new `augmentation` section in `training_config.yaml`.

### Config Addition
Added `augmentation` section under `common` in `configs/training_config.yaml`:
```yaml
augmentation:
  realclass_min_snr_db: 5.0
  realclass_max_snr_db: 20.0
  musan_min_snr_db: 0.0
  musan_max_snr_db: 15.0
```

### Kaggle Notebook
`notebooks/04_augmented.ipynb` orchestrates augmented training:
1. Installs dependencies
2. Configures paths for audio data + noise directories
3. Trains Whisper-large-v3 LoRA with augmented data
4. Trains Whisper-small with augmented data
5. Saves to separate HuggingFace Hub repos (`*-augmented`)

### Hub Model IDs
- Augmented LoRA: `nishantgaurav23/pasketti-whisper-lora-augmented`
- Augmented Small: `nishantgaurav23/pasketti-whisper-small-augmented`

## How It Works

The key integration point is the `augment_fn` parameter already supported by `WhisperDataset` (from S2.1) and `build_datasets()` (from S2.2/S3.1). S4.3 wires the `main()` function of each training script to:

1. Parse `--noise-dir` and `--realclass-dir` from CLI
2. Call `create_augment_fn()` which delegates to `create_augmentation()` from S4.1
3. Pass the resulting callable to `build_datasets()`, which passes it to `WhisperDataset`
4. During training, each audio sample has a 50% chance of RealClass noise, 20% MUSAN noise, 30% clean

The `--hub-model-id` override allows saving augmented models to separate repos without overwriting clean model checkpoints.

## Connections
- **S4.1** (Augmentation Pipeline): Provides `create_augmentation()` that this spec wires into training
- **S4.2** (Noisy Validation): Provides `apply_noise_to_val()` for evaluating noise robustness
- **S3.1** (LoRA Config): LoRA training script receives augmentation support
- **S2.2** (Whisper-small Training): Small training script receives augmentation support
- **S2.1** (Dataset Class): `WhisperDataset.augment_fn` parameter is the foundation
- **S5.x** (Phase 5): Augmented models feed into final submission packaging

## Test Coverage
26 new tests covering:
- CLI argument parsing (noise dirs, hub model ID, defaults)
- Augmentation config loading from YAML
- `create_augment_fn()` wiring (both dirs, no dirs, error on one dir)
- Hub model ID override
- Notebook existence and structure
- Backward compatibility (no augment when no noise dirs)
