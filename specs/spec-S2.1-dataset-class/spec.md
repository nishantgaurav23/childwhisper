# Spec S2.1 — PyTorch Dataset for Whisper

## Overview
Build a PyTorch `Dataset` class that loads competition audio data, preprocesses it via the existing pipeline (S1.2), tokenizes transcripts with Whisper's tokenizer (S1.3), extracts log-Mel features with Whisper's feature extractor, and provides a data collator for padded batching during training.

## Dependencies
- **S1.2** (Audio preprocessing): `src/preprocess.py` — `load_audio`, `trim_silence`, `is_silence`, `get_duration`, `is_valid_duration`, `load_metadata`
- **S1.3** (Text normalization): `src/utils.py` — `normalize_text`

## Location
`src/dataset.py`

## Functional Requirements

### FR-1: WhisperDataset class
- Accepts: path to JSONL metadata, path to audio directory, model name (e.g., `"openai/whisper-small"`)
- At init: loads metadata via `load_metadata()`, instantiates `WhisperProcessor` (feature extractor + tokenizer)
- Optional: augmentation transform callable (for Phase 4 noise augmentation)
- `__len__`: returns number of valid samples
- `__getitem__`: returns dict with `input_features` (log-Mel spectrogram) and `labels` (token IDs)

### FR-2: Audio processing in __getitem__
- Load audio with `load_audio()` (16 kHz mono)
- Trim silence with `trim_silence()`
- Check silence with `is_silence()` — if silent, set transcript to `""`
- Check duration with `is_valid_duration()` — skip invalid (but these should be filtered at init)
- Apply augmentation transform if provided
- Extract log-Mel features via `WhisperFeatureExtractor`

### FR-3: Transcript tokenization
- Normalize transcript with `normalize_text()`
- Tokenize with `WhisperTokenizer` (via processor)
- Set language to English, task to "transcribe"

### FR-4: Filtering at init time
- Pre-filter metadata: skip entries where audio file doesn't exist
- Pre-filter by duration bounds (configurable, default 0.3–30s) — check `audio_duration_sec` field in metadata if available, otherwise defer to runtime
- Store only valid entries in internal list

### FR-5: Data collator
- `WhisperDataCollator` class (or function) that:
  - Pads `input_features` to max length in batch (using feature extractor)
  - Pads `labels` to max length in batch, using `-100` as padding token (ignored by loss)
  - Returns batched tensors ready for `Seq2SeqTrainer`

### FR-6: Train/validation split helper
- `create_train_val_split(metadata, val_ratio=0.1, split_by="child_id", stratify_by="age_bucket")` function
- Split by unique `child_id` values (no speaker leakage)
- Stratify by `age_bucket` to ensure proportional representation
- Returns two lists of metadata dicts (train, val)
- Deterministic with seed parameter

## Non-Functional Requirements
- Must work on MacBook (CPU/MPS) — no CUDA-only code paths
- Feature extractor and tokenizer loaded lazily or via processor
- Memory efficient: load audio on-the-fly in `__getitem__`, don't cache waveforms
- Compatible with PyTorch DataLoader (num_workers >= 0)

## Key Design Decisions
- Use `WhisperProcessor` (combines feature extractor + tokenizer) rather than separate objects
- Pre-filter metadata at init to avoid runtime failures in training loop
- Collator pads labels with -100 (standard HuggingFace convention for ignored tokens)
- Augmentation is injected as a callable, not built into the dataset (separation of concerns for S4.1)

## TDD Plan

### Red Phase — Tests to write first:
1. **test_dataset_init**: Dataset loads metadata, filters invalid entries, stores correct count
2. **test_dataset_getitem**: Returns dict with `input_features` (correct shape) and `labels` (token IDs)
3. **test_dataset_silence_handling**: Silent audio returns empty transcript tokens
4. **test_dataset_missing_audio**: Missing audio files are filtered out at init
5. **test_dataset_augmentation**: When augment callable provided, it's applied to audio
6. **test_collator_padding**: Collator pads features and labels correctly, labels padded with -100
7. **test_collator_batch_shapes**: Output batch has consistent tensor shapes
8. **test_train_val_split_no_leakage**: No child_id appears in both train and val sets
9. **test_train_val_split_stratification**: Age buckets represented proportionally in both splits
10. **test_train_val_split_deterministic**: Same seed produces same split
11. **test_dataset_with_processor**: Verifies correct WhisperProcessor model name loading

### Green Phase:
- Implement minimum code in `src/dataset.py` to pass each test

### Refactor Phase:
- Run ruff, fix lint issues
- Ensure all 133 existing tests + new tests pass
