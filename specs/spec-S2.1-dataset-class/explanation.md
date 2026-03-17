# Explanation S2.1 — PyTorch Dataset for Whisper

## Why
Training Whisper models requires a PyTorch Dataset that bridges raw competition data (FLAC audio + JSONL metadata) with HuggingFace's `Seq2SeqTrainer`. This spec creates the data loading layer that all downstream training specs (S2.2 whisper-small, S3.1 LoRA) depend on. Without it, there's no way to feed preprocessed audio and tokenized transcripts into the training loop.

The train/val split function enforces the critical constraint of splitting by `child_id` (not by utterance) to prevent speaker leakage, which would inflate validation metrics and give a false sense of model performance.

## What

### `WhisperDataset` (PyTorch Dataset)
- Loads JSONL metadata via `load_metadata()` from S1.2
- Filters invalid entries at init: missing audio files, out-of-range durations
- On `__getitem__`: loads audio → trims silence → checks for silence → applies optional augmentation → extracts log-Mel features → tokenizes transcript
- Uses `WhisperProcessor` (feature extractor + tokenizer) from any Whisper model variant
- Silence detected via `is_silence()` from S1.2 produces empty transcript tokens
- Augmentation injected as callable `augment_fn(audio, sample_rate=sr)` for Phase 4 extensibility

### `WhisperDataCollator`
- Stacks `input_features` tensors (all same shape from feature extractor)
- Pads `labels` to max length in batch using `-100` (ignored by cross-entropy loss)
- Returns dict of `torch.Tensor` ready for `Seq2SeqTrainer`

### `create_train_val_split()`
- Groups children by `age_bucket`, selects `val_ratio` of children from each bucket
- Ensures zero `child_id` overlap between train and val (no speaker leakage)
- Deterministic via seed parameter for reproducible experiments
- Stratified sampling ensures proportional age representation in both splits

## How

### Architecture decisions
1. **Pre-filtering at init**: Invalid entries (missing files, bad durations) are filtered once during `__init__`, not at every `__getitem__` call. This prevents training loop failures and wasted GPU time.

2. **Augmentation as injectable callable**: Rather than hardcoding noise augmentation, the dataset accepts an optional `augment_fn`. This keeps S2.1 clean and lets S4.1 (augmentation pipeline) compose freely.

3. **WhisperProcessor over separate components**: Using the combined processor ensures feature extractor and tokenizer stay in sync for a given model variant.

4. **Labels padded with -100**: Standard HuggingFace convention — `CrossEntropyLoss` ignores positions with label `-100`, so padding doesn't affect gradients.

5. **Stratified child_id split**: Using `random.Random(seed)` for deterministic shuffling, then selecting proportional children from each age bucket. The `max(1, round(...))` ensures every bucket contributes at least one child to validation.

## Connections
- **Upstream**: Depends on S1.2 (`preprocess.py` functions) and S1.3 (`normalize_text`)
- **Downstream**: S2.2 (training script) will instantiate `WhisperDataset` + `WhisperDataCollator` and pass to `Seq2SeqTrainer`. S3.1 (LoRA training) will use the same dataset class with a different `model_name`. S4.1 (augmentation) will provide `augment_fn` to inject noise mixing.
- **Split function**: Used by both S2.2 and S1.5 (validation framework) to ensure consistent train/val splits across training and evaluation.

## Files
- `src/dataset.py` — WhisperDataset, WhisperDataCollator, create_train_val_split
- `tests/test_dataset.py` — 14 tests covering init, getitem, silence, filtering, collation, splitting
