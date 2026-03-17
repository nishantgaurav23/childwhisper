# Explanation — S2.2 Whisper-small Training Script

## Why
The zero-shot Whisper-small baseline (Phase 1) has WER ~0.25-0.30 on children's speech. Full fine-tuning on competition data is the first major WER improvement, targeting ~0.15-0.20. This script bridges the gap between having a working dataset pipeline (S2.1) and actually training the model on Kaggle T4 GPUs.

## What
`src/train_whisper_small.py` — a complete fine-tuning script for `openai/whisper-small` that:

1. **Config-driven**: Loads hyperparameters from `configs/training_config.yaml`, merging `common` and `whisper_small` sections. CLI overrides for paths, epochs, output dir.

2. **Model setup**: Loads Whisper-small in fp16 with gradient checkpointing (fits T4 16GB) and SpecAugment enabled (mask_time_prob=0.05, mask_feature_prob=0.04) — a free regularization win.

3. **Data pipeline**: Uses `WhisperDataset` and `create_train_val_split` from S2.1 for child_id-based splitting with age_bucket stratification. No speaker leakage.

4. **Training**: HuggingFace `Seq2SeqTrainer` with lr=1e-5, warmup=500, batch=2, grad_accum=8 (effective=16), eval every 500 steps, early stopping on WER.

5. **Metrics**: WER computed via `jiwer` with `EnglishTextNormalizer` applied (reuses `compute_wer` from S1.5).

6. **Hub integration**: Pushes checkpoints to HuggingFace Hub (private repo) every 500 steps — critical since Kaggle sessions are ephemeral.

7. **Dry-run mode**: `--dry-run` runs 1 train step + 1 eval step on CPU for local testing.

## How

### Key functions:
- `load_training_config(path)` — YAML → merged dict
- `parse_args(argv)` — argparse with `--dry-run`, `--no-push-to-hub`, `--num-train-epochs`
- `setup_model(config)` — model + processor with SpecAugment + gradient ckpt
- `setup_training_args(config, ...)` — builds `Seq2SeqTrainingArguments`, with dry-run overrides
- `make_compute_metrics(tokenizer)` — returns WER metric function (handles -100 padding)
- `build_datasets(config, ...)` — loads metadata, splits, creates `WhisperDataset` instances
- `main(argv)` — orchestrates everything, returns validation WER

### Design decisions:
- **fp16 training** over fp32: halves VRAM, fits batch_size=2 on T4
- **gradient checkpointing**: trades compute for memory, essential for 242M param model on 16GB
- **SpecAugment ON by default**: Whisper ships with it disabled; enabling is the highest-ROI change
- **forced_decoder_ids=None**: prevents Whisper from forcing language/task tokens that may conflict with fine-tuning
- **Temp files for split metadata**: `WhisperDataset` expects a JSONL path, so we write split subsets to temp files. Simple, no API changes needed.

## Connections
- **Depends on**: S2.1 (WhisperDataset, WhisperDataCollator, create_train_val_split), S1.5 (compute_wer)
- **Feeds into**: S2.3 (Kaggle notebook wraps this script), S2.4 (inference loads fine-tuned weights)
- **Uses config from**: `configs/training_config.yaml`
- **Reuses**: `src/preprocess.load_metadata`, `src/evaluate.compute_wer`, `src/utils.normalize_text` (via evaluate)
