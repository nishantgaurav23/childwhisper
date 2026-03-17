# Spec S2.2 — Whisper-small Training Script

## Overview
Full fine-tuning script for `openai/whisper-small` on children's speech data. Integrates with the existing `WhisperDataset` (S2.1) and validation framework (S1.5). Designed to run on Kaggle T4 GPU and be testable on MacBook (MPS/CPU).

## Dependencies
- S2.1 (PyTorch Dataset for Whisper) — done
- S1.5 (Local validation framework) — done

## Location
- `src/train_whisper_small.py`

## Functional Requirements

### FR-1: Config Loading
- Load training config from `configs/training_config.yaml` (`whisper_small` + `common` sections)
- Support CLI overrides for key params (data paths, output dir, epochs, etc.)
- Support `--dry-run` flag for local testing (1 step train + 1 step eval)

### FR-2: Model Setup
- Load `openai/whisper-small` with fp16
- Enable gradient checkpointing
- Enable SpecAugment (mask_time_prob=0.05, mask_feature_prob=0.04)
- Set forced_decoder_ids for English transcription
- Suppress special token generation (notimestamps)

### FR-3: Data Pipeline
- Use `WhisperDataset` from `src/dataset.py` for train and val splits
- Use `create_train_val_split` from `src/dataset.py` for child_id-based splitting
- Use `WhisperDataCollator` for batched padding
- Support optional `augment_fn` passthrough to dataset

### FR-4: Training Loop
- Use HuggingFace `Seq2SeqTrainer` with `Seq2SeqTrainingArguments`
- Training args from config: lr=1e-5, warmup=500, epochs=3, batch=2, grad_accum=8
- fp16 training, gradient checkpointing
- Evaluation every 500 steps with WER metric
- Save checkpoints every 500 steps, keep best 3
- Load best model at end based on validation WER

### FR-5: WER Metric Computation
- Compute WER during evaluation using `jiwer` + `EnglishTextNormalizer`
- Use `compute_wer` from `src/evaluate.py` or equivalent in-trainer metric
- Log WER per evaluation step

### FR-6: HuggingFace Hub Integration
- Push checkpoints to HF Hub (configurable repo ID)
- Private repo by default
- Support disabling hub push for local testing

### FR-7: Entry Point
- `main()` function that orchestrates: config load -> data split -> model init -> train -> evaluate
- CLI via argparse
- Return validation WER at end

## Non-Functional Requirements
- Must run on T4 (16 GB VRAM) with batch_size=2, grad_accum=8
- Must be testable on MacBook (CPU/MPS) with `--dry-run`
- No hardcoded paths — all paths via config or CLI
- Use `ruff` compatible code (line-length: 100)

## Outcomes
1. `src/train_whisper_small.py` exists and is importable
2. Training runs end-to-end with `--dry-run` on CPU
3. WER is computed and logged during evaluation
4. Config is loaded from YAML with CLI overrides
5. All tests pass with >80% coverage of the module

## TDD Notes
- Mock model loading, HF Hub push, and actual training in unit tests
- Test config loading, argument parsing, model setup, metric computation
- Test dry-run mode with tiny synthetic data
- Test that SpecAugment is enabled on model config
- Test that gradient checkpointing is enabled
