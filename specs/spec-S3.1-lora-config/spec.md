# Spec S3.1 — LoRA Configuration & Training Script

## Overview
Create `src/train_whisper_lora.py` — a training script for Whisper-large-v3 with LoRA (Low-Rank Adaptation) and INT8 quantization. Mirrors the architecture of `train_whisper_small.py` but adds PEFT/LoRA setup, bitsandbytes INT8 loading, and LoRA-specific training arguments.

## Depends On
- **S2.1** (PyTorch Dataset for Whisper) — reuses `WhisperDataset`, `WhisperDataCollator`, `create_train_val_split`

## Inputs
- `configs/training_config.yaml` (`whisper_large_v3` section + `common` section)
- Metadata JSONL + audio directory (same as whisper-small training)
- CLI arguments: `--metadata-path`, `--audio-dir`, `--config`, `--output-dir`, `--dry-run`, `--no-push-to-hub`

## Outputs
- `src/train_whisper_lora.py` — complete training script
- LoRA adapter saved to output dir (adapter_model.safetensors + adapter_config.json)
- Optional push to HuggingFace Hub

## Key Requirements

### R1: Config Loading
- Load YAML config merging `common` + `whisper_large_v3` sections
- Support all LoRA hyperparameters: r, alpha, target_modules, dropout, bias, task_type
- Support INT8 quantization flag (`load_in_8bit`)

### R2: Model Setup
- Load `openai/whisper-large-v3` with optional INT8 quantization via `bitsandbytes`
- Apply `prepare_model_for_kbit_training()` when using INT8
- Configure LoRA via `peft.LoraConfig` + `get_peft_model()`
- Enable SpecAugment (from config)
- Clear `forced_decoder_ids`, set empty `suppress_tokens`

### R3: Training Arguments
- Higher learning rate (1e-3 default for LoRA vs 1e-5 for full FT)
- Smaller batch size (per_device=1) with larger gradient accumulation (16)
- Support `--dry-run` mode (1 step, no hub push, no fp16, no grad ckpt)
- Support `adamw_8bit` optimizer

### R4: Training Loop
- Use `Seq2SeqTrainer` with `predict_with_generate=True`
- `compute_metrics` using WER (same as whisper-small)
- Save LoRA adapter only (not full model weights)
- Optional HF Hub push

### R5: CLI Interface
- Same arg structure as `train_whisper_small.py` for consistency
- `main(argv)` signature for testability
- Returns validation WER as float

## Architecture Notes
- The script mirrors `train_whisper_small.py` structure: `load_training_config()`, `parse_args()`, `setup_model()`, `setup_training_args()`, `make_compute_metrics()`, `build_datasets()`, `main()`
- Key differences: config section is `whisper_large_v3`, model setup includes INT8 + LoRA, adapter-only saving

## TDD Plan

### Red Phase — Tests to write first:
1. `test_load_lora_config` — loads whisper_large_v3 + common sections from YAML
2. `test_parse_args_defaults` — default CLI arguments
3. `test_parse_args_dry_run` — dry-run flag behavior
4. `test_setup_lora_config` — LoraConfig created with correct r, alpha, target_modules
5. `test_setup_model_spec_augment` — SpecAugment config applied
6. `test_setup_model_forced_decoder_ids_cleared` — forced_decoder_ids=None
7. `test_setup_training_args_lora_defaults` — LR=1e-3, batch=1, grad_accum=16
8. `test_setup_training_args_dry_run` — max_steps=1, no hub, no fp16
9. `test_build_datasets` — returns train/val WhisperDatasets
10. `test_main_dry_run` — full dry-run integration test (mocked model)

### Green Phase — Implement minimum code to pass each test
### Refactor Phase — Clean up, run ruff, ensure all tests pass

## Acceptance Criteria
- [ ] All tests pass
- [ ] `ruff check src/train_whisper_lora.py` clean
- [ ] Config loads LoRA params from YAML correctly
- [ ] Model loads with INT8 + LoRA when configured
- [ ] SpecAugment enabled
- [ ] Dry-run completes without GPU
- [ ] Adapter-only saving (not full model weights)
