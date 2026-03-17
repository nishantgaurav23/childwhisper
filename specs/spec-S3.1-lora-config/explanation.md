# Explanation S3.1 — LoRA Configuration & Training Script

## Why
Phase 3 requires fine-tuning Whisper-large-v3 (1.55B params) on children's speech, but T4 GPUs only have 16GB VRAM — far too little for full fine-tuning. LoRA (Low-Rank Adaptation) with INT8 quantization reduces trainable parameters to ~15M (~1% of the model) and VRAM usage to ~8GB, making training feasible on free Kaggle T4s. This script is the foundation for the large model in the two-model ensemble.

## What
Created `src/train_whisper_lora.py` — a complete training script that:

1. **Loads config** from `configs/training_config.yaml`, merging `common` + `whisper_large_v3` sections
2. **Sets up Whisper-large-v3** with optional INT8 quantization via bitsandbytes
3. **Applies LoRA** (r=32, alpha=64, targeting q_proj + v_proj) via PEFT library
4. **Enables SpecAugment** for regularization (disabled by default in Whisper)
5. **Trains** with Seq2SeqTrainer, evaluating WER at each checkpoint
6. **Saves LoRA adapter only** (~63MB vs ~3GB for full weights)
7. **Optionally pushes** adapter to HuggingFace Hub for persistence across Kaggle sessions

Also created `tests/test_train_lora.py` with 12 tests covering config loading, CLI parsing, LoRA config creation, model setup, training args, dataset building, and metrics computation.

## How

### Architecture
The script mirrors `train_whisper_small.py` for consistency, with key differences:

- **Config section**: reads `whisper_large_v3` instead of `whisper_small`
- **Model loading**: supports `load_in_8bit=True` + `prepare_model_for_kbit_training()`
- **LoRA application**: `create_lora_config()` → `get_peft_model()`
- **Higher learning rate**: 1e-3 (LoRA convention) vs 1e-5 (full FT)
- **Smaller batch**: per_device=1 with grad_accum=16 (effective batch 16)
- **Adapter saving**: `model.save_pretrained()` saves only LoRA weights

### Key Functions
| Function | Purpose |
|----------|---------|
| `load_training_config()` | Merge YAML common + whisper_large_v3 sections |
| `parse_args()` | CLI with --dry-run, --no-push-to-hub |
| `create_lora_config()` | Build peft.LoraConfig from YAML |
| `setup_model()` | INT8 + LoRA + SpecAugment + clear forced_decoder_ids |
| `setup_training_args()` | Seq2SeqTrainingArguments with dry-run support |
| `make_compute_metrics()` | WER computation via jiwer |
| `build_datasets()` | Train/val split by child_id, create WhisperDatasets |
| `main()` | Orchestrate training, return validation WER |

### Testing Strategy
All 12 tests mock model/processor loading to avoid downloading weights. Tests verify:
- Config merging and LoRA param extraction
- CLI argument defaults and flags
- LoRA config correctness (r, alpha, target_modules)
- SpecAugment and forced_decoder_ids configuration
- Training args for normal and dry-run modes
- Dataset splitting and creation
- WER metric computation

## Connections
- **Depends on S2.1**: Reuses `WhisperDataset`, `WhisperDataCollator`, `create_train_val_split`
- **Used by S3.2**: Kaggle notebook will call this script for LoRA training
- **Used by S3.3**: Ensemble inference loads the LoRA adapter produced by this script
- **Used by S4.3**: Augmented retraining will reuse this script with augmented data
