# Spec S3.2 — Kaggle LoRA Training Notebook

## Overview
Kaggle-ready Jupyter notebook for training Whisper-large-v3 with LoRA on children's speech data. Extends the existing `kaggle_utils.py` module with LoRA-specific path/arg helpers and creates `notebooks/03_train_lora.ipynb`. The notebook orchestrates INT8 model loading, LoRA adapter training via `train_whisper_lora.py`, HF Hub checkpoint management for LoRA adapters, and post-training evaluation.

## Dependencies
- S3.1 (LoRA configuration & training script) — done

## Locations
- `src/kaggle_utils.py` — Extended with LoRA-specific helpers
- `notebooks/03_train_lora.ipynb` — Kaggle LoRA training notebook
- `tests/test_kaggle_utils_lora.py` — Unit tests for LoRA extensions

## Functional Requirements

### FR-1: LoRA-Specific Path Configuration
- `get_kaggle_paths_lora(dataset_slug)` — return dict with `audio_dir`, `metadata_path`, `output_dir` pointing to LoRA-specific output directory (`/kaggle/working/checkpoints/whisper-large-v3-lora`)
- `get_local_paths_lora(data_dir)` — return same dict structure for local dev with LoRA output dir (`./checkpoints/whisper-large-v3-lora`)
- `get_paths_lora(dataset_slug, local_data_dir)` — auto-detect environment and return correct LoRA paths

### FR-2: LoRA Training Argument Construction
- `get_lora_training_args(config_path, metadata_path, audio_dir, output_dir, resume_from, num_epochs, dry_run)` — build CLI args list for `train_whisper_lora.main()`
- Force `--no-push-to-hub` when HF_TOKEN is not available
- Support `--dry-run` flag passthrough

### FR-3: LoRA Adapter Checkpoint Management
- Reuse existing `get_latest_checkpoint()` and `download_checkpoint()` from S2.3 (they work for any HF repo)
- LoRA adapter checkpoints are ~63 MB (much smaller than full model)

### FR-4: Notebook Structure
The notebook (`03_train_lora.ipynb`) must contain these cells in order:
1. **Setup** — pip install deps (peft, bitsandbytes, jiwer, audiomentations)
2. **Environment** — detect Kaggle vs local, configure LoRA-specific paths
3. **Data Verification** — check audio files exist, print dataset stats
4. **HF Hub Auth** — authenticate for adapter saving
5. **Training Config** — load YAML config, display LoRA hyperparameters (r, alpha, target_modules, INT8, LR)
6. **Resume Check** — check for existing LoRA adapter on Hub, download if found
7. **Train** — call `train_whisper_lora.main()` with configured args
8. **Evaluate** — print WER summary
9. **Save** — push LoRA adapter to HF Hub

### FR-5: GPU Memory Verification
- `check_gpu_memory()` — return dict with `gpu_name`, `total_memory_gb`, `is_sufficient` (True if >= 14 GB for INT8+LoRA)
- Display GPU info in notebook before training to catch OOM early

## Non-Functional Requirements
- Notebook must run on Kaggle T4 (16 GB VRAM) within 8-hour session
- All LoRA-specific logic in `src/kaggle_utils.py` (testable without Kaggle/GPU)
- No hardcoded paths — all via environment detection or config
- `ruff` clean (line-length: 100)
- Reuse existing `kaggle_utils.py` functions where possible (is_kaggle, setup_hub_auth, verify_kaggle_data, etc.)

## Outcomes
1. `src/kaggle_utils.py` extended with FR-1, FR-2, FR-5 functions
2. `notebooks/03_train_lora.ipynb` exists with all 9 required cells
3. Tests pass with >80% coverage of new LoRA functions
4. `ruff` clean on all new/modified files

## TDD Notes
- Mock filesystem operations for Kaggle path detection
- Mock torch.cuda for GPU memory checks
- Test LoRA path generation for both Kaggle and local environments
- Test training args construction with various configs (dry-run, with/without HF_TOKEN)
- Test notebook structure validation (cell count, cell types)
