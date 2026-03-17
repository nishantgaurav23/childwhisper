# Spec S2.3 — Kaggle Training Notebook (Whisper-small)

## Overview
Kaggle-ready Jupyter notebook and supporting utility module for training Whisper-small on children's speech data. The notebook orchestrates data setup, HuggingFace Hub authentication, training via the existing `train_whisper_small.py` script, checkpoint management (resume from HF Hub), and post-training evaluation. A `src/kaggle_utils.py` module extracts testable Kaggle-specific logic.

## Dependencies
- S2.2 (Whisper-small training script) — done

## Locations
- `src/kaggle_utils.py` — Kaggle environment utilities (testable)
- `notebooks/02_train_small.ipynb` — Kaggle training notebook
- `tests/test_kaggle_utils.py` — Unit tests

## Functional Requirements

### FR-1: Kaggle Environment Detection & Path Configuration
- `is_kaggle()` — detect if running inside Kaggle (check `/kaggle/` paths or env vars)
- `get_kaggle_paths(dataset_slug)` — return dict with `audio_dir`, `metadata_path`, `output_dir` for Kaggle
- `get_local_paths(data_dir)` — return same dict structure for local dev
- `get_paths(dataset_slug, local_data_dir)` — auto-detect environment and return correct paths

### FR-2: HuggingFace Hub Checkpoint Management
- `get_latest_checkpoint(hub_model_id)` — check if a checkpoint exists on HF Hub for resumption
- `setup_hub_auth()` — authenticate with HF Hub using `HF_TOKEN` env var or `huggingface_hub.login()`
- `download_checkpoint(hub_model_id, local_dir)` — download model checkpoint from Hub for resume

### FR-3: Training Configuration for Kaggle
- `get_kaggle_training_args(config_path, output_dir, resume_from)` — build CLI args list for `train_whisper_small.main()`
- Support resuming from a checkpoint directory
- Force `--no-push-to-hub` when HF_TOKEN is not available
- Default to Kaggle-optimized settings (batch_size=2, grad_accum=8, fp16)

### FR-4: Notebook Structure
The notebook (`02_train_small.ipynb`) must contain these cells in order:
1. **Setup** — pip install deps, import modules
2. **Environment** — detect Kaggle vs local, configure paths
3. **Data Verification** — check audio files exist, print dataset stats
4. **HF Hub Auth** — authenticate for checkpoint saving
5. **Training Config** — load YAML config, display key hyperparameters
6. **Resume Check** — check for existing checkpoint on Hub, download if found
7. **Train** — call `train_whisper_small.main()` with configured args
8. **Evaluate** — run validation, print WER summary
9. **Save** — push final model to HF Hub

### FR-5: Data Verification
- `verify_kaggle_data(audio_dir, metadata_path)` — check files exist, return stats dict with `num_utterances`, `num_audio_files`, `missing_audio`, `duration_stats`

## Non-Functional Requirements
- Notebook must run on Kaggle T4 (16 GB VRAM) within 6-hour session
- All Kaggle-specific logic in `src/kaggle_utils.py` (testable without Kaggle)
- No hardcoded paths — all via environment detection or config
- `ruff` clean (line-length: 100)

## Outcomes
1. `src/kaggle_utils.py` exists with all FR-1 through FR-5 functions
2. `notebooks/02_train_small.ipynb` exists with all required cells
3. Tests pass with >80% coverage of `kaggle_utils.py`
4. `ruff` clean on all new/modified files

## TDD Notes
- Mock filesystem operations for Kaggle path detection
- Mock HF Hub API calls (list_repo_refs, snapshot_download)
- Test path generation for both Kaggle and local environments
- Test data verification with synthetic metadata
- Test training args construction with various configs
- Test notebook structure validation (cell count, cell types)
