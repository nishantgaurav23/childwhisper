# Explanation S2.3 — Kaggle Training Notebook (Whisper-small)

## Why
The training script from S2.2 (`train_whisper_small.py`) was designed for general use, but Kaggle notebooks have specific constraints: ephemeral storage, dataset mounting at `/kaggle/input/`, limited GPU sessions, and the need to resume training from HuggingFace Hub checkpoints across sessions. This spec bridges the gap between the local training script and the Kaggle execution environment.

## What
Two artifacts were created:

1. **`src/kaggle_utils.py`** — A utility module with 8 functions covering:
   - **Environment detection** (`is_kaggle`): Checks for `KAGGLE_KERNEL_RUN_TYPE` env var or `/kaggle/working` path
   - **Path configuration** (`get_paths`, `get_kaggle_paths`, `get_local_paths`): Returns correct audio/metadata/output paths for either Kaggle or local development
   - **HF Hub integration** (`setup_hub_auth`, `get_latest_checkpoint`, `download_checkpoint`): Manages authentication and checkpoint persistence across ephemeral Kaggle sessions
   - **Training arg construction** (`get_kaggle_training_args`): Builds CLI argument list for `train_whisper_small.main()` with Kaggle-appropriate defaults
   - **Data verification** (`verify_kaggle_data`): Validates that uploaded data is accessible and returns statistics

2. **`notebooks/02_train_small.ipynb`** — A 20-cell Jupyter notebook with 9 logical sections:
   Setup → Environment → Data Verification → HF Auth → Config Display → Resume Check → Train → Evaluate → Save

## How
- **Separation of concerns**: All testable logic lives in `kaggle_utils.py`, not the notebook. The notebook is pure orchestration — calling utility functions and displaying results.
- **Auto-detection**: `get_paths()` automatically selects Kaggle vs local paths, making the notebook portable between environments.
- **Checkpoint resumption**: Before training, the notebook checks HF Hub for existing checkpoints and downloads them if found. This handles Kaggle's 12-hour session limit — if training is interrupted, the next session picks up from the last Hub checkpoint.
- **Graceful degradation**: If HF_TOKEN is not set, Hub push is disabled but training still works with local-only checkpoints.
- **23 unit tests** cover all utility functions with mocked filesystem and HF Hub API calls.

## Connections
- **Depends on S2.2**: Calls `train_whisper_small.main()` directly with constructed CLI args
- **Depends on S2.1**: Training script uses `WhisperDataset` and `WhisperDataCollator`
- **Feeds into S2.4**: The fine-tuned model checkpoint saved to HF Hub will be downloaded by the inference pipeline
- **Pattern reused by S3.2**: The Kaggle LoRA training notebook will follow the same structure and reuse `kaggle_utils.py`
