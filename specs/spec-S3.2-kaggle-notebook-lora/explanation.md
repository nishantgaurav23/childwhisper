# Explanation S3.2 — Kaggle LoRA Training Notebook

## Why
The LoRA training script (`train_whisper_lora.py` from S3.1) needs a Kaggle-ready notebook to actually run on GPU hardware. Kaggle T4 GPUs are the primary free training resource, and notebooks must handle environment detection, checkpoint persistence across ephemeral sessions, and GPU memory validation before attempting INT8+LoRA training. This spec bridges the gap between "training script exists" and "training actually happens."

## What
**Extended `src/kaggle_utils.py`** with 5 new functions:
- `get_kaggle_paths_lora()` / `get_local_paths_lora()` / `get_paths_lora()` — LoRA-specific path routing (output to `whisper-large-v3-lora` dir instead of `whisper-small`)
- `get_lora_training_args()` — builds CLI args for `train_whisper_lora.main()` with HF_TOKEN awareness
- `check_gpu_memory()` — validates GPU has sufficient VRAM (>=14 GB) before training starts

**Created `notebooks/03_train_lora.ipynb`** with 9 sections:
1. Setup (pip install deps including bitsandbytes)
2. Environment & Paths (auto-detect Kaggle vs local)
3. Data Verification (check audio files accessible)
4. HF Hub Auth (for adapter persistence)
5. Training Config & GPU Check (display LoRA hyperparams + VRAM check)
6. Resume Check (download existing adapter from Hub)
7. Train (call `train_whisper_lora.main()`)
8. Evaluate (WER summary)
9. Save (push LoRA adapter to Hub)

**Created `tests/test_kaggle_utils_lora.py`** — 19 tests covering all new functions + notebook structure validation.

## How
- Followed the exact same pattern as S2.3 (Kaggle small notebook) for consistency
- Reused existing `kaggle_utils.py` functions (is_kaggle, setup_hub_auth, verify_kaggle_data, get_latest_checkpoint, download_checkpoint) — no duplication
- Added `import torch` to kaggle_utils.py for GPU memory detection
- LoRA path functions mirror the existing whisper-small path functions but output to `whisper-large-v3-lora` directory
- GPU memory check uses `torch.cuda.get_device_properties()` with a configurable minimum threshold (default 14 GB for INT8+LoRA)
- Notebook uses `train_lora_main(train_args)` — same callable pattern as the small notebook

## Connections
- **Depends on S3.1** — uses `train_whisper_lora.main()` as the training entrypoint
- **Reuses S2.3** — same `kaggle_utils.py` module, same patterns for Hub auth, checkpoint management, data verification
- **Feeds into S3.3** — the trained LoRA adapter (saved to HF Hub) will be used by the ensemble inference pipeline
- **Feeds into S4.3** — the same notebook pattern will be reused for augmented retraining
