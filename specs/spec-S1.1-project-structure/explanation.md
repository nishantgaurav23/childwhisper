# Explanation S1.1 — Project Structure & Dependencies

## Why
This is the foundational spec. Every subsequent spec (S1.2–S5.4) depends on having a well-organized project structure, pinned dependencies, and proper Python packaging. Without this, imports break, configs are scattered, and onboarding to Kaggle/Colab environments becomes fragile.

## What Was Built

### requirements.txt
All Python dependencies pinned with minimum versions. Covers:
- **Core ASR**: torch, torchaudio, transformers, peft, accelerate, bitsandbytes, datasets
- **Audio**: librosa, soundfile
- **Evaluation**: jiwer
- **Augmentation**: audiomentations
- **Dev tools**: pytest, ruff, pyyaml, jupyter, pandas, matplotlib, tqdm

All dependencies are free/open-source, matching the $0-10/month budget constraint.

### configs/training_config.yaml
Centralized training configuration with three sections:
- **common**: Sample rate (16kHz), duration filters, silence threshold, SpecAugment settings, validation split strategy
- **whisper_small**: Full fine-tune hyperparameters (lr=1e-5, batch=2, grad_accum=8, 3 epochs)
- **whisper_large_v3**: LoRA fine-tune hyperparameters (lr=1e-3, INT8, LoRA r=32/alpha=64 on q_proj+v_proj)
- **inference**: Beam search settings (beam=5, batch=16, 90-min time budget)

### Python Packages
`__init__.py` files in `src/`, `tests/`, `submission/`, and `submission/utils/` making them proper importable packages. `tests/conftest.py` provides shared pytest fixtures (`project_root`, `configs_dir`, `data_dir`, `sample_audio_dir`).

### Shell Scripts
Three executable scripts in `scripts/`:
- `download_data.sh` — Guides user through competition data download
- `download_weights.sh` — Documents HuggingFace Hub weight download commands
- `build_submission.sh` — Packages `submission/` directory into `submission.zip`

### Test Coverage
55 tests in `tests/test_project_structure.py` covering:
- Directory existence (10 dirs)
- Python package init files (4 files)
- requirements.txt content (13 core packages)
- Training config YAML structure (9 key checks)
- Shell script existence, executability, and shebangs (9 tests)
- .gitignore critical patterns (6 patterns)
- conftest.py existence and fixtures (2 tests)

## How
TDD approach: wrote all 55 tests first (RED — 30 failures, 9 errors), then created each file to pass them (GREEN — 55/55), then fixed one ruff lint issue (REFACTOR — clean).

## Connections
- **S1.2 (Audio Preprocessing)**: Will use `configs/training_config.yaml` for sample rate, duration filters, and silence threshold. Will add code to `src/preprocess.py`.
- **S1.3 (Text Normalization)**: Will add code to `src/utils.py`. Uses `jiwer` from requirements.txt.
- **S2.1 (Dataset Class)**: Will use the config for feature extraction parameters. Depends on `src/` being a package.
- **S2.2/S3.1 (Training Scripts)**: Will read `configs/training_config.yaml` for all hyperparameters.
- **S3.4 (Submission Packaging)**: Will use `scripts/build_submission.sh`.
