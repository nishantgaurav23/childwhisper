# Checklist S3.2 — Kaggle LoRA Training Notebook

## Phase 1: Red (Write Tests First)
- [x] Write `tests/test_kaggle_utils_lora.py`
  - [x] `TestGetKagglePathsLora` — correct LoRA output paths on Kaggle
  - [x] `TestGetLocalPathsLora` — correct LoRA output paths locally
  - [x] `TestGetPathsLora` — auto-detect routing for LoRA
  - [x] `TestGetLoraTrainingArgs` — CLI args for train_whisper_lora.main()
  - [x] `TestGetLoraTrainingArgsDryRun` — dry-run flag passthrough
  - [x] `TestGetLoraTrainingArgsNoToken` — force --no-push-to-hub without HF_TOKEN
  - [x] `TestCheckGpuMemory` — GPU memory check with mocked torch.cuda
  - [x] `TestNotebookStructure` — validate notebook cell structure
- [x] All tests fail (Red)

## Phase 2: Green (Implement)
- [x] Extend `src/kaggle_utils.py` with LoRA functions
  - [x] `get_kaggle_paths_lora()`
  - [x] `get_local_paths_lora()`
  - [x] `get_paths_lora()`
  - [x] `get_lora_training_args()`
  - [x] `check_gpu_memory()`
- [x] Create `notebooks/03_train_lora.ipynb` with 9 cells
- [x] All tests pass (Green)

## Phase 3: Refactor
- [x] Run `ruff check` on modified files
- [x] Run `ruff format` on modified files
- [x] All tests still pass (42/42)
- [x] Coverage 98% on kaggle_utils.py
