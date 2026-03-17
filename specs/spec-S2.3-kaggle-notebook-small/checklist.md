# Checklist S2.3 — Kaggle Training Notebook (Whisper-small)

## Phase 1: Red — Write Tests
- [x] Create `tests/test_kaggle_utils.py`
- [x] Test `is_kaggle()` detection (Kaggle env vs local)
- [x] Test `get_kaggle_paths()` returns correct paths
- [x] Test `get_local_paths()` returns correct paths
- [x] Test `get_paths()` auto-detection
- [x] Test `setup_hub_auth()` with mock HF login
- [x] Test `get_latest_checkpoint()` with mock Hub API
- [x] Test `download_checkpoint()` with mock Hub API
- [x] Test `get_kaggle_training_args()` builds correct CLI args
- [x] Test `get_kaggle_training_args()` with resume checkpoint
- [x] Test `get_kaggle_training_args()` without HF token
- [x] Test `verify_kaggle_data()` with valid data
- [x] Test `verify_kaggle_data()` with missing files
- [x] All tests fail (Red)

## Phase 2: Green — Implement
- [x] Create `src/kaggle_utils.py` with all functions
- [x] Implement `is_kaggle()`
- [x] Implement `get_kaggle_paths()` / `get_local_paths()` / `get_paths()`
- [x] Implement `setup_hub_auth()`
- [x] Implement `get_latest_checkpoint()` / `download_checkpoint()`
- [x] Implement `get_kaggle_training_args()`
- [x] Implement `verify_kaggle_data()`
- [x] All tests pass (Green)
- [x] Create `notebooks/02_train_small.ipynb` with all required cells

## Phase 3: Refactor
- [x] Run ruff, fix any issues
- [x] All tests still pass (187/187)
- [x] Verify notebook structure (9 sections)
