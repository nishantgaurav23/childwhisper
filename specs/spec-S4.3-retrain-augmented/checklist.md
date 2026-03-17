# Checklist — S4.3 Retrain with Augmented Data

## Phase 1: Tests (Red)
- [x] Write tests for augmentation CLI args in train_whisper_small.py
- [x] Write tests for augmentation CLI args in train_whisper_lora.py
- [x] Write tests for augmentation config loading from YAML
- [x] Write tests for augment_fn wiring into build_datasets
- [x] Write tests for error when only one noise dir provided
- [x] Write tests for hub model ID override
- [x] Write tests for notebook structure
- [x] Verify all tests FAIL (Red) — 21 failed

## Phase 2: Implementation (Green)
- [x] Add augmentation section to training_config.yaml
- [x] Add --noise-dir and --realclass-dir args to train_whisper_small.py
- [x] Add --noise-dir and --realclass-dir args to train_whisper_lora.py
- [x] Add create_augment_fn() to both scripts
- [x] Wire augment_fn into main() in both scripts
- [x] Add --hub-model-id CLI arg to both scripts
- [x] Create notebooks/04_augmented.ipynb
- [x] Verify all tests PASS (Green) — 26 passed

## Phase 3: Refactor
- [x] Run ruff, fix lint issues (removed unused imports)
- [x] Verify all 347 tests still pass
- [x] No code duplication to address (create_augment_fn is intentionally duplicated for script independence)
