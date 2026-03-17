# Checklist S2.1 — PyTorch Dataset for Whisper

## Phase 1: Red (Write Tests)
- [x] test_dataset_init
- [x] test_dataset_getitem
- [x] test_dataset_silence_handling
- [x] test_dataset_missing_audio
- [x] test_dataset_augmentation
- [x] test_dataset_duration_filter
- [x] test_dataset_with_processor
- [x] test_collator_padding
- [x] test_collator_batch_shapes
- [x] test_train_val_split_no_leakage
- [x] test_train_val_split_stratification
- [x] test_train_val_split_deterministic
- [x] test_train_val_split_coverage
- [x] test_train_val_split_ratio
- [x] All tests FAIL (Red)

## Phase 2: Green (Implement)
- [x] WhisperDataset.__init__ — load metadata, filter, init processor
- [x] WhisperDataset.__len__
- [x] WhisperDataset.__getitem__ — audio loading, feature extraction, tokenization
- [x] WhisperDataCollator — padding for features and labels
- [x] create_train_val_split — child_id split with age_bucket stratification
- [x] All tests PASS (Green)

## Phase 3: Refactor
- [x] Run ruff — lint clean
- [x] All 133 existing tests still pass
- [x] All 14 new tests pass (147 total)
- [x] Code review: no hardcoded paths, no CUDA-only code

## Outcomes
- [x] `src/dataset.py` exists with WhisperDataset, WhisperDataCollator, create_train_val_split
- [x] `tests/test_dataset.py` has 14 comprehensive tests
- [x] Ready for S2.2 (training script) to consume
