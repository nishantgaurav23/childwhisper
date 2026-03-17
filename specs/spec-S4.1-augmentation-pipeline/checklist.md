# Checklist — S4.1 Noise Augmentation Pipeline

## Phase 1: Red (Write Tests)
- [x] Create `tests/test_augment.py`
- [x] Test `create_augmentation` returns callable
- [x] Test output shape, dtype, length match input
- [x] Test `FileNotFoundError` for missing directories
- [x] Test `create_noise_only_augmentation` returns callable
- [x] Test augmented audio differs from input
- [x] Test factory accepts str and Path
- [x] Test integration with WhisperDataset augment_fn signature
- [x] All tests fail (module doesn't exist yet)

## Phase 2: Green (Implement)
- [x] Create `src/augment.py`
- [x] Implement `create_augmentation(noise_dir, realclass_dir, ...)`
- [x] Implement `create_noise_only_augmentation(noise_dir, ...)`
- [x] All tests pass

## Phase 3: Refactor
- [x] Run `ruff` — lint clean
- [x] Review for simplicity
- [x] All tests still pass (307/307)

## Completion
- [x] Update roadmap.md status to "done"
- [x] Generate explanation.md
