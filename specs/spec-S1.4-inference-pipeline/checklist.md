# Checklist S1.4 — Zero-Shot Inference Pipeline

## Phase 1: Red (Write Failing Tests)
- [x] Test `get_device` returns valid device string
- [x] Test `load_metadata` reads JSONL and returns list of dicts
- [x] Test `load_metadata` handles empty/missing file
- [x] Test `load_model` returns model and processor (mocked)
- [x] Test `transcribe_batch` produces text outputs (mocked model)
- [x] Test `transcribe_batch` handles empty batch
- [x] Test `run_inference` processes all utterances and returns predictions dict
- [x] Test `run_inference` sorts utterances by duration (longest first)
- [x] Test `run_inference` returns empty string for silent audio
- [x] Test `run_inference` applies text normalization
- [x] Test `write_submission` writes valid JSONL with correct schema
- [x] Test `write_submission` includes all utterance IDs
- [x] Test `write_submission` handles missing predictions

## Phase 2: Green (Implement)
- [x] Implement `get_device`
- [x] Implement `load_metadata`
- [x] Implement `load_model`
- [x] Implement `transcribe_batch`
- [x] Implement `run_inference`
- [x] Implement `write_submission`
- [x] Implement `main`
- [x] All 19 tests pass

## Phase 3: Refactor
- [x] Run ruff, fix lint issues — all checks passed
- [x] All 57 tests still pass (19 inference + 23 preprocess + 15 utils)
- [x] Code review for clarity and correctness
