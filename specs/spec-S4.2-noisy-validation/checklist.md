# Checklist S4.2 — Noisy Validation Set

## Phase 1: Red (Write Tests)
- [x] Create `tests/test_noisy_validation.py`
- [x] Write tests for `apply_noise_to_val` (7 tests)
- [x] Write tests for `combined_validation_summary` (4 tests)
- [x] Write tests for `format_validation_report` (3 tests)
- [x] Verify all tests fail (functions don't exist yet)

## Phase 2: Green (Implement)
- [x] Implement `apply_noise_to_val` in `src/evaluate.py`
- [x] Implement `combined_validation_summary` in `src/evaluate.py`
- [x] Implement `format_validation_report` in `src/evaluate.py`
- [x] All 14 tests pass

## Phase 3: Refactor
- [x] Run `ruff` — lint clean
- [x] Existing `test_evaluate.py` tests still pass (21 passing)
- [x] Code is clean, no duplication
- [x] All outcomes from spec met
