# Checklist S1.5 — Local Validation Framework

## Phase 1: Red (Write Tests)
- [x] Create `tests/test_evaluate.py`
- [x] Write split tests (child_id, stratification, ratio, deterministic, edge cases)
- [x] Write WER computation tests (basic, perfect, empty ref, empty hyp, normalization)
- [x] Write per-age WER tests
- [x] Write validation summary test
- [x] Verify all tests FAIL (no implementation yet)

## Phase 2: Green (Implement)
- [x] Create `src/evaluate.py`
- [x] Implement `split_by_child_id()` with stratification
- [x] Implement `compute_wer()` with normalization
- [x] Implement `compute_per_age_wer()`
- [x] Implement `validation_summary()`
- [x] All tests pass

## Phase 3: Refactor
- [x] Run `ruff` lint — clean
- [x] Review code for clarity
- [x] All tests still pass

## Done
- [x] All outcomes from spec.md met
