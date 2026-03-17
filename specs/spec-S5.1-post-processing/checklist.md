# Checklist S5.1 — Post-Processing Corrections

## Phase 1: RED — Write Failing Tests
- [x] Test `postprocess_text` with repeated word collapse
- [x] Test `postprocess_text` with Whisper artifact removal
- [x] Test `postprocess_text` with whitespace cleanup
- [x] Test `postprocess_text` preserves valid child speech ("goed", "tooths")
- [x] Test `postprocess_text` with empty/None input
- [x] Test `postprocess_text` idempotency
- [x] Test `normalize_and_postprocess` combines both steps
- [x] Test ASR error correction dictionary replacements
- [x] Test integration in inference pipeline flow
- [x] All tests FAIL (RED) ✓

## Phase 2: GREEN — Implement
- [x] Implement `_remove_artifacts(text)` — strip hallucination tokens
- [x] Implement `_collapse_repeated_words(text)` — deduplicate consecutive repeats
- [x] Implement `_clean_whitespace(text)` — normalize spaces
- [x] Implement `postprocess_text(text)` — orchestrator
- [x] Implement `normalize_and_postprocess(text)` — combined pipeline
- [x] Update `submission/main.py` to use `normalize_and_postprocess`
- [x] All tests PASS (GREEN) ✓

## Phase 3: REFACTOR
- [x] Run ruff — no lint errors
- [x] All existing tests still pass (379 total)
- [x] Code is clean, no dead code
