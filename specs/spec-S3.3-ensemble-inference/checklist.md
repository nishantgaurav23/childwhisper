# Checklist S3.3 — Ensemble Inference Pipeline

## Phase 1: Red (Write Tests)
- [x] Write tests for `merge_predictions()` — all merge cases
- [x] Write tests for `check_time_budget()` — time decisions
- [x] Write tests for `load_large_model()` — PeftModel loading, fallback
- [x] Write tests for `run_ensemble_inference()` — full flow with mocks
- [x] Write tests for graceful degradation — Model B skipped
- [x] Write tests for backward compatibility — no adapter → small-only
- [x] Verify all tests FAIL (no implementation yet)

## Phase 2: Green (Implement)
- [x] Add `merge_predictions()` to `submission/main.py`
- [x] Add `check_time_budget()` to `submission/main.py`
- [x] Add `load_large_model()` to `submission/main.py`
- [x] Refactor `main()` to run ensemble pipeline via `run_ensemble_inference()`
- [x] Add constants: TIME_LIMIT_SEC, SAFETY_MARGIN_SEC, MODEL_B_CUTOFF_SEC
- [x] Add LoRA adapter path constant (LORA_ADAPTER_DIR)
- [x] All tests pass (24 new + 29 existing = 53 total)

## Phase 3: Refactor
- [x] Run ruff — lint clean
- [x] Ensure backward compatibility with S2.4 tests (29/29 pass)
- [x] All tests still pass
