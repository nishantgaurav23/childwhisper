# Checklist — S2.4 Inference with Fine-Tuned Whisper-small

## Phase 1: Red (Write Tests)
- [x] Test `load_model` with custom model_name_or_path
- [x] Test `load_model` fallback when path doesn't exist
- [x] Test processor loaded from model dir when available
- [x] Test processor fallback to openai/whisper-small
- [x] Test `resolve_model_path` finds fine-tuned weights
- [x] Test `resolve_model_path` returns default when no fine-tuned weights
- [x] Test `resolve_model_path` returns default when dir empty (no config.json)
- [x] Test `resolve_model_path` accepts string paths
- [x] Test `resolve_model_path` custom default
- [x] Verify all existing S1.4 tests still pass

## Phase 2: Green (Implement)
- [x] Add `resolve_model_path()` function
- [x] Update `load_model()` signature with `model_name_or_path` param
- [x] Add processor path resolution logic with fallback
- [x] Update `main()` to use `resolve_model_path()` before loading
- [x] Add logging for model source selection

## Phase 3: Refactor
- [x] Run ruff, fix any lint issues — all clean
- [x] Run full test suite — 197 passed
- [x] Verify backward compatibility (zero-shot still works)
