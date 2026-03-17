# Checklist S2.2 — Whisper-small Training Script

## Phase 1: Red (Write Failing Tests)
- [x] test_load_training_config — loads YAML and merges common + whisper_small sections
- [x] test_parse_args — CLI argument parsing with defaults and overrides
- [x] test_setup_model — model has SpecAugment enabled, gradient ckpt on, correct dtype
- [x] test_setup_training_args — Seq2SeqTrainingArguments match config values
- [x] test_compute_metrics — WER metric function works with tokenizer decode
- [x] test_build_datasets — creates train/val datasets with correct split
- [x] test_dry_run — covered by dry_run_overrides test
- [x] test_hub_push_disabled — no hub interaction when push_to_hub=False
- [x] All tests fail (RED)

## Phase 2: Green (Implement to Pass)
- [x] Implement load_training_config()
- [x] Implement parse_args()
- [x] Implement setup_model()
- [x] Implement setup_training_args()
- [x] Implement compute_metrics()
- [x] Implement build_datasets()
- [x] Implement main() with dry-run support
- [x] All tests pass (GREEN)

## Phase 3: Refactor
- [x] Run ruff, fix any lint issues
- [x] Ensure all tests still pass (164/164)
- [x] Check coverage >80%

## Phase 4: Verify
- [x] Code exists at src/train_whisper_small.py
- [x] Tests exist at tests/test_train_whisper_small.py
- [x] All tests pass (17/17)
- [x] Lint clean
- [x] Outcomes from spec.md met
