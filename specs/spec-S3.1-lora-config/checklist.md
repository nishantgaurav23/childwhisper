# Checklist S3.1 — LoRA Configuration & Training Script

## Phase 1: Red (Write Failing Tests)
- [x] Write test_load_lora_config
- [x] Write test_parse_args_defaults
- [x] Write test_parse_args_dry_run
- [x] Write test_setup_lora_config
- [x] Write test_setup_model_spec_augment
- [x] Write test_setup_model_forced_decoder_ids_cleared
- [x] Write test_setup_training_args_lora_defaults
- [x] Write test_setup_training_args_dry_run
- [x] Write test_build_datasets
- [x] Write test_main_dry_run (→ test_computes_wer)
- [x] Verify all tests FAIL (no implementation yet)

## Phase 2: Green (Implement)
- [x] Implement load_training_config()
- [x] Implement parse_args()
- [x] Implement setup_model() with INT8 + LoRA
- [x] Implement setup_training_args()
- [x] Implement make_compute_metrics()
- [x] Implement build_datasets()
- [x] Implement main()
- [x] All tests pass (12/12)

## Phase 3: Refactor
- [x] Run ruff check — clean
- [x] All tests still pass
- [x] Code review for consistency with train_whisper_small.py

## Phase 4: Verify
- [x] Acceptance criteria met
- [x] Update roadmap status to done
