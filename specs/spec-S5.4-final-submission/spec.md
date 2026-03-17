# Spec S5.4 — Final Submission Package

## Overview
Build a comprehensive final submission validator that verifies the complete submission package is ready for DrivenData upload. This includes runtime environment validation, submission format compliance checking, end-to-end dry-run testing, and a pre-submission checklist runner.

## Depends On
- S5.1 (Post-processing corrections) — done
- S5.2 (Inference optimization) — done

## Location
- `src/final_submission.py` — Final submission validator and checker
- `scripts/build_submission.sh` — Updated build script with final checks

## Requirements

### R1: Submission Format Validator
- Validate `submission.jsonl` output format: each line must be valid JSON with `utterance_id` (string) and `orthographic_text` (string)
- Verify all utterance IDs from metadata are present in output (no missing IDs)
- Verify no duplicate utterance IDs in output
- Verify no extra utterance IDs beyond what's in metadata
- Validate that all `orthographic_text` values are strings (not null/int/etc.)

### R2: Runtime Environment Checker
- Check that all required Python packages are importable (torch, transformers, peft, librosa, soundfile)
- Check CUDA availability and report GPU info when available
- Check available disk space in submission directory
- Check available memory (RAM)
- Report Python version compatibility (must be 3.11+)

### R3: Size Budget Validator
- Compute total submission package size
- Warn if > 4 GB (upload may be slow)
- Error if > 10 GB (likely won't upload)
- Break down size by: code, model weights (per model), other files
- Verify model weights directories exist and contain expected files

### R4: Dry-Run End-to-End Test
- Run inference pipeline on a small sample (1-5 utterances) without real models
- Verify the pipeline produces valid JSONL output
- Verify output file is written to correct path
- Measure approximate timing for single-utterance inference

### R5: Pre-Submission Checklist
- All required files present (main.py, src/preprocess.py, src/utils.py)
- No __pycache__, .pyc, .DS_Store in submission zip
- model_weights/ directory structure is correct
- main.py has `if __name__ == "__main__"` entrypoint
- No hardcoded paths outside /code_execution/
- No network calls (no import requests, urllib, http.client at top level)
- EnglishTextNormalizer is applied to all predictions

## Outcomes
- `validate_submission_output()` validates JSONL format and completeness
- `check_runtime_environment()` reports environment readiness
- `validate_size_budget()` checks package size with detailed breakdown
- `run_dry_run()` does a mock end-to-end test
- `run_prechecks()` runs all pre-submission checks and returns pass/fail report
- All functions return structured results (dicts with status + details)

## TDD Notes
- Test JSONL validation with valid, invalid, and edge-case inputs
- Test environment checker with mocked sys/torch modules
- Test size validation with tmp_path fixtures
- Test pre-submission checklist with valid and broken submission dirs
- Test dry-run with fully mocked inference pipeline
- All tests use mocks — no real model loading or audio files
