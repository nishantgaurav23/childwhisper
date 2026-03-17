# Explanation — S5.4 Final Submission Package

## Why
The competition requires uploading a `submission.zip` to DrivenData with strict format requirements: valid JSONL output, offline execution on A100, specific directory structure, and size limits. Without automated validation, it's easy to submit a broken package (missing files, hardcoded paths, network imports, wrong output format) and waste one of the limited 3-per-week submission slots.

## What
A comprehensive pre-submission validation module (`src/final_submission.py`) with five functions:

1. **`validate_submission_output()`** — Validates the `submission.jsonl` file: checks JSON parsing, required fields (`utterance_id`, `orthographic_text`), type correctness, completeness (no missing/duplicate/extra IDs).

2. **`check_runtime_environment()`** — Reports Python version, checks importability of required packages (torch, transformers, peft, librosa, soundfile), and detects the compute device (cuda/mps/cpu).

3. **`validate_size_budget()`** — Delegates to `compute_size_budget()` from S3.4, adds pass/fail based on the 10 GB hard limit, returns detailed code vs. weights breakdown.

4. **`run_dry_run()`** — Reads metadata and writes a dummy `submission.jsonl` with empty predictions. Validates the pipeline end-to-end without loading real models. Reports timing.

5. **`run_prechecks()`** — Runs a checklist of structural checks:
   - Required files present (main.py, src/preprocess.py, src/utils.py)
   - `__main__` entrypoint exists in main.py
   - No hardcoded paths outside `/code_execution/`
   - No network library imports (requests, urllib, etc.)
   - No `__pycache__` directories
   - `model_weights/` directory exists

## How
- All validators return structured dicts with `valid`/`passed` booleans and detailed error/failure lists for programmatic consumption.
- Hardcoded path detection uses a regex that finds string literals containing absolute paths not under `/code_execution/`.
- Network import detection scans for `import requests`, `from urllib`, etc. patterns.
- The dry-run writes empty predictions to test the pipeline plumbing without any model loading.
- Reuses `compute_size_budget()` from `src/submission_builder.py` (S3.4) to avoid duplication.

## Connections
- **S3.4** (submission builder): Reuses `compute_size_budget()` for size validation.
- **S5.1** (post-processing): The submission output validator ensures normalized text is properly formatted.
- **S5.2** (faster inference): The dry-run validates the pipeline structure that S5.2 optimized.
- **S1.4** (inference pipeline): The prechecks validate the main.py entrypoint created in S1.4 and extended in S2.4/S3.3.

## Test Coverage
- 34 tests covering all 5 public functions
- Edge cases: missing files, invalid JSON, duplicate IDs, empty predictions, hardcoded paths, network imports, nonexistent directories
