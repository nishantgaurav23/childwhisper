# Spec S3.4 — Submission Zip Builder

## Overview
Build a robust submission packaging script and Python module that bundles model weights + inference code into a `submission.zip` for DrivenData upload. Includes size validation, structure verification, manifest generation, and a dry-run test mode.

## Depends On
- **S3.3**: Ensemble inference pipeline (provides final `submission/main.py` with two-model ensemble)

## Location
- `scripts/build_submission.sh` (upgrade existing shell script)
- `src/submission_builder.py` (new Python module for validation logic)
- `tests/test_submission_builder.py` (tests)

## Outcomes
1. Python module `src/submission_builder.py` with functions to:
   - Validate submission directory structure (main.py at root, model_weights present)
   - Check required files exist (main.py, src/ modules, model weights dirs)
   - Compute and report zip size with warnings for large packages
   - Generate a manifest of included files with sizes
   - Build the zip file excluding __pycache__, .pyc, .DS_Store, .git
2. Enhanced `scripts/build_submission.sh` that:
   - Calls the Python validation before building
   - Reports file count and total size
   - Warns if zip exceeds 4 GB (DrivenData practical limit)
   - Supports `--dry-run` flag (validate without building)
   - Supports `--output` flag for custom output path
3. Validates that submission structure matches competition requirements:
   - `main.py` at submission root
   - All `src/` dependencies are bundled (preprocess.py, utils.py)
   - `model_weights/` directory exists (may be empty for test runs)
4. Size budget tracking: reports per-component sizes (code vs weights)

## Design Details

### Submission Structure (inside zip)
```
submission.zip/
├── main.py                    # Entrypoint
├── src/
│   ├── __init__.py
│   ├── preprocess.py          # Audio preprocessing
│   └── utils.py               # Text normalization
├── model_weights/
│   ├── lora_large_v3/         # LoRA adapter (~63 MB)
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── whisper_small_ft/      # Full fine-tuned small (~500 MB)
│       ├── config.json
│       ├── model.safetensors
│       └── ...
└── utils/
    └── __init__.py
```

### Key Functions (src/submission_builder.py)
- `validate_submission_dir(submission_dir) -> list[str]`: Return list of errors (empty = valid)
- `get_submission_manifest(submission_dir) -> list[dict]`: File paths + sizes
- `compute_size_budget(submission_dir) -> dict`: Code size, weights size, total
- `build_submission_zip(submission_dir, output_path, dry_run=False) -> Path`: Build the zip
- `get_excludes() -> list[str]`: Patterns to exclude from zip

### Size Thresholds
- Warning at > 4 GB (DrivenData upload may be slow/fail)
- Error at > 10 GB (definitely too large)

## TDD Notes

### Test Categories
1. **validate_submission_dir**: Test valid dir, missing main.py, missing src/, missing model_weights
2. **get_submission_manifest**: Test file listing with sizes
3. **compute_size_budget**: Test code vs weights breakdown
4. **build_submission_zip**: Test zip creation, exclusion patterns, dry-run mode
5. **get_excludes**: Verify exclusion patterns include __pycache__, .pyc, .DS_Store
6. **Shell script**: Test --dry-run flag, --output flag, error handling

### What to Mock
- No external services needed — all filesystem operations
- Use tmp_path for test directories with realistic structure
