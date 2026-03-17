# Explanation S3.4 — Submission Zip Builder

## Why
The competition requires uploading a `submission.zip` to DrivenData containing `main.py`, all source dependencies, and model weights. Without a reliable build pipeline, it's easy to forget files, include junk (`__pycache__`, `.DS_Store`), or exceed size limits. This spec creates a validated, repeatable packaging workflow.

## What
Two components were built:

1. **`src/submission_builder.py`** — Python module with 5 public functions:
   - `validate_submission_dir()` — checks main.py, src/ modules, model_weights/ exist
   - `get_submission_manifest()` — lists all files with sizes, excluding junk patterns
   - `compute_size_budget()` — breaks down code vs weights size, warns at 4 GB / errors at 10 GB
   - `build_submission_zip()` — creates the zip with validation, supports dry-run mode
   - `get_excludes()` — returns the exclusion pattern list

2. **`scripts/build_submission.sh`** — enhanced shell script that:
   - Copies `src/preprocess.py` and `src/utils.py` into `submission/src/` (needed at runtime)
   - Calls the Python validator
   - Reports size budget (code vs weights breakdown)
   - Supports `--dry-run` (validate + show manifest) and `--output` flags

## How
- **Validation**: Checks for required files against a `REQUIRED_FILES` list. Missing `model_weights/` is a warning (not blocking), since local test runs may not have weights.
- **Exclusion**: File paths are checked against `EXCLUDE_PATTERNS` by matching against path parts and suffixes, catching nested `__pycache__`, `.pyc` files, and `.DS_Store`.
- **Zip creation**: Uses Python's `zipfile.ZipFile` with `ZIP_DEFLATED` compression. Files are written with paths relative to the submission directory root.
- **Size budget**: Files under `model_weights/` count as weights; everything else counts as code. Thresholds at 4 GB (warning) and 10 GB (error).
- **Shell script**: Bundles `src/` into `submission/` before validation because `main.py` imports from `src/` via `sys.path` manipulation.

## Connections
- **S3.3** (Ensemble inference): Provides the `main.py` that this spec packages
- **S5.4** (Final submission): Will use this builder for the competition submission
- **S2.4** (Fine-tuned inference): `main.py` loads weights from `model_weights/` paths validated here
- **scripts/download_weights.sh**: Downloads weights into the `model_weights/` structure this builder expects

## Test Coverage
40 tests covering all public functions:
- 8 tests for `validate_submission_dir` (valid, missing files, nonexistent dir, string paths)
- 8 tests for `get_submission_manifest` (file listing, exclusions, weight files)
- 8 tests for `compute_size_budget` (breakdown, warnings, human-readable output)
- 5 tests for `get_excludes` (all patterns present)
- 11 tests for `build_submission_zip` (creation, contents, exclusions, dry-run, error handling)
