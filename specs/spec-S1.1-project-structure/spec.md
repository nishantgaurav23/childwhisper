# Spec S1.1 — Project Structure & Dependencies

## Summary
Establish the complete project directory structure, Python dependency manifest, training configuration, and shell utility scripts for the ChildWhisper project.

## Depends On
None — this is the foundational spec.

## Outcomes
1. `requirements.txt` with all pinned dependencies (free/open-source only)
2. `configs/training_config.yaml` with default training hyperparameters for both models
3. `src/__init__.py` making `src` a proper Python package
4. `tests/__init__.py` making `tests` a proper Python package
5. `submission/__init__.py` and `submission/utils/__init__.py`
6. `scripts/download_data.sh` — placeholder script to download competition data
7. `scripts/download_weights.sh` — placeholder script to pull weights from HF Hub
8. `scripts/build_submission.sh` — placeholder script to package submission.zip
9. All directories from the project structure in CLAUDE.md exist
10. `.gitignore` covers all required patterns (already exists, verify completeness)

## Acceptance Criteria
- `pip install -r requirements.txt` succeeds (dry-run check: valid package names + versions)
- `configs/training_config.yaml` is valid YAML with whisper-small and whisper-large-v3 sections
- All `__init__.py` files exist and are importable
- All shell scripts are executable (`chmod +x`)
- `ruff check` passes on all Python files
- pytest discovers and runs tests from the `tests/` directory

## TDD Notes
- Test that `requirements.txt` exists and contains expected core packages
- Test that `configs/training_config.yaml` is valid YAML with expected keys
- Test that `src`, `tests`, `submission`, `submission/utils` are importable packages
- Test that shell scripts exist and are executable
- Test that all expected directories exist
- Test that `.gitignore` contains critical patterns (data/, .env, __pycache__)

## Files to Create/Modify
- `requirements.txt` (create)
- `configs/training_config.yaml` (create)
- `src/__init__.py` (create)
- `tests/__init__.py` (create)
- `tests/conftest.py` (create — shared fixtures)
- `submission/__init__.py` (create)
- `submission/utils/__init__.py` (create)
- `scripts/download_data.sh` (create)
- `scripts/download_weights.sh` (create)
- `scripts/build_submission.sh` (create)
- `tests/test_project_structure.py` (create — structure validation tests)
