# Checklist S1.1 — Project Structure & Dependencies

## Phase 1: Red (Write Tests)
- [x] Write `tests/test_project_structure.py` with all structure validation tests
- [x] Verify all tests fail (nothing implemented yet)

## Phase 2: Green (Implement)
- [x] Create `requirements.txt` with pinned dependencies
- [x] Create `configs/training_config.yaml` with default hyperparameters
- [x] Create `src/__init__.py`
- [x] Create `tests/__init__.py`
- [x] Create `tests/conftest.py` with shared fixtures
- [x] Create `submission/__init__.py`
- [x] Create `submission/utils/__init__.py`
- [x] Create `scripts/download_data.sh`
- [x] Create `scripts/download_weights.sh`
- [x] Create `scripts/build_submission.sh`
- [x] Make shell scripts executable
- [x] Verify all tests pass (55/55)

## Phase 3: Refactor
- [x] Run `ruff check` on all Python files — clean
- [x] Fixed ambiguous variable name lint error
- [x] Verify all tests still pass (55/55)

## Phase 4: Verify & Document
- [x] Run `/verify-spec` checks
- [x] Generate `explanation.md`
- [x] Update `roadmap.md` status to "done"
