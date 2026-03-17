# Checklist -- Spec S5.5: Hyperparameter Sweep with Kaggle API Integration

## Phase 1: Setup & Dependencies
- [x] Verify all dependency specs are "done" (S2.2, S3.1, S4.1)
- [x] Create target files/directories (`src/sweep.py`, `src/kaggle_runner.py`, `scripts/`, `configs/sweep_configs/`)
- [x] Verify `kaggle` CLI is installed (`pip install kaggle`)
- [x] Create `configs/sweep_space.yaml` with default search space

## Phase 2: Tests First (TDD Red Phase)
- [x] Create test file: `tests/test_sweep.py`
  - [x] `test_generate_grid_configs`
  - [x] `test_generate_random_configs`
  - [x] `test_generate_configs_empty_space`
  - [x] `test_generate_configs_lora_params_ignored_for_small`
  - [x] `test_notebook_generator_creates_valid_ipynb`
  - [x] `test_notebook_generator_embeds_config`
  - [x] `test_aggregate_results_sorts_by_wer`
  - [x] `test_aggregate_results_handles_failures`
  - [x] `test_aggregate_produces_best_config`
- [x] Create test file: `tests/test_kaggle_runner.py`
  - [x] `test_kaggle_push_calls_cli`
  - [x] `test_kaggle_status_parses_output`
  - [x] `test_kaggle_pull_downloads_files`
  - [x] `test_kaggle_missing_credentials`
  - [x] `test_kernel_metadata_format`
- [x] Run tests — confirm all FAIL (Red) — 22 failed

## Phase 3: Implementation (TDD Green Phase)
- [x] Implement FR-1 (sweep config generator) in `src/sweep.py`
- [x] Implement FR-2 (notebook generator) in `src/sweep.py`
- [x] Implement FR-3 (Kaggle API wrapper) in `src/kaggle_runner.py`
- [x] Implement FR-4 (results aggregator) in `src/sweep.py`
- [x] Implement FR-5 (shell scripts) — all 5 scripts created
- [x] Create `configs/sweep_space.yaml` with default search space
- [x] All tests pass (Green) — 22 passed

## Phase 4: Refactor
- [x] Clean up code, remove duplication
- [x] Run ruff — fixed 4 lint issues, formatted 4 files
- [x] Run full test suite — all 22 pass

## Phase 5: Integration & Verification
- [x] Dry-run: generate configs locally — verified 3 configs generated
- [x] Dry-run: generate a notebook — verified valid .ipynb with 6 cells
- [x] Verify shell scripts are executable
- [x] No hardcoded secrets (Kaggle credentials via ~/.kaggle/kaggle.json only)
- [x] Update roadmap.md status: "spec-written" -> "done"
