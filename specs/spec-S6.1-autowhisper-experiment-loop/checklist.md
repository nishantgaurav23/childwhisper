# Checklist S6.1 — AutoWhisper: Autonomous AI Experiment Loop

## Phase 1: Fixed Evaluation Harness (`prepare.py`)
- [x] Create `src/autowhisper/__init__.py`
- [x] Write tests for `load_fast_eval_set` (count, determinism, no speaker leakage)
- [x] Write tests for `evaluate_wer` (perfect, all wrong, normalizer, empty)
- [x] Write tests for `evaluate_wer_by_age` (all buckets)
- [x] Implement `load_fast_eval_set` → tests pass
- [x] Implement `evaluate_wer` → tests pass
- [x] Implement `evaluate_wer_by_age` → tests pass
- [x] Define constants `TIME_BUDGET`, `EVAL_SAMPLES`
- [x] ruff passes

## Phase 2: Results Logger (`logger.py`)
- [x] Write tests for `init_log`, `append_result`, `load_results`
- [x] Write tests for `get_best_wer` (ignores crashes, includes baseline)
- [x] Write tests for `get_frontier` (monotonically decreasing)
- [x] Write tests for `print_summary` format
- [x] Write tests for `plot_progress` (mock matplotlib)
- [x] Implement `init_log` → tests pass
- [x] Implement `append_result` → tests pass
- [x] Implement `load_results` → tests pass
- [x] Implement `get_best_wer` → tests pass
- [x] Implement `get_frontier` → tests pass
- [x] Implement `print_summary` → tests pass
- [x] Implement `plot_progress` → tests pass
- [x] ruff passes

## Phase 3: Experiment Runner (`runner.py`)
- [x] Write tests for `init_run` (branch creation)
- [x] Write tests for `run_experiment` (parse WER, parse VRAM, timeout, crash)
- [x] Write tests for `evaluate_and_decide` (keep, discard, crash)
- [x] Write tests for `keep_experiment` and `revert_experiment` (git operations)
- [x] Write tests for `log_result` integration
- [x] Implement `init_run` → tests pass
- [x] Implement `run_experiment` → tests pass
- [x] Implement `evaluate_and_decide` → tests pass
- [x] Implement `keep_experiment` → tests pass
- [x] Implement `revert_experiment` → tests pass
- [x] Implement `log_result` → tests pass
- [x] Implement CLI entry point (`__main__.py`) for init/run/revert subcommands
- [x] ruff passes

## Phase 4: Mutable Training Script (`train.py`)
- [x] Create `src/autowhisper/train.py` with baseline Whisper-small fine-tune
- [x] Verify train.py prints `val_wer:` and `peak_vram_mb:` to stdout
- [x] Verify train.py respects TIME_BUDGET
- [x] Verify train.py uses `prepare.py` eval harness for evaluation
- [x] Test train.py runs on CPU with tiny synthetic data (smoke test) — structural only (requires GPU for full run)
- [x] ruff passes

## Phase 5: Agent Instructions (`program.md`)
- [x] Write setup section (branch creation, file reading, baseline run)
- [x] Write experiment loop specification (step-by-step)
- [x] Write rules section (what can/cannot be modified, output format)
- [x] Write search directions (initial list of ideas to explore)
- [x] Write results format specification
- [x] Create `configs/autowhisper/base_config.yaml`

## Phase 6: Kaggle Notebook (`05_autowhisper.ipynb`)
- [x] Create notebook structure (install, data download, git setup)
- [x] Implement scripted mode (pre-defined patch sequence, no API key)
- [x] Implement agent mode cell (Claude API integration, optional)
- [x] Add session time detection and graceful shutdown
- [x] Add HF Hub checkpoint upload for best model
- [x] Add results summary output cell
- [x] Test notebook cell structure

## Phase 7: Integration & Verification
- [x] End-to-end test: init → baseline → one experiment → keep/revert → log
- [x] Verify results.tsv schema compliance
- [x] Verify git history is clean (keeps committed, discards reverted)
- [x] Verify progress.png is generated correctly
- [x] All tests pass (`pytest tests/test_autowhisper_*.py`) — 47 passing
- [x] ruff passes on all `src/autowhisper/*.py`
- [x] Coverage >85% on autowhisper modules
