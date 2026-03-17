# Checklist S5.3 — Error Analysis Tooling

## Phase 1: Tests (Red)
- [x] Write test_error_analysis.py with tests for:
  - [x] compute_error_breakdown — known S/I/D counts
  - [x] compute_error_breakdown — empty refs skipped
  - [x] compute_error_breakdown — all correct (0 WER)
  - [x] compute_per_age_error_breakdown — multi-bucket
  - [x] detect_hallucinations — normal ratio (no flag)
  - [x] detect_hallucinations — high ratio (flagged)
  - [x] detect_hallucinations — empty ref with non-empty hyp
  - [x] detect_hallucinations — empty hyp (not flagged)
  - [x] error_analysis_summary — integration
  - [x] format_error_analysis_report — contains expected sections
- [x] All tests fail (Red)

## Phase 2: Implementation (Green)
- [x] Implement compute_error_breakdown in src/evaluate.py
- [x] Implement compute_per_age_error_breakdown in src/evaluate.py
- [x] Implement detect_hallucinations in src/evaluate.py
- [x] Implement error_analysis_summary in src/evaluate.py
- [x] Implement format_error_analysis_report in src/evaluate.py
- [x] All tests pass (Green)

## Phase 3: Refactor
- [x] Run ruff, fix any lint issues
- [x] Ensure existing evaluate.py tests still pass
- [x] Review for code quality
