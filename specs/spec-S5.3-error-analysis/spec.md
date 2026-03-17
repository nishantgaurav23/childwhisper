# Spec S5.3 — Error Analysis Tooling

## Overview
Build error analysis utilities for detailed WER breakdown: substitution/insertion/deletion counts, hallucination detection, and per-utterance error reporting. Extends `src/evaluate.py` with granular error diagnostics.

## Depends On
- S1.5 (Local validation framework) — `compute_wer`, `validation_summary`, `split_by_child_id`
- S3.3 (Ensemble inference pipeline) — ensemble predictions to analyze

## Location
- `src/evaluate.py` — new functions added to existing module

## Outcomes

### O1: Substitution / Insertion / Deletion Breakdown
- `compute_error_breakdown(references, hypotheses)` returns dict with:
  - `substitutions`, `insertions`, `deletions`, `hits` (total counts)
  - `total_ref_words` (reference word count)
  - `wer`, `sub_rate`, `ins_rate`, `del_rate` (rates = count / total_ref_words)
- Normalizes both sides with `normalize_text` before computing
- Skips pairs where normalized reference is empty

### O2: Per-Age Error Breakdown
- `compute_per_age_error_breakdown(references, hypotheses, age_buckets)` returns dict mapping age_bucket to error breakdown dict from O1
- Reuses O1 per bucket

### O3: Hallucination Detection
- `detect_hallucinations(references, hypotheses, threshold=3.0)` returns list of dicts:
  - `index`, `reference`, `hypothesis`, `ref_word_count`, `hyp_word_count`, `ratio`
- Flags utterances where `hyp_word_count > threshold * ref_word_count`
- Also flags non-empty hypothesis when reference is empty (infinite ratio)
- Normalizes both sides before comparison

### O4: Error Analysis Summary
- `error_analysis_summary(references, hypotheses, age_buckets, hallucination_threshold=3.0)` returns dict:
  - `overall_breakdown`: from O1
  - `per_age_breakdown`: from O2
  - `hallucinations`: from O3
  - `hallucination_count`: int
  - `hallucination_rate`: float (count / total utterances)

### O5: Formatted Error Analysis Report
- `format_error_analysis_report(summary)` returns human-readable string
- Table with S/I/D counts and rates per age bucket
- Hallucination summary section
- Suitable for terminal output

## TDD Notes
- Test O1 with known reference/hypothesis pairs where S/I/D counts are manually verified
- Test O2 with multi-bucket data
- Test O3 with hallucination edge cases: empty ref, empty hyp, normal ratio, high ratio
- Test O4 integration
- Test O5 output format (contains expected sections/labels)
- All tests mock-free (pure computation on strings)

## Out of Scope
- Notebook visualization (01_eda.ipynb) — separate follow-up
- Per-session error analysis (would need session_id data)
- Phoneme-level error analysis
