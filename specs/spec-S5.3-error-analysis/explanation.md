# Explanation S5.3 — Error Analysis Tooling

## Why
The existing validation framework (S1.5) computes overall and per-age WER but doesn't break errors down into substitutions, insertions, and deletions. Without this breakdown, it's impossible to know whether the model is hallucinating extra words (insertions), missing words (deletions), or confusing words (substitutions). Hallucination detection is critical for Whisper models, which are known to repeat or fabricate text on short/silent clips. This tooling enables targeted improvements in Phase 5.

## What
Five new functions added to `src/evaluate.py`:

1. **`compute_error_breakdown`** — Returns S/I/D/hit counts and rates using `jiwer.process_words`. Normalizes both sides, skips empty references.
2. **`compute_per_age_error_breakdown`** — Groups by age bucket and calls `compute_error_breakdown` per bucket. Reveals which age groups have the most substitutions vs deletions.
3. **`detect_hallucinations`** — Flags utterances where hypothesis word count exceeds `threshold * reference word count` (default 3x). Also catches non-empty predictions on empty references.
4. **`error_analysis_summary`** — Aggregates all three above into a single dict with hallucination count and rate.
5. **`format_error_analysis_report`** — Renders the summary as a terminal-friendly table with S/I/D breakdown per age bucket and hallucination examples.

## How
- Uses `jiwer.process_words()` which returns a `WordOutput` object with `.substitutions`, `.insertions`, `.deletions`, `.hits` attributes — these are the alignment-based counts from the standard Levenshtein edit distance algorithm.
- Hallucination detection is a simple word-count ratio heuristic (not model-confidence-based) — fast, no model dependency, catches the most common Whisper failure mode.
- All functions normalize text with `normalize_text()` (Whisper EnglishTextNormalizer) before analysis, matching the competition evaluation.

## Connections
- **Depends on S1.5**: Uses `normalize_text` from `src/utils.py` and extends the evaluation module.
- **Depends on S3.3**: The error analysis is designed to analyze ensemble predictions.
- **Feeds into S5.4**: Final submission packaging can use the error analysis to validate prediction quality before submission.
- **Design.md Section 10**: Implements the "Error Analysis Checklist" from the monitoring section — S/I/D fractions, per-age analysis, hallucination detection.
