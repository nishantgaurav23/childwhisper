# Spec S1.5 — Local Validation Framework

## Overview
Build a local validation framework in `src/evaluate.py` that supports:
1. Splitting data by `child_id` (no speaker leakage) with age-bucket stratification
2. Computing WER using `jiwer` with Whisper `EnglishTextNormalizer` applied
3. Per-age-bucket WER breakdown
4. Overall validation summary reporting

## Depends On
- **S1.2** (Audio preprocessing) — for `load_metadata()` from `src/preprocess.py`
- **S1.3** (Text normalization) — for `normalize_text()` from `src/utils.py`

## Location
- `src/evaluate.py` — main module

## Functional Requirements

### FR-1: Child-ID-Based Train/Val Split
- Split metadata by unique `child_id` values (not individual utterances)
- Default 90/10 split ratio (train/val)
- **No speaker leakage**: all utterances from a given child must be in the same split
- Stratify by `age_bucket` so each split has proportional representation
- Deterministic split via a configurable random seed (default: 42)
- Return two lists of metadata dicts (train_meta, val_meta)

### FR-2: WER Computation
- Compute WER using `jiwer.wer()` on normalized text (both reference and hypothesis)
- Apply `normalize_text()` from `src/utils.py` to both sides before comparison
- Handle edge cases:
  - Empty reference → skip (do not count)
  - Empty hypothesis → count as 100% error for that utterance
  - Both empty → skip
- Return overall WER as a float

### FR-3: Per-Age-Bucket WER
- Group utterances by `age_bucket` field
- Compute WER separately for each bucket
- Return dict mapping `age_bucket → WER`
- Handle missing/unknown age buckets gracefully

### FR-4: Validation Summary
- Return a structured dict with:
  - `overall_wer`: float
  - `per_age_wer`: dict[str, float]
  - `num_utterances`: int
  - `num_empty_refs_skipped`: int
  - `num_empty_preds`: int

## Non-Functional Requirements
- No dependency on audio files or model inference — operates on text pairs
- Must work on MacBook (CPU only, no GPU needed)
- All functions should be importable and testable independently

## TDD Notes

### Tests to Write First
1. **test_split_by_child_id** — verify no child_id appears in both train and val
2. **test_split_stratified** — verify age buckets are represented in both splits
3. **test_split_ratio** — verify ~90/10 ratio of child_ids
4. **test_split_deterministic** — same seed → same split
5. **test_compute_wer_basic** — simple WER calculation
6. **test_compute_wer_perfect** — identical refs/hyps → WER 0.0
7. **test_compute_wer_empty_ref_skipped** — empty references are excluded
8. **test_compute_wer_empty_hyp** — empty hypothesis counts as error
9. **test_compute_wer_normalization** — verifies normalizer is applied
10. **test_per_age_wer** — correct per-bucket breakdown
11. **test_validation_summary_structure** — all expected keys present
12. **test_split_single_child** — edge case: only 1 unique child_id
13. **test_split_empty_input** — edge case: empty metadata list

## Outcomes
- `src/evaluate.py` exists with all functions
- `tests/test_evaluate.py` has ≥13 tests, all passing
- `ruff` lint passes on `src/evaluate.py`
- Functions can be imported and used standalone (no side effects on import)
