# Explanation — S1.5 Local Validation Framework

## Why
Without a proper validation framework, there's no way to measure model quality locally before submitting to DrivenData. The competition allows only 3 submissions per rolling 7-day window, making local validation critical. Speaker leakage (same child in train and val) would give inflated metrics, so splitting by `child_id` is essential.

## What Was Built

### `src/evaluate.py` — 4 public functions:

1. **`split_by_child_id(metadata, val_ratio, seed)`** — Splits metadata into train/val by unique `child_id` with stratification by `age_bucket`. Uses `random.Random(seed)` for determinism. Handles edge cases: empty input, single child, too-few-per-bucket fallback.

2. **`compute_wer(references, hypotheses)`** — Computes Word Error Rate using `jiwer.wer()` after applying Whisper's `EnglishTextNormalizer` to both sides. Skips pairs where the reference is empty (after normalization). Returns 0.0 when no valid pairs exist.

3. **`compute_per_age_wer(references, hypotheses, age_buckets)`** — Groups utterances by `age_bucket` and computes WER separately for each group. Returns a dict mapping bucket names to WER values.

4. **`validation_summary(references, hypotheses, age_buckets)`** — Returns a comprehensive dict with `overall_wer`, `per_age_wer`, `num_utterances`, `num_empty_refs_skipped`, and `num_empty_preds`.

### `tests/test_evaluate.py` — 21 tests across 4 test classes:
- `TestSplitByChildId` (8 tests): leakage prevention, ratio, stratification, determinism, edge cases
- `TestComputeWer` (7 tests): basic WER, perfect match, empty handling, normalization
- `TestPerAgeWer` (3 tests): per-bucket breakdown, single bucket, unknown bucket
- `TestValidationSummary` (3 tests): structure, counts, WER values

## How

### Stratified Split Algorithm
1. Group `child_id`s by their `age_bucket`
2. For each bucket, shuffle children (deterministically via seed) and pick `round(n * val_ratio)` for validation
3. If rounding results in 0 val children across all buckets, fall back to global sampling
4. Partition original metadata by child membership

### WER with Normalization
- Both reference and hypothesis are passed through `normalize_text()` (Whisper's `EnglishTextNormalizer`) before comparison
- Empty-after-normalization references are excluded entirely (they contribute no signal)
- Empty hypotheses against non-empty references count as 100% error (all words are deletions)
- Uses `jiwer.wer()` which concatenates all utterances for corpus-level WER

## Connections
- **Upstream**: Uses `normalize_text()` from S1.3 (`src/utils.py`)
- **Downstream**: S2.2 (training script) will use `split_by_child_id` for train/val splits. S4.2 (noisy validation) will extend this with synthetic noise. S5.3 (error analysis) will build on per-age WER.
- **Bug fix**: Updated `EnglishTextNormalizer()` → `EnglishTextNormalizer({})` in `src/utils.py` to match current transformers API (required `english_spelling_mapping` argument).
