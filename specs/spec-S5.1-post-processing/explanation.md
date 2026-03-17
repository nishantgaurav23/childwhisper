# Explanation S5.1 — Post-Processing Corrections

## Why
Whisper models produce systematic artifacts on children's speech: hallucinated repeated words/phrases, music symbols, bracketed annotations like `[inaudible]`, and ellipsis sequences. These artifacts inflate WER by adding spurious insertions. A lightweight post-processing step after text normalization removes these patterns without any model retraining, providing a "free" WER improvement.

## What
Added two public functions to `src/utils.py`:

- **`postprocess_text(text)`** — Applies a three-stage correction pipeline:
  1. `_remove_artifacts()` — Regex-based removal of music symbols (♪), bracketed/parenthesized annotations, and ellipsis
  2. `_collapse_repeated_words()` — Collapses 3+ consecutive identical words or phrases to a single occurrence (handles multi-word n-gram repeats like "you know you know you know")
  3. `_clean_whitespace()` — Normalizes all whitespace to single spaces, strips edges

- **`normalize_and_postprocess(text)`** — Combines `normalize_text()` (EnglishTextNormalizer) with `postprocess_text()` in a single call for inference use

Updated `submission/main.py` to use `normalize_and_postprocess` instead of `normalize_text` in the inference loop.

## How
- **Artifact removal** uses a single compiled regex matching ♪, `[...]`, `(...)`, and `..` patterns
- **Repeated word collapse** iterates phrase lengths from longest viable (len/3) down to 1, counting consecutive identical n-grams. Only collapses when count >= 3 (preserves intentional doubles like "no no")
- **Whitespace cleanup** uses `\s+` replacement
- All operations are pure string processing — no external APIs, no model calls, < 1ms per utterance
- Child speech forms (goed, tooths, bestest, runned, mouses) are intentionally preserved — the system only removes ASR artifacts, not valid child language

## Connections
- **Depends on S3.3** (ensemble inference) — post-processing is applied to ensemble output
- **Feeds into S5.4** (final submission) — improves WER before final packaging
- **Complements S1.3** (text normalization) — runs after EnglishTextNormalizer as an additional cleanup layer
- **Independent of S5.2** (inference optimization) — post-processing is orthogonal to speed improvements

## Files Modified
- `src/utils.py` — Added post-processing functions
- `submission/main.py` — Switched to `normalize_and_postprocess`
- `tests/test_postprocess.py` — 32 new tests

## Test Coverage
- 32 tests across 7 test classes covering: artifact removal, repeated word collapse, ASR error corrections, child speech preservation, whitespace cleanup, edge cases, and integration with normalization
