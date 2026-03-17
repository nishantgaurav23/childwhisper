# Spec S1.3 — Text Normalization

## Overview
Implement a text normalization module in `src/utils.py` that wraps Whisper's `EnglishTextNormalizer` for consistent transcript normalization across the entire pipeline (preprocessing, training, inference, evaluation).

## Depends On
- S1.1 (Project structure & dependencies) — **done**

## Location
- `src/utils.py`

## Requirements

### R1: EnglishTextNormalizer Wrapper
- Provide a `normalize_text(text: str) -> str` function that applies Whisper's `EnglishTextNormalizer`
- The normalizer handles: lowercase, contraction expansion, number standardization, punctuation removal, whitespace normalization, diacritics removal
- Must be importable as `from src.utils import normalize_text`

### R2: Singleton/Cached Normalizer
- Provide a `get_normalizer()` function that returns a cached `EnglishTextNormalizer` instance
- Avoid re-instantiating the normalizer on every call (it has non-trivial init cost)

### R3: Edge Cases
- Empty string input → returns empty string
- Whitespace-only input → returns empty string
- None input → returns empty string (defensive)
- Already-normalized text → returns same text (idempotent for normalized input)

### R4: Batch Normalization
- Provide a `normalize_texts(texts: list[str]) -> list[str]` function for batch processing
- Applies `normalize_text` to each item in the list

## Outcomes
- `src/utils.py` contains `normalize_text`, `normalize_texts`, and `get_normalizer`
- All functions are tested with >80% coverage
- `ruff` passes with no warnings

## TDD Notes
- Test edge cases: empty, None, whitespace, punctuation-only
- Test that normalizer is idempotent on already-clean text
- Test known normalizer behaviors: lowercase, contraction expansion, number words
- Test batch function with mixed inputs
- Mock `EnglishTextNormalizer` in unit tests to avoid importing full transformers
