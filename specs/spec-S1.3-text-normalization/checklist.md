# Checklist — S1.3 Text Normalization

## Phase 1: Red (Write Tests)
- [x] Create `tests/test_utils.py`
- [x] Test `normalize_text` with normal input
- [x] Test `normalize_text` edge cases (empty, None, whitespace)
- [x] Test normalizer behaviors (lowercase, contractions, numbers, punctuation)
- [x] Test `get_normalizer` returns cached instance
- [x] Test `normalize_texts` batch function
- [x] Verify all tests FAIL (no implementation yet)

## Phase 2: Green (Implement)
- [x] Create `src/utils.py` with `get_normalizer()`
- [x] Implement `normalize_text(text: str) -> str`
- [x] Implement `normalize_texts(texts: list[str]) -> list[str]`
- [x] All tests pass

## Phase 3: Refactor
- [x] Run `ruff check src/utils.py tests/test_utils.py`
- [x] Fix any lint issues
- [x] All tests still pass
