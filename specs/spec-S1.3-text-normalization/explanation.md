# Explanation — S1.3 Text Normalization

## Why
The Pasketti competition scores predictions using WER computed after Whisper's `EnglishTextNormalizer`. All model outputs must pass through this normalizer to ensure consistent comparison with reference transcripts. Without centralized normalization, different pipeline stages (training label prep, inference output, evaluation) could produce inconsistent text, causing WER measurement drift.

## What
Three functions in `src/utils.py`:

- **`get_normalizer()`** — Returns a cached `EnglishTextNormalizer` singleton. Lazy import avoids requiring `transformers` at module import time, which keeps tests fast and the module importable without heavy dependencies.
- **`normalize_text(text)`** — Normalizes a single string: lowercase, contraction expansion, punctuation removal, whitespace normalization, diacritics removal. Defensively handles `None`, empty, and whitespace-only inputs by returning `""`.
- **`normalize_texts(texts)`** — Batch wrapper that applies `normalize_text` to each item in a list.

## How
- **Lazy import**: The `EnglishTextNormalizer` class is imported inside `get_normalizer()` rather than at module top level. This allows `src/utils.py` to be imported in test environments without `transformers` installed, and avoids the normalizer's non-trivial initialization cost until first use.
- **Singleton pattern**: A module-level `_normalizer` variable caches the instance. Subsequent calls to `get_normalizer()` return the same object.
- **Edge case guard**: `normalize_text` checks for `None` and whitespace-only input before invoking the normalizer, preventing unnecessary processing and potential errors.
- **Test mocking**: Tests use `sys.modules` patching to simulate the `transformers` module hierarchy for `get_normalizer` tests, and patch `get_normalizer` directly for all other tests. This means the full test suite runs in <0.1s with no external dependencies.

## Connections
- **S1.2 (Audio Preprocessing)**: `preprocess.py` will use `normalize_text` to clean reference transcripts during data preparation.
- **S1.4 (Inference Pipeline)**: `submission/main.py` will call `normalize_text` on all model predictions before writing output.
- **S1.5 (Local Validation)**: `evaluate.py` will normalize both predictions and references before computing WER.
- **S2.1 (Dataset Class)**: The PyTorch Dataset will use `normalize_text` to prepare decoder labels during training.
