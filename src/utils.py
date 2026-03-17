"""Shared utilities for ChildWhisper — text normalization and post-processing.

Wraps Whisper's EnglishTextNormalizer for consistent transcript normalization
across preprocessing, training, inference, and evaluation. Adds post-processing
corrections for common ASR errors in children's speech (Spec S5.1).
"""

from __future__ import annotations

import re

_normalizer = None

# Regex for Whisper hallucination artifacts: music symbols, bracketed/parenthesized
# annotations like [inaudible], (laughing), and standalone ellipsis.
_ARTIFACT_PATTERN = re.compile(
    r"\u266a"               # ♪ music symbol
    r"|\[[^\]]*\]"          # [anything in brackets]
    r"|\([^)]*\)"           # (anything in parens)
    r"|\.{2,}"              # two or more dots (ellipsis)
)


def get_normalizer():
    """Return a cached EnglishTextNormalizer instance.

    Lazily imports transformers to avoid hard dependency at import time.
    """
    global _normalizer
    if _normalizer is None:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        _normalizer = EnglishTextNormalizer({})
    return _normalizer


def normalize_text(text: str | None) -> str:
    """Normalize a single text string using Whisper's EnglishTextNormalizer.

    Handles: lowercase, contraction expansion, number standardization,
    punctuation removal, whitespace normalization, diacritics removal.

    Returns empty string for None, empty, or whitespace-only input.
    """
    if text is None or not text.strip():
        return ""
    normalizer = get_normalizer()
    return normalizer(text)


def normalize_texts(texts: list[str | None]) -> list[str]:
    """Normalize a list of text strings."""
    return [normalize_text(t) for t in texts]


def _remove_artifacts(text: str) -> str:
    """Remove Whisper hallucination artifacts (music symbols, brackets, ellipsis)."""
    return _ARTIFACT_PATTERN.sub("", text)


def _collapse_repeated_words(text: str) -> str:
    """Collapse 3+ consecutive identical words/phrases to a single occurrence.

    Handles both single-word repeats ("the the the" -> "the") and multi-word
    phrase repeats ("you know you know you know" -> "you know").
    Preserves double occurrences (e.g., "no no") which may be intentional.
    Case-insensitive matching; output uses lowercase.
    """
    words = text.lower().split()
    if len(words) <= 2:
        return " ".join(words)

    # Try phrase lengths from longest viable down to 1
    max_phrase_len = len(words) // 3
    for phrase_len in range(max_phrase_len, 0, -1):
        i = 0
        result = []
        while i < len(words):
            phrase = words[i : i + phrase_len]
            if i + phrase_len <= len(words):
                # Count consecutive repeats of this phrase
                count = 1
                j = i + phrase_len
                while (
                    j + phrase_len <= len(words)
                    and words[j : j + phrase_len] == phrase
                ):
                    count += 1
                    j += phrase_len
                if count >= 3:
                    result.extend(phrase)
                    i = j
                    continue
            result.append(words[i])
            i += 1
        words = result

    return " ".join(words)


def _clean_whitespace(text: str) -> str:
    """Normalize all whitespace: collapse runs, strip leading/trailing."""
    return re.sub(r"\s+", " ", text).strip()


def postprocess_text(text: str | None) -> str:
    """Apply post-processing corrections to a transcription.

    Pipeline:
    1. Remove Whisper hallucination artifacts (music symbols, brackets, ellipsis)
    2. Collapse 3+ consecutive repeated words (hallucination pattern)
    3. Clean up whitespace

    Preserves valid child speech forms (goed, tooths, bestest, etc.).
    Returns empty string for None, empty, or whitespace-only input.
    """
    if text is None or not text.strip():
        return ""

    text = _remove_artifacts(text)
    text = _clean_whitespace(text)
    text = _collapse_repeated_words(text)
    text = _clean_whitespace(text)

    return text


def normalize_and_postprocess(text: str | None) -> str:
    """Normalize text with EnglishTextNormalizer, then apply post-processing.

    Combined pipeline for inference: normalize first, then fix ASR artifacts.
    """
    normalized = normalize_text(text)
    if not normalized:
        return ""
    return postprocess_text(normalized)
