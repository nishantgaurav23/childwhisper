"""Shared utilities for ChildWhisper — text normalization.

Wraps Whisper's EnglishTextNormalizer for consistent transcript normalization
across preprocessing, training, inference, and evaluation.
"""

from __future__ import annotations

_normalizer = None


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
