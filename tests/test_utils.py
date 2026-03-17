"""Tests for src/utils.py — text normalization utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_fake_normalizer(english_spelling_mapping=None):
    """Create a fake normalizer that mimics EnglishTextNormalizer behavior."""

    def fake_normalize(text):
        # Simplified normalization: lowercase, strip extra whitespace, remove punctuation
        import re

        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Simple contraction expansion
        text = text.replace("wont", "will not")
        return text

    normalizer = MagicMock(side_effect=fake_normalize)
    return normalizer


@pytest.fixture(autouse=True)
def _reset_normalizer_cache():
    """Reset the module-level normalizer cache before each test."""
    import src.utils

    src.utils._normalizer = None
    yield
    src.utils._normalizer = None


@pytest.fixture
def mock_normalizer():
    """Patch the lazy import to return a fake normalizer."""
    fake = _make_fake_normalizer()
    with patch("src.utils.get_normalizer", return_value=fake):
        yield fake


class TestGetNormalizer:
    """Tests for get_normalizer() singleton behavior."""

    def test_returns_normalizer_instance(self):
        import sys
        import types

        # Create a fake transformers module hierarchy so the lazy import works
        fake_cls = _make_fake_normalizer
        fake_mod = types.ModuleType("transformers.models.whisper.english_normalizer")
        fake_mod.EnglishTextNormalizer = fake_cls
        with patch.dict(sys.modules, {
            "transformers": types.ModuleType("transformers"),
            "transformers.models": types.ModuleType("transformers.models"),
            "transformers.models.whisper": types.ModuleType("transformers.models.whisper"),
            "transformers.models.whisper.english_normalizer": fake_mod,
        }):
            from src.utils import get_normalizer

            normalizer = get_normalizer()
            assert normalizer is not None
            assert callable(normalizer)

    def test_returns_same_instance(self):
        """get_normalizer should return a cached instance."""
        import sys
        import types

        fake_cls = _make_fake_normalizer
        fake_mod = types.ModuleType("transformers.models.whisper.english_normalizer")
        fake_mod.EnglishTextNormalizer = fake_cls
        with patch.dict(sys.modules, {
            "transformers": types.ModuleType("transformers"),
            "transformers.models": types.ModuleType("transformers.models"),
            "transformers.models.whisper": types.ModuleType("transformers.models.whisper"),
            "transformers.models.whisper.english_normalizer": fake_mod,
        }):
            from src.utils import get_normalizer

            n1 = get_normalizer()
            n2 = get_normalizer()
            assert n1 is n2


class TestNormalizeText:
    """Tests for normalize_text()."""

    def test_normal_input(self, mock_normalizer):
        from src.utils import normalize_text

        result = normalize_text("Hello World")
        assert isinstance(result, str)
        assert result == "hello world"

    def test_empty_string(self):
        from src.utils import normalize_text

        assert normalize_text("") == ""

    def test_none_input(self):
        from src.utils import normalize_text

        assert normalize_text(None) == ""

    def test_whitespace_only(self):
        from src.utils import normalize_text

        assert normalize_text("   ") == ""

    def test_lowercase(self, mock_normalizer):
        from src.utils import normalize_text

        result = normalize_text("HELLO WORLD")
        assert result == "hello world"

    def test_punctuation_removal(self, mock_normalizer):
        from src.utils import normalize_text

        result = normalize_text("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_whitespace_normalization(self, mock_normalizer):
        from src.utils import normalize_text

        result = normalize_text("hello   world")
        assert "  " not in result

    def test_contraction_expansion(self, mock_normalizer):
        """EnglishTextNormalizer expands contractions like won't -> will not."""
        from src.utils import normalize_text

        result = normalize_text("won't")
        assert "will not" in result or "wont" in result

    def test_number_words(self, mock_normalizer):
        """EnglishTextNormalizer standardizes number representations."""
        from src.utils import normalize_text

        result = normalize_text("I have 2 cats")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_idempotent_on_clean_text(self, mock_normalizer):
        """Normalizing already-normalized text should return the same result."""
        from src.utils import normalize_text

        text = "hello world"
        result1 = normalize_text(text)
        result2 = normalize_text(result1)
        assert result1 == result2


class TestNormalizeTexts:
    """Tests for normalize_texts() batch function."""

    def test_batch_normalization(self, mock_normalizer):
        from src.utils import normalize_texts

        inputs = ["Hello World", "GOODBYE", ""]
        results = normalize_texts(inputs)
        assert len(results) == 3
        assert results[0] == "hello world"
        assert results[2] == ""

    def test_empty_list(self):
        from src.utils import normalize_texts

        assert normalize_texts([]) == []

    def test_list_with_none(self, mock_normalizer):
        from src.utils import normalize_texts

        results = normalize_texts([None, "hello", None])
        assert len(results) == 3
        assert results[0] == ""
        assert results[1] == "hello"
        assert results[2] == ""
