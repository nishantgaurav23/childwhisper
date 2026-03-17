"""Tests for post-processing corrections (Spec S5.1).

Tests cover: artifact removal, repeated word collapse, ASR error corrections,
whitespace cleanup, child speech preservation, and integration with normalize_text.
"""

from __future__ import annotations

from src.utils import postprocess_text, normalize_and_postprocess


class TestRemoveArtifacts:
    """Test removal of Whisper hallucination artifacts."""

    def test_removes_music_symbols(self):
        assert postprocess_text("hello \u266a world") == "hello world"

    def test_removes_bracketed_annotations(self):
        assert postprocess_text("hello [inaudible] world") == "hello world"
        assert postprocess_text("[laughing] hello") == "hello"
        assert postprocess_text("hello (applause)") == "hello"

    def test_removes_ellipsis_artifacts(self):
        assert postprocess_text("hello... world") == "hello world"
        assert postprocess_text("...hello") == "hello"

    def test_removes_multiple_artifacts(self):
        assert postprocess_text("\u266a [music] hello \u266a") == "hello"

    def test_preserves_clean_text(self):
        assert postprocess_text("the cat sat on the mat") == "the cat sat on the mat"


class TestCollapseRepeatedWords:
    """Test collapsing of consecutive repeated words (hallucination pattern)."""

    def test_collapses_triple_repeat(self):
        assert postprocess_text("the the the cat") == "the cat"

    def test_collapses_many_repeats(self):
        assert postprocess_text("go go go go go") == "go"

    def test_preserves_double_occurrence(self):
        # Two repetitions could be intentional (e.g., "no no")
        assert postprocess_text("no no") == "no no"

    def test_collapses_multiple_groups(self):
        assert postprocess_text("the the the cat sat sat sat") == "the cat sat"

    def test_preserves_non_consecutive_repeats(self):
        assert postprocess_text("the cat and the dog") == "the cat and the dog"

    def test_case_insensitive_collapse(self):
        # After normalization, text is lowercase, but test robustness
        assert postprocess_text("The the the cat") == "the cat"


class TestASRErrorCorrections:
    """Test dictionary-based ASR error corrections."""

    def test_fixes_gonna_artifact(self):
        # EnglishTextNormalizer may not catch all of these
        result = postprocess_text("i wanna go home")
        assert result == "i wanna go home"  # preserve valid child speech

    def test_preserves_real_words(self):
        assert postprocess_text("hello world") == "hello world"

    def test_fixes_common_whisper_errors(self):
        # "you know" repeated as filler — Whisper sometimes hallucinates this
        result = postprocess_text("you know you know you know hello")
        assert result == "you know hello"


class TestChildSpeechPreservation:
    """Test that valid child speech patterns are NOT corrected."""

    def test_preserves_goed(self):
        assert postprocess_text("i goed to the store") == "i goed to the store"

    def test_preserves_tooths(self):
        assert postprocess_text("i have two tooths") == "i have two tooths"

    def test_preserves_bestest(self):
        assert postprocess_text("you are the bestest") == "you are the bestest"

    def test_preserves_runned(self):
        assert postprocess_text("i runned fast") == "i runned fast"

    def test_preserves_mouses(self):
        assert postprocess_text("three mouses") == "three mouses"


class TestWhitespaceCleanup:
    """Test whitespace normalization."""

    def test_collapses_multiple_spaces(self):
        assert postprocess_text("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert postprocess_text("  hello world  ") == "hello world"

    def test_handles_tabs_and_newlines(self):
        assert postprocess_text("hello\t\nworld") == "hello world"


class TestEdgeCases:
    """Test edge cases and special inputs."""

    def test_empty_string(self):
        assert postprocess_text("") == ""

    def test_none_input(self):
        assert postprocess_text(None) == ""

    def test_whitespace_only(self):
        assert postprocess_text("   ") == ""

    def test_single_word(self):
        assert postprocess_text("hello") == "hello"

    def test_idempotent(self):
        text = "the the the cat \u266a [inaudible]"
        result1 = postprocess_text(text)
        result2 = postprocess_text(result1)
        assert result1 == result2


class TestNormalizeAndPostprocess:
    """Test the combined normalize + postprocess pipeline."""

    def test_normalizes_and_postprocesses(self):
        # EnglishTextNormalizer lowercases + removes punctuation
        result = normalize_and_postprocess("Hello World!")
        assert result == "hello world"

    def test_handles_none(self):
        assert normalize_and_postprocess(None) == ""

    def test_handles_empty(self):
        assert normalize_and_postprocess("") == ""

    def test_postprocesses_after_normalization(self):
        # Artifacts should be removed even after normalization
        result = normalize_and_postprocess("Hello \u266a World")
        assert "♪" not in result

    def test_collapses_repeats_after_normalization(self):
        result = normalize_and_postprocess("The The The cat")
        assert result == "the cat"
