"""Tests for src/autowhisper/prepare.py — Fixed evaluation harness."""

import pytest
from unittest.mock import patch


class FakeNormalizer:
    """Mimics EnglishTextNormalizer: lowercase, strip punctuation, collapse whitespace."""

    def __call__(self, text):
        import re

        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Handle contractions: "it's" → "its"
        text = text.replace("it s", "its")
        return text


@pytest.fixture
def sample_metadata():
    """Create sample validation metadata for testing."""
    entries = []
    child_ids = [f"child_{i}" for i in range(20)]
    age_buckets = ["3-4", "5-6", "7-8", "9-10"]
    for i, cid in enumerate(child_ids):
        bucket = age_buckets[i % len(age_buckets)]
        for j in range(15):
            entries.append(
                {
                    "audio_path": f"/data/audio/{cid}_{j:03d}.flac",
                    "transcript": f"word{j}",
                    "child_id": cid,
                    "age_bucket": bucket,
                    "duration": 1.5 + (j * 0.1),
                }
            )
    return entries


@pytest.fixture
def val_metadata(sample_metadata):
    """Simulate validation-only metadata (subset of children)."""
    val_children = {f"child_{i}" for i in range(10, 20)}
    return [e for e in sample_metadata if e["child_id"] in val_children]


class TestLoadFastEvalSet:
    def test_returns_correct_count(self, val_metadata):
        with patch("src.autowhisper.prepare.load_validation_metadata") as mock_load:
            mock_load.return_value = val_metadata
            from src.autowhisper.prepare import load_fast_eval_set

            result = load_fast_eval_set("/data", n_samples=50)
            assert len(result) == 50

    def test_returns_all_when_n_exceeds_available(self, val_metadata):
        with patch("src.autowhisper.prepare.load_validation_metadata") as mock_load:
            mock_load.return_value = val_metadata
            from src.autowhisper.prepare import load_fast_eval_set

            result = load_fast_eval_set("/data", n_samples=9999)
            assert len(result) == len(val_metadata)

    def test_deterministic(self, val_metadata):
        with patch("src.autowhisper.prepare.load_validation_metadata") as mock_load:
            mock_load.return_value = val_metadata
            from src.autowhisper.prepare import load_fast_eval_set

            result1 = load_fast_eval_set("/data", n_samples=50)
            result2 = load_fast_eval_set("/data", n_samples=50)
            assert result1 == result2

    def test_no_speaker_leakage(self, val_metadata):
        """All returned items should be from validation children only."""
        val_child_ids = {e["child_id"] for e in val_metadata}
        with patch("src.autowhisper.prepare.load_validation_metadata") as mock_load:
            mock_load.return_value = val_metadata
            from src.autowhisper.prepare import load_fast_eval_set

            result = load_fast_eval_set("/data", n_samples=50)
            returned_child_ids = {e["child_id"] for e in result}
            assert returned_child_ids.issubset(val_child_ids)

    def test_biased_toward_shorter(self, val_metadata):
        """Shorter utterances should appear earlier / be preferred."""
        with patch("src.autowhisper.prepare.load_validation_metadata") as mock_load:
            mock_load.return_value = val_metadata
            from src.autowhisper.prepare import load_fast_eval_set

            result = load_fast_eval_set("/data", n_samples=50)
            durations = [e["duration"] for e in result]
            all_durations = [e["duration"] for e in val_metadata]
            # Average duration of selected should be <= average of all
            assert sum(durations) / len(durations) <= sum(all_durations) / len(all_durations)

    def test_returns_required_fields(self, val_metadata):
        with patch("src.autowhisper.prepare.load_validation_metadata") as mock_load:
            mock_load.return_value = val_metadata
            from src.autowhisper.prepare import load_fast_eval_set

            result = load_fast_eval_set("/data", n_samples=5)
            for item in result:
                assert "audio_path" in item
                assert "transcript" in item
                assert "child_id" in item
                assert "age_bucket" in item


class TestEvaluateWer:
    @patch("src.autowhisper.prepare.get_normalizer")
    def test_perfect_predictions(self, mock_norm):
        mock_norm.return_value = FakeNormalizer()
        from src.autowhisper.prepare import evaluate_wer

        refs = ["hello", "world", "test"]
        preds = ["hello", "world", "test"]
        result = evaluate_wer(preds, refs)
        assert result["wer"] == 0.0
        assert result["n_samples"] == 3

    @patch("src.autowhisper.prepare.get_normalizer")
    def test_all_wrong(self, mock_norm):
        mock_norm.return_value = FakeNormalizer()
        from src.autowhisper.prepare import evaluate_wer

        refs = ["hello", "world"]
        preds = ["goodbye", "earth"]
        result = evaluate_wer(preds, refs)
        assert result["wer"] > 0

    @patch("src.autowhisper.prepare.get_normalizer")
    def test_applies_normalizer(self, mock_norm):
        mock_norm.return_value = FakeNormalizer()
        from src.autowhisper.prepare import evaluate_wer

        refs = ["IT'S"]
        preds = ["its"]
        result = evaluate_wer(preds, refs)
        assert result["wer"] == 0.0

    @patch("src.autowhisper.prepare.get_normalizer")
    def test_returns_error_breakdown(self, mock_norm):
        mock_norm.return_value = FakeNormalizer()
        from src.autowhisper.prepare import evaluate_wer

        refs = ["hello world"]
        preds = ["hello earth"]
        result = evaluate_wer(preds, refs)
        assert "substitutions" in result
        assert "deletions" in result
        assert "insertions" in result

    def test_empty_predictions_raises(self):
        from src.autowhisper.prepare import evaluate_wer

        with pytest.raises(ValueError):
            evaluate_wer([], [])

    @patch("src.autowhisper.prepare.get_normalizer")
    def test_mismatched_lengths_raises(self, mock_norm):
        mock_norm.return_value = FakeNormalizer()
        from src.autowhisper.prepare import evaluate_wer

        with pytest.raises(ValueError):
            evaluate_wer(["hello"], ["hello", "world"])


class TestEvaluateWerByAge:
    @patch("src.autowhisper.prepare.get_normalizer")
    def test_returns_all_buckets(self, mock_norm):
        mock_norm.return_value = FakeNormalizer()
        from src.autowhisper.prepare import evaluate_wer_by_age

        refs = ["hello", "world", "test", "cat"]
        preds = ["hello", "world", "test", "cat"]
        ages = ["3-4", "5-6", "3-4", "7-8"]
        result = evaluate_wer_by_age(preds, refs, ages)
        assert "3-4" in result
        assert "5-6" in result
        assert "7-8" in result

    @patch("src.autowhisper.prepare.get_normalizer")
    def test_per_age_perfect(self, mock_norm):
        mock_norm.return_value = FakeNormalizer()
        from src.autowhisper.prepare import evaluate_wer_by_age

        refs = ["hello", "world"]
        preds = ["hello", "world"]
        ages = ["3-4", "5-6"]
        result = evaluate_wer_by_age(preds, refs, ages)
        assert result["3-4"] == 0.0
        assert result["5-6"] == 0.0

    @patch("src.autowhisper.prepare.get_normalizer")
    def test_empty_raises(self, mock_norm):
        mock_norm.return_value = FakeNormalizer()
        from src.autowhisper.prepare import evaluate_wer_by_age

        with pytest.raises(ValueError):
            evaluate_wer_by_age([], [], [])


class TestConstants:
    def test_time_budget_exists(self):
        from src.autowhisper.prepare import TIME_BUDGET

        assert TIME_BUDGET == 900

    def test_eval_samples_exists(self):
        from src.autowhisper.prepare import EVAL_SAMPLES

        assert EVAL_SAMPLES == 200
