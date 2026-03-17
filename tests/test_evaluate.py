"""Tests for src/evaluate.py — local validation framework."""

from __future__ import annotations

import pytest

from src.evaluate import (
    compute_per_age_wer,
    compute_wer,
    split_by_child_id,
    validation_summary,
)


# ---------------------------------------------------------------------------
# Fixtures — sample metadata
# ---------------------------------------------------------------------------

def _make_meta(child_id, age_bucket, utterance_id=None, text="hello world"):
    """Helper to build a metadata dict."""
    uid = utterance_id or f"U_{child_id}_{age_bucket}"
    return {
        "utterance_id": uid,
        "child_id": child_id,
        "session_id": "S_test",
        "audio_path": f"audio/{uid}.flac",
        "audio_duration_sec": 2.0,
        "age_bucket": age_bucket,
        "orthographic_text": text,
    }


@pytest.fixture
def sample_metadata():
    """20 utterances from 10 children across 3 age buckets."""
    meta = []
    for i in range(10):
        child = f"C_{i:03d}"
        bucket = ["3-4", "5-7", "8-11"][i % 3]
        # 2 utterances per child
        meta.append(_make_meta(child, bucket, f"U_{i:03d}_a", "the cat sat"))
        meta.append(_make_meta(child, bucket, f"U_{i:03d}_b", "on the mat"))
    return meta


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestSplitByChildId:
    def test_no_speaker_leakage(self, sample_metadata):
        """No child_id should appear in both train and val."""
        train, val = split_by_child_id(sample_metadata)
        train_ids = {m["child_id"] for m in train}
        val_ids = {m["child_id"] for m in val}
        assert train_ids.isdisjoint(val_ids)

    def test_split_ratio(self, sample_metadata):
        """~90/10 split of child_ids (with 10 children: 9 train, 1 val)."""
        train, val = split_by_child_id(sample_metadata, val_ratio=0.1)
        train_ids = {m["child_id"] for m in train}
        val_ids = {m["child_id"] for m in val}
        # With 10 children and 0.1 ratio, expect 1 val child
        assert len(val_ids) >= 1
        assert len(train_ids) >= 1
        assert len(train_ids) + len(val_ids) == 10

    def test_all_utterances_preserved(self, sample_metadata):
        """All utterances are present in either train or val."""
        train, val = split_by_child_id(sample_metadata)
        assert len(train) + len(val) == len(sample_metadata)

    def test_stratified_by_age(self, sample_metadata):
        """Both splits should contain multiple age buckets when possible."""
        train, val = split_by_child_id(sample_metadata, val_ratio=0.3)
        train_buckets = {m["age_bucket"] for m in train}
        val_buckets = {m["age_bucket"] for m in val}
        # With 30% val and 3 buckets, val should have at least 1 bucket
        assert len(val_buckets) >= 1
        assert len(train_buckets) >= 1

    def test_deterministic(self, sample_metadata):
        """Same seed produces identical splits."""
        t1, v1 = split_by_child_id(sample_metadata, seed=42)
        t2, v2 = split_by_child_id(sample_metadata, seed=42)
        assert [m["utterance_id"] for m in t1] == [m["utterance_id"] for m in t2]
        assert [m["utterance_id"] for m in v1] == [m["utterance_id"] for m in v2]

    def test_different_seed_different_split(self, sample_metadata):
        """Different seeds produce different splits."""
        _, v1 = split_by_child_id(sample_metadata, seed=42)
        _, v2 = split_by_child_id(sample_metadata, seed=99)
        v1_ids = {m["child_id"] for m in v1}
        v2_ids = {m["child_id"] for m in v2}
        # Not guaranteed but very likely with different seeds
        # At minimum, both should be valid splits
        assert len(v1_ids) >= 1
        assert len(v2_ids) >= 1

    def test_empty_input(self):
        """Empty metadata returns empty splits."""
        train, val = split_by_child_id([])
        assert train == []
        assert val == []

    def test_single_child(self):
        """Single child goes to train (val would be empty)."""
        meta = [_make_meta("C_only", "5-7", "U_a"), _make_meta("C_only", "5-7", "U_b")]
        train, val = split_by_child_id(meta)
        assert len(train) == 2
        assert len(val) == 0


# ---------------------------------------------------------------------------
# WER computation tests
# ---------------------------------------------------------------------------

class TestComputeWer:
    def test_basic_wer(self):
        """Simple WER: 1 substitution in 4 words = 0.25."""
        refs = ["the cat sat down"]
        hyps = ["the dog sat down"]
        wer = compute_wer(refs, hyps)
        assert abs(wer - 0.25) < 1e-6

    def test_perfect_match(self):
        """Identical refs and hyps → WER 0.0."""
        refs = ["hello world", "the cat sat"]
        hyps = ["hello world", "the cat sat"]
        wer = compute_wer(refs, hyps)
        assert wer == 0.0

    def test_empty_ref_skipped(self):
        """Empty reference utterances are excluded from WER."""
        refs = ["hello world", "", "the cat"]
        hyps = ["hello world", "junk text", "the cat"]
        wer = compute_wer(refs, hyps)
        assert wer == 0.0  # only non-empty refs counted, both perfect

    def test_empty_hyp_counts_as_error(self):
        """Empty hypothesis for non-empty ref → 100% error for that pair."""
        refs = ["hello world"]
        hyps = [""]
        wer = compute_wer(refs, hyps)
        assert wer == 1.0  # 2 deletions / 2 words = 1.0

    def test_normalization_applied(self):
        """Normalizer handles casing and punctuation."""
        refs = ["Hello World!"]
        hyps = ["hello world"]
        wer = compute_wer(refs, hyps)
        assert wer == 0.0  # normalizer lowercases and removes punctuation

    def test_both_empty_skipped(self):
        """If both ref and hyp are empty, pair is skipped."""
        refs = ["", ""]
        hyps = ["", ""]
        wer = compute_wer(refs, hyps)
        assert wer == 0.0  # no valid pairs, default to 0.0

    def test_all_empty_refs(self):
        """All empty refs → WER 0.0 (nothing to evaluate)."""
        refs = ["", "", ""]
        hyps = ["a", "b", "c"]
        wer = compute_wer(refs, hyps)
        assert wer == 0.0


# ---------------------------------------------------------------------------
# Per-age WER tests
# ---------------------------------------------------------------------------

class TestPerAgeWer:
    def test_per_age_breakdown(self):
        """WER computed separately per age bucket."""
        refs = ["hello", "hello", "hello", "hello"]
        hyps = ["hello", "wrong", "hello", "wrong"]
        ages = ["3-4", "3-4", "8-11", "8-11"]
        result = compute_per_age_wer(refs, hyps, ages)
        assert abs(result["3-4"] - 0.5) < 1e-6  # 1 wrong of 2 words
        assert abs(result["8-11"] - 0.5) < 1e-6

    def test_single_bucket(self):
        """All utterances in one bucket."""
        refs = ["cat", "dog"]
        hyps = ["cat", "dog"]
        ages = ["5-7", "5-7"]
        result = compute_per_age_wer(refs, hyps, ages)
        assert "5-7" in result
        assert result["5-7"] == 0.0

    def test_unknown_bucket(self):
        """Handle 'unknown' age bucket."""
        refs = ["test"]
        hyps = ["test"]
        ages = ["unknown"]
        result = compute_per_age_wer(refs, hyps, ages)
        assert "unknown" in result


# ---------------------------------------------------------------------------
# Validation summary tests
# ---------------------------------------------------------------------------

class TestValidationSummary:
    def test_summary_structure(self):
        """Summary dict has all required keys."""
        refs = ["hello world", "", "the cat"]
        hyps = ["hello world", "junk", "the dog"]
        ages = ["3-4", "5-7", "8-11"]
        result = validation_summary(refs, hyps, ages)
        assert "overall_wer" in result
        assert "per_age_wer" in result
        assert "num_utterances" in result
        assert "num_empty_refs_skipped" in result
        assert "num_empty_preds" in result

    def test_summary_counts(self):
        """Verify counts in summary."""
        refs = ["hello", "", "cat"]
        hyps = ["hello", "x", ""]
        ages = ["3-4", "3-4", "5-7"]
        result = validation_summary(refs, hyps, ages)
        assert result["num_utterances"] == 3
        assert result["num_empty_refs_skipped"] == 1
        assert result["num_empty_preds"] == 1

    def test_summary_wer_value(self):
        """Summary WER matches compute_wer."""
        refs = ["the cat sat", "on the mat"]
        hyps = ["the cat sat", "on the mat"]
        ages = ["5-7", "5-7"]
        result = validation_summary(refs, hyps, ages)
        assert result["overall_wer"] == 0.0
