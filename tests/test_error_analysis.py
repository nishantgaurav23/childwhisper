"""Tests for error analysis tooling (Spec S5.3).

Tests compute_error_breakdown, detect_hallucinations, per-age breakdown,
error_analysis_summary, and format_error_analysis_report.
"""

from __future__ import annotations

from src.evaluate import (
    compute_error_breakdown,
    compute_per_age_error_breakdown,
    detect_hallucinations,
    error_analysis_summary,
    format_error_analysis_report,
)


class TestComputeErrorBreakdown:
    """Tests for compute_error_breakdown."""

    def test_known_substitution(self):
        """One substitution: 'cat' vs 'bat'."""
        refs = ["the cat sat"]
        hyps = ["the bat sat"]
        result = compute_error_breakdown(refs, hyps)
        assert result["substitutions"] == 1
        assert result["insertions"] == 0
        assert result["deletions"] == 0
        assert result["hits"] == 2
        assert result["total_ref_words"] == 3
        assert abs(result["wer"] - 1 / 3) < 1e-6

    def test_known_insertion(self):
        """One insertion: 'the cat' vs 'the big cat'."""
        refs = ["the cat"]
        hyps = ["the big cat"]
        result = compute_error_breakdown(refs, hyps)
        assert result["insertions"] == 1
        assert result["substitutions"] == 0
        assert result["deletions"] == 0

    def test_known_deletion(self):
        """One deletion: 'the big cat' vs 'the cat'."""
        refs = ["the big cat"]
        hyps = ["the cat"]
        result = compute_error_breakdown(refs, hyps)
        assert result["deletions"] == 1
        assert result["substitutions"] == 0
        assert result["insertions"] == 0

    def test_empty_refs_skipped(self):
        """Pairs with empty normalized reference are skipped."""
        refs = ["", "the cat"]
        hyps = ["something", "the cat"]
        result = compute_error_breakdown(refs, hyps)
        assert result["total_ref_words"] == 2
        assert result["wer"] == 0.0

    def test_all_correct(self):
        """Perfect match yields 0 WER."""
        refs = ["hello world", "good morning"]
        hyps = ["hello world", "good morning"]
        result = compute_error_breakdown(refs, hyps)
        assert result["wer"] == 0.0
        assert result["substitutions"] == 0
        assert result["insertions"] == 0
        assert result["deletions"] == 0
        assert result["hits"] == 4

    def test_rates_computed(self):
        """sub_rate, ins_rate, del_rate are count / total_ref_words."""
        refs = ["the cat sat on the mat"]  # 6 words
        hyps = ["a cat sit on a"]  # sub: the->a, sat->sit, the->a, del: mat => 3 sub + 1 del
        result = compute_error_breakdown(refs, hyps)
        assert result["total_ref_words"] == 6
        assert abs(result["sub_rate"] - result["substitutions"] / 6) < 1e-6
        assert abs(result["ins_rate"] - result["insertions"] / 6) < 1e-6
        assert abs(result["del_rate"] - result["deletions"] / 6) < 1e-6

    def test_empty_input(self):
        """Empty lists return zeros."""
        result = compute_error_breakdown([], [])
        assert result["wer"] == 0.0
        assert result["total_ref_words"] == 0


class TestComputePerAgeErrorBreakdown:
    """Tests for compute_per_age_error_breakdown."""

    def test_multi_bucket(self):
        refs = ["the cat", "hello world", "good morning"]
        hyps = ["the bat", "hello world", "good evening"]
        ages = ["3-4", "5-7", "3-4"]
        result = compute_per_age_error_breakdown(refs, hyps, ages)
        assert "3-4" in result
        assert "5-7" in result
        # 3-4 bucket: "the cat"->"the bat" (1 sub), "good morning"->"good evening" (1 sub)
        assert result["3-4"]["substitutions"] == 2
        # 5-7 bucket: perfect match
        assert result["5-7"]["wer"] == 0.0


class TestDetectHallucinations:
    """Tests for detect_hallucinations."""

    def test_normal_ratio_not_flagged(self):
        """Predictions with word count <= threshold * ref are not flagged."""
        refs = ["the cat sat"]
        hyps = ["the cat sat on"]
        result = detect_hallucinations(refs, hyps, threshold=3.0)
        assert len(result) == 0

    def test_high_ratio_flagged(self):
        """Predictions with word count > threshold * ref are flagged."""
        refs = ["hello"]
        hyps = ["hello hello hello hello"]  # 4x ratio
        result = detect_hallucinations(refs, hyps, threshold=3.0)
        assert len(result) == 1
        assert result[0]["index"] == 0
        assert result[0]["ratio"] == 4.0

    def test_empty_ref_nonempty_hyp_flagged(self):
        """Empty reference with non-empty hypothesis is flagged."""
        refs = [""]
        hyps = ["something was said"]
        result = detect_hallucinations(refs, hyps, threshold=3.0)
        assert len(result) == 1
        assert result[0]["ratio"] == float("inf")

    def test_empty_hyp_not_flagged(self):
        """Empty hypothesis is never a hallucination."""
        refs = ["the cat"]
        hyps = [""]
        result = detect_hallucinations(refs, hyps, threshold=3.0)
        assert len(result) == 0

    def test_both_empty_not_flagged(self):
        """Both empty — not a hallucination."""
        refs = [""]
        hyps = [""]
        result = detect_hallucinations(refs, hyps, threshold=3.0)
        assert len(result) == 0


class TestErrorAnalysisSummary:
    """Tests for error_analysis_summary."""

    def test_integration(self):
        refs = ["the cat sat", "hello world"]
        hyps = ["the bat sat", "hello world"]
        ages = ["3-4", "5-7"]
        result = error_analysis_summary(refs, hyps, ages)
        assert "overall_breakdown" in result
        assert "per_age_breakdown" in result
        assert "hallucinations" in result
        assert "hallucination_count" in result
        assert "hallucination_rate" in result
        assert result["overall_breakdown"]["substitutions"] == 1
        assert result["hallucination_count"] == 0
        assert result["hallucination_rate"] == 0.0


class TestFormatErrorAnalysisReport:
    """Tests for format_error_analysis_report."""

    def test_contains_expected_sections(self):
        refs = ["the cat sat", "hello"]
        hyps = ["the bat sat", "hello hello hello hello"]
        ages = ["3-4", "5-7"]
        summary = error_analysis_summary(refs, hyps, ages, hallucination_threshold=3.0)
        report = format_error_analysis_report(summary)
        assert "Substitutions" in report or "Sub" in report
        assert "Insertions" in report or "Ins" in report
        assert "Deletions" in report or "Del" in report
        assert "Hallucination" in report or "hallucination" in report
        assert "3-4" in report
        assert "5-7" in report
