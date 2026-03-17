"""Tests for src/autowhisper/logger.py — Results TSV logger."""

import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def tmp_log(tmp_path):
    return str(tmp_path / "results.tsv")


@pytest.fixture
def sample_result():
    return {
        "experiment_id": "001",
        "commit_hash": "abc1234",
        "val_wer": 0.1523,
        "peak_vram_mb": 14200,
        "duration_sec": 845,
        "status": "keep",
        "description": "Increased LoRA rank to 64",
    }


@pytest.fixture
def baseline_result():
    return {
        "experiment_id": "000",
        "commit_hash": "baseline",
        "val_wer": 0.2000,
        "peak_vram_mb": 12000,
        "duration_sec": 800,
        "status": "baseline",
        "description": "Baseline configuration",
    }


@pytest.fixture
def crash_result():
    return {
        "experiment_id": "002",
        "commit_hash": "def5678",
        "val_wer": -1.0,
        "peak_vram_mb": -1,
        "duration_sec": 10,
        "status": "crash",
        "description": "OOM with batch_size=16",
    }


class TestInitLog:
    def test_creates_header(self, tmp_log):
        from src.autowhisper.logger import init_log

        init_log(tmp_log)
        with open(tmp_log) as f:
            header = f.readline().strip()
        expected_fields = [
            "experiment_id",
            "commit_hash",
            "val_wer",
            "peak_vram_mb",
            "duration_sec",
            "status",
            "description",
        ]
        assert header == "\t".join(expected_fields)

    def test_creates_file(self, tmp_log):
        from src.autowhisper.logger import init_log

        init_log(tmp_log)
        assert os.path.exists(tmp_log)

    def test_creates_parent_dirs(self, tmp_path):
        log_path = str(tmp_path / "sub" / "dir" / "results.tsv")
        from src.autowhisper.logger import init_log

        init_log(log_path)
        assert os.path.exists(log_path)


class TestAppendResult:
    def test_adds_row(self, tmp_log, sample_result):
        from src.autowhisper.logger import init_log, append_result

        init_log(tmp_log)
        append_result(tmp_log, sample_result)
        with open(tmp_log) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 row

    def test_multiple_rows(self, tmp_log, sample_result, crash_result):
        from src.autowhisper.logger import init_log, append_result

        init_log(tmp_log)
        append_result(tmp_log, sample_result)
        append_result(tmp_log, crash_result)
        with open(tmp_log) as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows

    def test_preserves_tab_format(self, tmp_log, sample_result):
        from src.autowhisper.logger import init_log, append_result

        init_log(tmp_log)
        append_result(tmp_log, sample_result)
        with open(tmp_log) as f:
            lines = f.readlines()
        data_line = lines[1].strip()
        fields = data_line.split("\t")
        assert len(fields) == 7


class TestLoadResults:
    def test_parses_all_fields(self, tmp_log, sample_result):
        from src.autowhisper.logger import init_log, append_result, load_results

        init_log(tmp_log)
        append_result(tmp_log, sample_result)
        results = load_results(tmp_log)
        assert len(results) == 1
        r = results[0]
        assert r["experiment_id"] == "001"
        assert r["commit_hash"] == "abc1234"
        assert isinstance(r["val_wer"], float)
        assert abs(r["val_wer"] - 0.1523) < 1e-6
        assert isinstance(r["peak_vram_mb"], int)
        assert r["peak_vram_mb"] == 14200
        assert isinstance(r["duration_sec"], int)
        assert r["duration_sec"] == 845
        assert r["status"] == "keep"
        assert r["description"] == "Increased LoRA rank to 64"

    def test_empty_log(self, tmp_log):
        from src.autowhisper.logger import init_log, load_results

        init_log(tmp_log)
        results = load_results(tmp_log)
        assert results == []


class TestGetBestWer:
    def test_ignores_crashes(self, tmp_log, baseline_result, crash_result, sample_result):
        from src.autowhisper.logger import (
            init_log,
            append_result,
            get_best_wer,
        )

        init_log(tmp_log)
        append_result(tmp_log, baseline_result)
        append_result(tmp_log, crash_result)
        append_result(tmp_log, sample_result)
        best = get_best_wer(tmp_log)
        assert abs(best - 0.1523) < 1e-6

    def test_includes_baseline(self, tmp_log, baseline_result):
        from src.autowhisper.logger import init_log, append_result, get_best_wer

        init_log(tmp_log)
        append_result(tmp_log, baseline_result)
        best = get_best_wer(tmp_log)
        assert abs(best - 0.2000) < 1e-6

    def test_no_valid_results_returns_inf(self, tmp_log, crash_result):
        from src.autowhisper.logger import init_log, append_result, get_best_wer

        init_log(tmp_log)
        append_result(tmp_log, crash_result)
        best = get_best_wer(tmp_log)
        assert best == float("inf")

    def test_empty_log_returns_inf(self, tmp_log):
        from src.autowhisper.logger import init_log, get_best_wer

        init_log(tmp_log)
        best = get_best_wer(tmp_log)
        assert best == float("inf")


class TestGetFrontier:
    def test_monotonically_decreasing(self, tmp_log):
        from src.autowhisper.logger import init_log, append_result, get_frontier

        init_log(tmp_log)
        results_data = [
            {"experiment_id": "000", "commit_hash": "baseline", "val_wer": 0.25,
             "peak_vram_mb": 12000, "duration_sec": 800, "status": "baseline",
             "description": "Baseline"},
            {"experiment_id": "001", "commit_hash": "aaa", "val_wer": 0.22,
             "peak_vram_mb": 12000, "duration_sec": 800, "status": "keep",
             "description": "Improvement 1"},
            {"experiment_id": "002", "commit_hash": "bbb", "val_wer": 0.23,
             "peak_vram_mb": 12000, "duration_sec": 800, "status": "discard",
             "description": "Regression"},
            {"experiment_id": "003", "commit_hash": "ccc", "val_wer": 0.18,
             "peak_vram_mb": 12000, "duration_sec": 800, "status": "keep",
             "description": "Improvement 2"},
            {"experiment_id": "004", "commit_hash": "ddd", "val_wer": -1.0,
             "peak_vram_mb": -1, "duration_sec": 10, "status": "crash",
             "description": "Crash"},
        ]
        for r in results_data:
            append_result(tmp_log, r)

        frontier = get_frontier(tmp_log)
        wers = [r["val_wer"] for r in frontier]
        # Frontier should be strictly decreasing
        for i in range(1, len(wers)):
            assert wers[i] < wers[i - 1]
        # Should include baseline + keeps that improved
        assert len(frontier) >= 2

    def test_empty_frontier(self, tmp_log):
        from src.autowhisper.logger import init_log, get_frontier

        init_log(tmp_log)
        frontier = get_frontier(tmp_log)
        assert frontier == []


class TestPrintSummary:
    def test_format(self, tmp_log, baseline_result, sample_result, crash_result, capsys):
        from src.autowhisper.logger import (
            init_log,
            append_result,
            print_summary,
        )

        init_log(tmp_log)
        append_result(tmp_log, baseline_result)
        append_result(tmp_log, sample_result)
        append_result(tmp_log, crash_result)
        print_summary(tmp_log)
        captured = capsys.readouterr()
        output = captured.out.lower()
        assert "total" in output or "3" in output
        assert "keep" in output
        assert "discard" in output or "crash" in output
        assert "best" in output or "0.1523" in output


class TestPlotProgress:
    @patch("src.autowhisper.logger.plt")
    def test_creates_file(self, mock_plt, tmp_log, tmp_path, baseline_result, sample_result):
        from src.autowhisper.logger import init_log, append_result, plot_progress

        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        init_log(tmp_log)
        append_result(tmp_log, baseline_result)
        append_result(tmp_log, sample_result)
        output_path = str(tmp_path / "progress.png")
        plot_progress(tmp_log, output_path)
        mock_plt.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")
