"""Tests for src/autowhisper/runner.py — Experiment loop orchestrator."""

import pytest
from unittest.mock import patch, MagicMock
import subprocess


@pytest.fixture
def mock_git():
    """Mock all git subprocess calls."""
    with patch("src.autowhisper.runner.subprocess") as mock_sub:
        mock_sub.run.return_value = MagicMock(
            returncode=0, stdout="", stderr=""
        )
        mock_sub.PIPE = subprocess.PIPE
        mock_sub.TimeoutExpired = subprocess.TimeoutExpired
        yield mock_sub


class TestInitRun:
    def test_creates_branch(self, mock_git):
        from src.autowhisper.runner import init_run

        branch = init_run("run_test", base_branch="main")
        assert branch == "autowhisper/run_test"
        # Verify git checkout -b was called
        calls = mock_git.run.call_args_list
        branch_call = [
            c for c in calls
            if any("checkout" in str(a) for a in c[0])
            or any("checkout" in str(a) for a in c.get("args", [()]))
        ]
        assert len(branch_call) > 0

    def test_returns_branch_name(self, mock_git):
        from src.autowhisper.runner import init_run

        branch = init_run("mar17")
        assert branch == "autowhisper/mar17"


class TestRunExperiment:
    def test_parses_wer(self):
        with patch("src.autowhisper.runner.subprocess") as mock_sub:
            mock_sub.PIPE = subprocess.PIPE
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "Training...\nval_wer: 0.1523\npeak_vram_mb: 14200\nDone."
            mock_proc.stderr = ""
            mock_sub.run.return_value = mock_proc

            from src.autowhisper.runner import run_experiment

            result = run_experiment("train.py", time_budget=900)
            assert abs(result["val_wer"] - 0.1523) < 1e-6

    def test_parses_vram(self):
        with patch("src.autowhisper.runner.subprocess") as mock_sub:
            mock_sub.PIPE = subprocess.PIPE
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "val_wer: 0.15\npeak_vram_mb: 14200\n"
            mock_proc.stderr = ""
            mock_sub.run.return_value = mock_proc

            from src.autowhisper.runner import run_experiment

            result = run_experiment("train.py", time_budget=900)
            assert result["peak_vram_mb"] == 14200

    def test_timeout_returns_crash(self):
        with patch("src.autowhisper.runner.subprocess") as mock_sub:
            mock_sub.PIPE = subprocess.PIPE
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            mock_sub.run.side_effect = subprocess.TimeoutExpired(
                cmd="python train.py", timeout=960
            )

            from src.autowhisper.runner import run_experiment

            result = run_experiment("train.py", time_budget=900)
            assert result["status"] == "crash"
            assert result["val_wer"] == -1.0

    def test_crash_returns_crash_status(self):
        with patch("src.autowhisper.runner.subprocess") as mock_sub:
            mock_sub.PIPE = subprocess.PIPE
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.stdout = "Error: OOM"
            mock_proc.stderr = "Traceback..."
            mock_sub.run.return_value = mock_proc

            from src.autowhisper.runner import run_experiment

            result = run_experiment("train.py", time_budget=900)
            assert result["status"] == "crash"
            assert result["val_wer"] == -1.0
            assert result["peak_vram_mb"] == -1

    def test_records_duration(self):
        with patch("src.autowhisper.runner.subprocess") as mock_sub, \
             patch("src.autowhisper.runner.time") as mock_time:
            mock_sub.PIPE = subprocess.PIPE
            mock_sub.TimeoutExpired = subprocess.TimeoutExpired
            mock_time.time.side_effect = [100.0, 945.0]
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "val_wer: 0.15\npeak_vram_mb: 14200\n"
            mock_proc.stderr = ""
            mock_sub.run.return_value = mock_proc

            from src.autowhisper.runner import run_experiment

            result = run_experiment("train.py", time_budget=900)
            assert result["duration_sec"] == 845


class TestEvaluateAndDecide:
    def test_keep_when_improved(self):
        from src.autowhisper.runner import evaluate_and_decide

        result = {"val_wer": 0.15, "status": "ok"}
        assert evaluate_and_decide(result, best_wer=0.20) == "keep"

    def test_discard_when_equal(self):
        from src.autowhisper.runner import evaluate_and_decide

        result = {"val_wer": 0.20, "status": "ok"}
        assert evaluate_and_decide(result, best_wer=0.20) == "discard"

    def test_discard_when_worse(self):
        from src.autowhisper.runner import evaluate_and_decide

        result = {"val_wer": 0.25, "status": "ok"}
        assert evaluate_and_decide(result, best_wer=0.20) == "discard"

    def test_crash(self):
        from src.autowhisper.runner import evaluate_and_decide

        result = {"val_wer": -1.0, "status": "crash"}
        assert evaluate_and_decide(result, best_wer=0.20) == "crash"


class TestKeepExperiment:
    def test_commits(self, mock_git):
        from src.autowhisper.runner import keep_experiment

        keep_experiment("Increased LoRA rank to 64")
        calls = mock_git.run.call_args_list
        commit_calls = [
            c for c in calls
            if any("commit" in str(a) for a in c[0])
        ]
        assert len(commit_calls) > 0


class TestRevertExperiment:
    def test_resets(self, mock_git):
        from src.autowhisper.runner import revert_experiment

        revert_experiment()
        calls = mock_git.run.call_args_list
        reset_calls = [
            c for c in calls
            if any("checkout" in str(a) or "reset" in str(a) for a in c[0])
        ]
        assert len(reset_calls) > 0


class TestLogResult:
    def test_appends_to_log(self, tmp_path):
        from src.autowhisper.logger import init_log
        from src.autowhisper.runner import log_result

        log_path = str(tmp_path / "results.tsv")
        init_log(log_path)
        result = {
            "val_wer": 0.1523,
            "peak_vram_mb": 14200,
            "duration_sec": 845,
            "status": "ok",
        }
        log_result(
            result=result,
            decision="keep",
            description="Test change",
            experiment_id="001",
            commit_hash="abc1234",
            log_path=log_path,
        )
        with open(log_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
