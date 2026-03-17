"""Tests for S3.3 — Ensemble inference pipeline (submission/main.py).

Tests the two-model ensemble: Whisper-large-v3 + LoRA (Model A) and
Whisper-small fine-tuned (Model B), with confidence-based merging and
time budget management.

All model loading and audio I/O is mocked. No real Whisper model or audio files needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure submission/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))

from main import (  # noqa: E402
    check_time_budget,
    load_large_model,
    merge_predictions,
    MODEL_B_CUTOFF_SEC,
    SAFETY_MARGIN_SEC,
    TIME_LIMIT_SEC,
    LORA_ADAPTER_DIR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_utterances():
    """Sample utterance metadata dicts."""
    return [
        {
            "utterance_id": "utt_001",
            "audio_path": "audio/utt_001.flac",
            "audio_duration_sec": 2.5,
            "child_id": "child_1",
        },
        {
            "utterance_id": "utt_002",
            "audio_path": "audio/utt_002.flac",
            "audio_duration_sec": 1.0,
            "child_id": "child_2",
        },
        {
            "utterance_id": "utt_003",
            "audio_path": "audio/utt_003.flac",
            "audio_duration_sec": 5.0,
            "child_id": "child_1",
        },
    ]


@pytest.fixture
def mock_audio():
    """Return a 1-second 16 kHz sine wave (non-silent)."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestEnsembleConstants:
    def test_time_limit(self):
        assert TIME_LIMIT_SEC == 7200

    def test_safety_margin(self):
        assert SAFETY_MARGIN_SEC == 300

    def test_model_b_cutoff(self):
        assert MODEL_B_CUTOFF_SEC == 5400

    def test_lora_adapter_dir_is_path(self):
        assert isinstance(LORA_ADAPTER_DIR, Path)


# ---------------------------------------------------------------------------
# merge_predictions
# ---------------------------------------------------------------------------

class TestMergePredictions:
    def test_prefers_model_a_when_both_non_empty(self):
        preds_a = {"utt_001": "hello", "utt_002": "world"}
        preds_b = {"utt_001": "hi", "utt_002": "earth"}
        result = merge_predictions(preds_a, preds_b)
        assert result["utt_001"] == "hello"
        assert result["utt_002"] == "world"

    def test_falls_back_to_b_when_a_empty(self):
        preds_a = {"utt_001": "", "utt_002": "world"}
        preds_b = {"utt_001": "hello from b", "utt_002": "earth"}
        result = merge_predictions(preds_a, preds_b)
        assert result["utt_001"] == "hello from b"
        assert result["utt_002"] == "world"

    def test_falls_back_to_b_when_a_whitespace(self):
        preds_a = {"utt_001": "   ", "utt_002": "ok"}
        preds_b = {"utt_001": "fallback", "utt_002": "other"}
        result = merge_predictions(preds_a, preds_b)
        assert result["utt_001"] == "fallback"

    def test_empty_when_both_empty(self):
        preds_a = {"utt_001": ""}
        preds_b = {"utt_001": ""}
        result = merge_predictions(preds_a, preds_b)
        assert result["utt_001"] == ""

    def test_empty_when_b_missing_key(self):
        preds_a = {"utt_001": ""}
        preds_b = {}  # no prediction for utt_001
        result = merge_predictions(preds_a, preds_b)
        assert result["utt_001"] == ""

    def test_no_model_b_predictions(self):
        """When model B didn't run (None), use model A predictions as-is."""
        preds_a = {"utt_001": "hello", "utt_002": ""}
        result = merge_predictions(preds_a, None)
        assert result["utt_001"] == "hello"
        assert result["utt_002"] == ""

    def test_preserves_all_keys_from_a(self):
        preds_a = {"utt_001": "a", "utt_002": "b", "utt_003": "c"}
        preds_b = {"utt_001": "x"}
        result = merge_predictions(preds_a, preds_b)
        assert set(result.keys()) == {"utt_001", "utt_002", "utt_003"}


# ---------------------------------------------------------------------------
# check_time_budget
# ---------------------------------------------------------------------------

class TestCheckTimeBudget:
    def test_allows_model_b_when_plenty_of_time(self):
        # 30 minutes elapsed, well under 90 minute cutoff
        assert check_time_budget(elapsed_sec=1800) is True

    def test_denies_model_b_when_past_cutoff(self):
        # 100 minutes elapsed, past 90 minute cutoff
        assert check_time_budget(elapsed_sec=6000) is False

    def test_denies_at_exact_cutoff(self):
        assert check_time_budget(elapsed_sec=MODEL_B_CUTOFF_SEC) is False

    def test_allows_just_before_cutoff(self):
        assert check_time_budget(elapsed_sec=MODEL_B_CUTOFF_SEC - 1) is True

    def test_custom_cutoff(self):
        assert check_time_budget(elapsed_sec=100, cutoff_sec=200) is True
        assert check_time_budget(elapsed_sec=200, cutoff_sec=200) is False


# ---------------------------------------------------------------------------
# load_large_model
# ---------------------------------------------------------------------------

class TestLoadLargeModel:
    @patch("main.PeftModel")
    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_loads_base_model_and_adapter(
        self, mock_proc_cls, mock_model_cls, mock_peft_cls
    ):
        mock_base = MagicMock()
        mock_base.to.return_value = mock_base
        mock_model_cls.from_pretrained.return_value = mock_base
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        mock_peft_model = MagicMock()
        mock_peft_model.eval.return_value = mock_peft_model
        mock_peft_cls.from_pretrained.return_value = mock_peft_model

        model, processor = load_large_model("cpu", adapter_path="/fake/adapter")

        # Should load base Whisper-large-v3
        mock_model_cls.from_pretrained.assert_called_once()
        base_call = mock_model_cls.from_pretrained.call_args
        assert "whisper-large-v3" in base_call[0][0]

        # Should load LoRA adapter
        mock_peft_cls.from_pretrained.assert_called_once()

    @patch("main.PeftModel")
    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_returns_model_and_processor(
        self, mock_proc_cls, mock_model_cls, mock_peft_cls
    ):
        mock_base = MagicMock()
        mock_base.to.return_value = mock_base
        mock_model_cls.from_pretrained.return_value = mock_base
        mock_proc_cls.from_pretrained.return_value = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.eval.return_value = mock_peft_model
        mock_peft_cls.from_pretrained.return_value = mock_peft_model

        model, processor = load_large_model("cpu", adapter_path="/fake/adapter")
        assert model is not None
        assert processor is not None

    @patch("main.PeftModel")
    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_model_set_to_eval(
        self, mock_proc_cls, mock_model_cls, mock_peft_cls
    ):
        mock_base = MagicMock()
        mock_base.to.return_value = mock_base
        mock_model_cls.from_pretrained.return_value = mock_base
        mock_proc_cls.from_pretrained.return_value = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.eval.return_value = mock_peft_model
        mock_peft_cls.from_pretrained.return_value = mock_peft_model

        model, _ = load_large_model("cpu", adapter_path="/fake/adapter")
        mock_peft_model.eval.assert_called_once()

    @patch("main.PeftModel")
    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_uses_fp16_on_cuda(
        self, mock_proc_cls, mock_model_cls, mock_peft_cls
    ):
        mock_base = MagicMock()
        mock_base.to.return_value = mock_base
        mock_model_cls.from_pretrained.return_value = mock_base
        mock_proc_cls.from_pretrained.return_value = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.eval.return_value = mock_peft_model
        mock_peft_cls.from_pretrained.return_value = mock_peft_model

        load_large_model("cuda", adapter_path="/fake/adapter")
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        import torch
        assert call_kwargs.get("torch_dtype") == torch.float16


# ---------------------------------------------------------------------------
# Ensemble integration: run_ensemble_inference
# ---------------------------------------------------------------------------

class TestRunEnsembleInference:
    @patch("main.check_time_budget", return_value=True)
    @patch("main.run_inference")
    @patch("main.load_model")
    @patch("main.load_large_model")
    def test_runs_both_models_when_time_allows(
        self,
        mock_load_large,
        mock_load_small,
        mock_run_inf,
        mock_budget,
        sample_utterances,
    ):
        from main import run_ensemble_inference

        mock_model_a = MagicMock()
        mock_proc_a = MagicMock()
        mock_load_large.return_value = (mock_model_a, mock_proc_a)

        mock_model_b = MagicMock()
        mock_proc_b = MagicMock()
        mock_load_small.return_value = (mock_model_b, mock_proc_b)

        preds_a = {"utt_001": "hello", "utt_002": "world", "utt_003": "test"}
        preds_b = {"utt_001": "hi", "utt_002": "earth", "utt_003": "exam"}
        mock_run_inf.side_effect = [preds_a, preds_b]

        run_ensemble_inference(
            utterances=sample_utterances,
            data_dir=Path("/fake"),
            device="cpu",
            adapter_path="/fake/adapter",
            small_model_path="openai/whisper-small",
        )

        assert mock_load_large.call_count == 1
        assert mock_load_small.call_count == 1
        assert mock_run_inf.call_count == 2

    @patch("main.check_time_budget", return_value=False)
    @patch("main.run_inference")
    @patch("main.load_large_model")
    def test_skips_model_b_when_no_time(
        self,
        mock_load_large,
        mock_run_inf,
        mock_budget,
        sample_utterances,
    ):
        from main import run_ensemble_inference

        mock_model_a = MagicMock()
        mock_proc_a = MagicMock()
        mock_load_large.return_value = (mock_model_a, mock_proc_a)

        preds_a = {"utt_001": "hello", "utt_002": "world", "utt_003": "test"}
        mock_run_inf.return_value = preds_a

        result = run_ensemble_inference(
            utterances=sample_utterances,
            data_dir=Path("/fake"),
            device="cpu",
            adapter_path="/fake/adapter",
            small_model_path="openai/whisper-small",
        )

        assert mock_run_inf.call_count == 1
        # Result should still contain all utterance ids
        for u in sample_utterances:
            assert u["utterance_id"] in result

    @patch("main.check_time_budget", return_value=True)
    @patch("main.run_inference")
    @patch("main.load_model")
    @patch("main.load_large_model")
    def test_merges_predictions(
        self,
        mock_load_large,
        mock_load_small,
        mock_run_inf,
        mock_budget,
        sample_utterances,
    ):
        from main import run_ensemble_inference

        mock_load_large.return_value = (MagicMock(), MagicMock())
        mock_load_small.return_value = (MagicMock(), MagicMock())

        preds_a = {"utt_001": "", "utt_002": "world", "utt_003": ""}
        preds_b = {"utt_001": "fallback", "utt_002": "other", "utt_003": "backup"}
        mock_run_inf.side_effect = [preds_a, preds_b]

        result = run_ensemble_inference(
            utterances=sample_utterances,
            data_dir=Path("/fake"),
            device="cpu",
            adapter_path="/fake/adapter",
            small_model_path="openai/whisper-small",
        )

        # Model A empty → fallback to Model B
        assert result["utt_001"] == "fallback"
        # Model A non-empty → keep Model A
        assert result["utt_002"] == "world"
        assert result["utt_003"] == "backup"


# ---------------------------------------------------------------------------
# Backward compatibility: no adapter → small-only
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    @patch("main.run_inference")
    @patch("main.load_model")
    @patch("main.load_large_model")
    def test_falls_back_to_small_only_when_adapter_missing(
        self,
        mock_load_large,
        mock_load_small,
        mock_run_inf,
        sample_utterances,
    ):
        from main import run_ensemble_inference

        # Simulate adapter loading failure
        mock_load_large.side_effect = FileNotFoundError("No adapter found")

        mock_model_b = MagicMock()
        mock_proc_b = MagicMock()
        mock_load_small.return_value = (mock_model_b, mock_proc_b)

        preds_b = {"utt_001": "hello", "utt_002": "world", "utt_003": "test"}
        mock_run_inf.return_value = preds_b

        result = run_ensemble_inference(
            utterances=sample_utterances,
            data_dir=Path("/fake"),
            device="cpu",
            adapter_path="/nonexistent/adapter",
            small_model_path="openai/whisper-small",
        )

        # Should still produce results via small model
        for u in sample_utterances:
            assert u["utterance_id"] in result
        assert result["utt_001"] == "hello"
