"""Tests for S5.2 — Inference Optimization (Faster Inference).

Tests for: dynamic batch sizing, SDPA attention loading, torch.compile guards,
enhanced beam search config, and backward compatibility with existing pipeline.

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


# ---------------------------------------------------------------------------
# get_optimal_batch_size
# ---------------------------------------------------------------------------

class TestGetOptimalBatchSize:
    def test_cuda_large_model(self):
        from main import get_optimal_batch_size
        bs = get_optimal_batch_size("cuda", model_size="large")
        assert bs == 32

    def test_cuda_small_model(self):
        from main import get_optimal_batch_size
        bs = get_optimal_batch_size("cuda", model_size="small")
        assert bs == 64

    def test_cpu_large_model(self):
        from main import get_optimal_batch_size
        bs = get_optimal_batch_size("cpu", model_size="large")
        assert bs == 4

    def test_cpu_small_model(self):
        from main import get_optimal_batch_size
        bs = get_optimal_batch_size("cpu", model_size="small")
        assert bs == 4

    def test_mps_large_model(self):
        from main import get_optimal_batch_size
        bs = get_optimal_batch_size("mps", model_size="large")
        assert bs == 16

    def test_mps_small_model(self):
        from main import get_optimal_batch_size
        bs = get_optimal_batch_size("mps", model_size="small")
        assert bs == 16

    def test_unknown_device_returns_safe_default(self):
        from main import get_optimal_batch_size
        bs = get_optimal_batch_size("xpu", model_size="large")
        assert bs == 4

    def test_unknown_model_size_returns_conservative(self):
        from main import get_optimal_batch_size
        bs = get_optimal_batch_size("cuda", model_size="unknown")
        assert bs == 16


# ---------------------------------------------------------------------------
# SDPA attention implementation
# ---------------------------------------------------------------------------

class TestSDPAAttention:
    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_load_model_requests_sdpa(self, mock_proc_cls, mock_model_cls):
        """load_model should request attn_implementation='sdpa'."""
        from main import load_model

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        load_model("cpu")

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs.get("attn_implementation") == "sdpa"

    @patch("main.PeftModel")
    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_load_large_model_requests_sdpa(
        self, mock_proc_cls, mock_model_cls, mock_peft_cls
    ):
        """load_large_model should request attn_implementation='sdpa'."""
        from main import load_large_model

        mock_base = MagicMock()
        mock_base.to.return_value = mock_base
        mock_model_cls.from_pretrained.return_value = mock_base
        mock_proc_cls.from_pretrained.return_value = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.eval.return_value = mock_peft_model
        mock_peft_cls.from_pretrained.return_value = mock_peft_model

        load_large_model("cpu", adapter_path="/fake/adapter")

        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs.get("attn_implementation") == "sdpa"


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------

class TestMaybeCompile:
    def test_returns_model_on_cpu(self):
        """On CPU, should return model unchanged (no compile)."""
        from main import maybe_compile

        model = MagicMock()
        result = maybe_compile(model, "cpu")
        assert result is model

    def test_returns_model_on_mps(self):
        """On MPS, should return model unchanged (no compile)."""
        from main import maybe_compile

        model = MagicMock()
        result = maybe_compile(model, "mps")
        assert result is model

    @patch("torch.compile")
    def test_compiles_on_cuda(self, mock_compile):
        """On CUDA, should call torch.compile."""
        from main import maybe_compile

        model = MagicMock()
        compiled = MagicMock()
        mock_compile.return_value = compiled

        result = maybe_compile(model, "cuda")
        mock_compile.assert_called_once()
        assert result is compiled

    @patch("torch.compile", side_effect=Exception("compile failed"))
    def test_falls_back_on_compile_failure(self, mock_compile):
        """If torch.compile fails, return original model."""
        from main import maybe_compile

        model = MagicMock()
        result = maybe_compile(model, "cuda")
        assert result is model


# ---------------------------------------------------------------------------
# Beam search config
# ---------------------------------------------------------------------------

class TestBeamConfig:
    def test_large_model_num_beams(self):
        """Large model should use num_beams=8."""
        from main import get_beam_config
        config = get_beam_config(model_size="large")
        assert config["num_beams"] == 8

    def test_small_model_num_beams(self):
        """Small model should use num_beams=5."""
        from main import get_beam_config
        config = get_beam_config(model_size="small")
        assert config["num_beams"] == 5

    def test_config_has_length_penalty(self):
        """Config should include length_penalty."""
        from main import get_beam_config
        config = get_beam_config(model_size="large")
        assert "length_penalty" in config
        assert config["length_penalty"] == 1.0

    def test_config_has_max_new_tokens(self):
        """Config should include max_new_tokens."""
        from main import get_beam_config
        config = get_beam_config(model_size="large")
        assert "max_new_tokens" in config
        assert config["max_new_tokens"] == 225


# ---------------------------------------------------------------------------
# transcribe_batch with num_beams parameter
# ---------------------------------------------------------------------------

class TestTranscribeBatchBeams:
    def test_accepts_num_beams_parameter(self):
        """transcribe_batch should accept and forward num_beams."""
        from main import transcribe_batch

        model = MagicMock()
        processor = MagicMock()

        mock_features = MagicMock()
        mock_features.input_features = MagicMock()
        mock_features.input_features.to.return_value = mock_features.input_features
        processor.return_value = mock_features

        mock_output = MagicMock()
        model.generate.return_value = mock_output
        processor.batch_decode.return_value = ["hello"]

        audio = np.ones(16000, dtype=np.float32)
        transcribe_batch(model, processor, [audio], "cpu", num_beams=8)

        # Verify num_beams=8 was passed to model.generate
        gen_kwargs = model.generate.call_args[1]
        assert gen_kwargs["num_beams"] == 8

    def test_default_num_beams_is_5(self):
        """Without num_beams arg, should default to 5."""
        from main import transcribe_batch

        model = MagicMock()
        processor = MagicMock()

        mock_features = MagicMock()
        mock_features.input_features = MagicMock()
        mock_features.input_features.to.return_value = mock_features.input_features
        processor.return_value = mock_features

        mock_output = MagicMock()
        model.generate.return_value = mock_output
        processor.batch_decode.return_value = ["hello"]

        audio = np.ones(16000, dtype=np.float32)
        transcribe_batch(model, processor, [audio], "cpu")

        gen_kwargs = model.generate.call_args[1]
        assert gen_kwargs["num_beams"] == 5

    def test_accepts_length_penalty(self):
        """transcribe_batch should accept and forward length_penalty."""
        from main import transcribe_batch

        model = MagicMock()
        processor = MagicMock()

        mock_features = MagicMock()
        mock_features.input_features = MagicMock()
        mock_features.input_features.to.return_value = mock_features.input_features
        processor.return_value = mock_features

        mock_output = MagicMock()
        model.generate.return_value = mock_output
        processor.batch_decode.return_value = ["hello"]

        audio = np.ones(16000, dtype=np.float32)
        transcribe_batch(
            model, processor, [audio], "cpu", num_beams=8, length_penalty=1.0
        )

        gen_kwargs = model.generate.call_args[1]
        assert gen_kwargs["length_penalty"] == 1.0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    @patch("main.transcribe_batch")
    @patch("main.load_audio")
    def test_run_inference_still_works(
        self, mock_load, mock_transcribe, sample_utterances
    ):
        """run_inference should still work with new parameters."""
        from main import run_inference

        mock_load.return_value = (np.ones(16000, dtype=np.float32), 16000)
        mock_transcribe.side_effect = lambda m, p, audios, d, **kw: (
            ["hello"] * len(audios)
        )

        model = MagicMock()
        processor = MagicMock()

        result = run_inference(
            model, processor, sample_utterances, Path("/fake"), "cpu", batch_size=16
        )
        assert isinstance(result, dict)
        for u in sample_utterances:
            assert u["utterance_id"] in result

    @patch("main.check_time_budget", return_value=True)
    @patch("main.run_inference")
    @patch("main.load_model")
    @patch("main.load_large_model")
    def test_ensemble_still_works(
        self,
        mock_load_large,
        mock_load_small,
        mock_run_inf,
        mock_budget,
        sample_utterances,
    ):
        """run_ensemble_inference should still produce correct results."""
        from main import run_ensemble_inference

        mock_load_large.return_value = (MagicMock(), MagicMock())
        mock_load_small.return_value = (MagicMock(), MagicMock())

        preds_a = {"utt_001": "hello", "utt_002": "", "utt_003": "test"}
        preds_b = {"utt_001": "hi", "utt_002": "world", "utt_003": "exam"}
        mock_run_inf.side_effect = [preds_a, preds_b]

        result = run_ensemble_inference(
            utterances=sample_utterances,
            data_dir=Path("/fake"),
            device="cpu",
            adapter_path="/fake/adapter",
            small_model_path="openai/whisper-small",
        )

        assert result["utt_001"] == "hello"  # A non-empty, prefer A
        assert result["utt_002"] == "world"  # A empty, fallback to B
        assert result["utt_003"] == "test"   # A non-empty, prefer A
