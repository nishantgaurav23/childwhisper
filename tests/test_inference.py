"""Tests for S1.4 + S2.4 — Inference pipeline (submission/main.py).

S1.4: Zero-shot inference pipeline.
S2.4: Inference with fine-tuned Whisper-small weights.

All model loading and audio I/O is mocked. No real Whisper model or audio files needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure submission/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))

from main import (  # noqa: E402
    get_device,
    load_metadata,
    load_model,
    resolve_model_path,
    run_inference,
    transcribe_batch,
    write_submission,
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
def metadata_jsonl(tmp_path, sample_utterances):
    """Write sample metadata to a JSONL file and return parent dir."""
    meta_file = tmp_path / "utterance_metadata.jsonl"
    lines = [json.dumps(u) for u in sample_utterances]
    meta_file.write_text("\n".join(lines))
    return tmp_path


@pytest.fixture
def mock_audio():
    """Return a 1-second 16 kHz sine wave (non-silent)."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def silent_audio():
    """Return near-silent audio (RMS < -40 dB)."""
    return np.zeros(16000, dtype=np.float32)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_string(self):
        device = get_device()
        assert isinstance(device, str)

    def test_returns_valid_device(self):
        device = get_device()
        assert device in ("cuda", "mps", "cpu")


# ---------------------------------------------------------------------------
# load_metadata
# ---------------------------------------------------------------------------

class TestLoadMetadata:
    def test_loads_jsonl(self, metadata_jsonl):
        result = load_metadata(metadata_jsonl)
        assert len(result) == 3
        assert result[0]["utterance_id"] == "utt_001"

    def test_returns_list_of_dicts(self, metadata_jsonl):
        result = load_metadata(metadata_jsonl)
        assert all(isinstance(r, dict) for r in result)

    def test_empty_file(self, tmp_path):
        meta_file = tmp_path / "utterance_metadata.jsonl"
        meta_file.write_text("")
        result = load_metadata(tmp_path)
        assert result == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_metadata(tmp_path / "nonexistent_dir")


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_returns_model_and_processor(self, mock_proc_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_proc

        model, processor = load_model("cpu")
        assert model is not None
        assert processor is not None

    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_loads_whisper_small(self, mock_proc_cls, mock_model_cls):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        load_model("cpu")
        mock_model_cls.from_pretrained.assert_called_once()
        call_args = mock_model_cls.from_pretrained.call_args
        assert "whisper-small" in call_args[0][0]


# ---------------------------------------------------------------------------
# transcribe_batch
# ---------------------------------------------------------------------------

class TestTranscribeBatch:
    def test_returns_list_of_strings(self, mock_audio):
        model = MagicMock()
        processor = MagicMock()

        # Mock processor to return tensor-like object
        mock_features = MagicMock()
        mock_features.input_features = MagicMock()
        mock_features.input_features.to.return_value = mock_features.input_features
        processor.return_value = mock_features

        # Mock model.generate to return sequences
        mock_output = MagicMock()
        mock_output.sequences = [[1, 2, 3]]
        model.generate.return_value = mock_output

        # Mock batch_decode
        processor.batch_decode.return_value = ["hello world"]

        result = transcribe_batch(model, processor, [mock_audio], "cpu")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_empty_batch_returns_empty_list(self):
        model = MagicMock()
        processor = MagicMock()

        result = transcribe_batch(model, processor, [], "cpu")
        assert result == []


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

class TestRunInference:
    @patch("main.transcribe_batch")
    @patch("main.load_audio")
    def test_returns_dict_with_all_ids(
        self, mock_load, mock_transcribe, sample_utterances
    ):
        mock_load.return_value = (np.ones(16000, dtype=np.float32), 16000)
        # Return one prediction per audio in the batch
        mock_transcribe.side_effect = lambda m, p, audios, d: ["hello"] * len(audios)

        model = MagicMock()
        processor = MagicMock()
        data_dir = Path("/fake")

        result = run_inference(
            model, processor, sample_utterances, data_dir, "cpu", batch_size=16
        )
        assert isinstance(result, dict)
        for u in sample_utterances:
            assert u["utterance_id"] in result

    @patch("main.transcribe_batch")
    @patch("main.load_audio")
    def test_sorts_by_duration_longest_first(
        self, mock_load, mock_transcribe, sample_utterances
    ):
        """Verify utterances are processed sorted by duration (longest first)."""
        processed_order = []

        def track_load(path, target_sr=16000):
            # Extract utterance ID from path to track order
            processed_order.append(str(path))
            return np.ones(16000, dtype=np.float32), 16000

        mock_load.side_effect = track_load
        mock_transcribe.side_effect = lambda m, p, audios, d: ["text"] * len(audios)

        model = MagicMock()
        processor = MagicMock()

        run_inference(
            model, processor, sample_utterances, Path("/fake"), "cpu", batch_size=1
        )

        # With batch_size=1, utt_003 (5.0s) should be first, then utt_001 (2.5s)
        assert "utt_003" in processed_order[0]
        assert "utt_001" in processed_order[1]
        assert "utt_002" in processed_order[2]

    @patch("main.transcribe_batch")
    @patch("main.load_audio")
    @patch("main.is_silence")
    def test_returns_empty_for_silent_audio(
        self, mock_silence, mock_load, mock_transcribe, sample_utterances
    ):
        mock_load.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_silence.return_value = True
        mock_transcribe.return_value = ["some hallucination"]

        model = MagicMock()
        processor = MagicMock()

        result = run_inference(
            model, processor, sample_utterances[:1], Path("/fake"), "cpu"
        )
        assert result[sample_utterances[0]["utterance_id"]] == ""

    @patch("main.transcribe_batch")
    @patch("main.load_audio")
    @patch("main.is_silence")
    def test_normalizes_text(
        self, mock_silence, mock_load, mock_transcribe, sample_utterances
    ):
        """Predictions should be normalized (e.g., lowercase, no punctuation)."""
        mock_load.return_value = (np.ones(16000, dtype=np.float32), 16000)
        mock_silence.return_value = False
        mock_transcribe.return_value = ["Hello, World!"]

        model = MagicMock()
        processor = MagicMock()

        result = run_inference(
            model, processor, sample_utterances[:1], Path("/fake"), "cpu"
        )
        text = result[sample_utterances[0]["utterance_id"]]
        # EnglishTextNormalizer lowercases and removes punctuation
        assert text == text.lower()
        assert "," not in text
        assert "!" not in text


# ---------------------------------------------------------------------------
# write_submission
# ---------------------------------------------------------------------------

class TestWriteSubmission:
    def test_writes_valid_jsonl(self, tmp_path, sample_utterances):
        predictions = {
            "utt_001": "hello",
            "utt_002": "world",
            "utt_003": "test",
        }
        output_path = write_submission(predictions, sample_utterances, tmp_path)

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            data = json.loads(line)
            assert "utterance_id" in data
            assert "orthographic_text" in data

    def test_includes_all_utterance_ids(self, tmp_path, sample_utterances):
        predictions = {
            "utt_001": "hello",
            "utt_002": "world",
            "utt_003": "",
        }
        output_path = write_submission(predictions, sample_utterances, tmp_path)

        lines = output_path.read_text().strip().split("\n")
        written_ids = {json.loads(line)["utterance_id"] for line in lines}
        expected_ids = {u["utterance_id"] for u in sample_utterances}
        assert written_ids == expected_ids

    def test_empty_prediction_writes_empty_string(self, tmp_path, sample_utterances):
        predictions = {
            "utt_001": "",
            "utt_002": "",
            "utt_003": "",
        }
        output_path = write_submission(predictions, sample_utterances, tmp_path)

        lines = output_path.read_text().strip().split("\n")
        for line in lines:
            data = json.loads(line)
            assert data["orthographic_text"] == ""

    def test_creates_output_dir(self, tmp_path, sample_utterances):
        output_dir = tmp_path / "new_sub_dir"
        predictions = {"utt_001": "a", "utt_002": "b", "utt_003": "c"}
        output_path = write_submission(predictions, sample_utterances, output_dir)
        assert output_path.exists()

    def test_missing_prediction_defaults_to_empty(self, tmp_path, sample_utterances):
        """If a prediction is missing for an utterance, default to empty string."""
        predictions = {"utt_001": "hello"}  # missing utt_002 and utt_003
        output_path = write_submission(predictions, sample_utterances, tmp_path)

        lines = output_path.read_text().strip().split("\n")
        for line in lines:
            data = json.loads(line)
            if data["utterance_id"] in ("utt_002", "utt_003"):
                assert data["orthographic_text"] == ""


# ===========================================================================
# S2.4 — Inference with Fine-Tuned Whisper-small
# ===========================================================================

# ---------------------------------------------------------------------------
# resolve_model_path
# ---------------------------------------------------------------------------

class TestResolveModelPath:
    def test_returns_finetuned_path_when_exists(self, tmp_path):
        """When fine-tuned weights dir exists, return its path."""
        ft_dir = tmp_path / "model_weights" / "whisper_small_ft"
        ft_dir.mkdir(parents=True)
        # Must contain a config file to be a valid model dir
        (ft_dir / "config.json").write_text("{}")

        result = resolve_model_path(ft_dir)
        assert result == str(ft_dir)

    def test_returns_default_when_finetuned_missing(self, tmp_path):
        """When fine-tuned dir doesn't exist, return default HF model ID."""
        nonexistent = tmp_path / "model_weights" / "whisper_small_ft"
        result = resolve_model_path(nonexistent)
        assert result == "openai/whisper-small"

    def test_returns_default_when_dir_empty(self, tmp_path):
        """When dir exists but has no config.json, return default."""
        ft_dir = tmp_path / "model_weights" / "whisper_small_ft"
        ft_dir.mkdir(parents=True)
        # No config.json inside
        result = resolve_model_path(ft_dir)
        assert result == "openai/whisper-small"

    def test_accepts_string_path(self, tmp_path):
        """Should accept string paths, not just Path objects."""
        ft_dir = tmp_path / "model_weights" / "whisper_small_ft"
        ft_dir.mkdir(parents=True)
        (ft_dir / "config.json").write_text("{}")

        result = resolve_model_path(str(ft_dir))
        assert result == str(ft_dir)

    def test_custom_default(self, tmp_path):
        """Should use custom default when provided and path doesn't exist."""
        nonexistent = tmp_path / "nonexistent"
        result = resolve_model_path(nonexistent, default="my-org/custom-model")
        assert result == "my-org/custom-model"


# ---------------------------------------------------------------------------
# load_model (S2.4 extensions)
# ---------------------------------------------------------------------------

class TestLoadModelFineTuned:
    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_loads_from_custom_path(self, mock_proc_cls, mock_model_cls):
        """load_model loads from specified model_name_or_path."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        load_model("cpu", model_name_or_path="/fake/finetuned")
        call_args = mock_model_cls.from_pretrained.call_args
        assert call_args[0][0] == "/fake/finetuned"

    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_processor_loaded_from_model_path(self, mock_proc_cls, mock_model_cls):
        """Processor should be loaded from same path as model."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        load_model("cpu", model_name_or_path="my-org/finetuned-whisper")
        proc_call_args = mock_proc_cls.from_pretrained.call_args
        assert proc_call_args[0][0] == "my-org/finetuned-whisper"

    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_defaults_to_whisper_small(self, mock_proc_cls, mock_model_cls):
        """Without model_name_or_path, defaults to openai/whisper-small."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        load_model("cpu")
        call_args = mock_model_cls.from_pretrained.call_args
        assert call_args[0][0] == "openai/whisper-small"

    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_processor_fallback_on_error(self, mock_proc_cls, mock_model_cls):
        """If processor load fails from model path, fall back to openai/whisper-small."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        # First call (from model path) raises, second call (fallback) succeeds
        mock_proc_cls.from_pretrained.side_effect = [
            OSError("No processor found"),
            MagicMock(),
        ]

        model, processor = load_model("cpu", model_name_or_path="/fake/finetuned")
        assert processor is not None
        # Should have tried twice: once with model path, once with fallback
        assert mock_proc_cls.from_pretrained.call_count == 2
        fallback_args = mock_proc_cls.from_pretrained.call_args_list[1]
        assert fallback_args[0][0] == "openai/whisper-small"

    @patch("main.WhisperForConditionalGeneration")
    @patch("main.WhisperProcessor")
    def test_model_set_to_eval(self, mock_proc_cls, mock_model_cls):
        """Model should be in eval mode."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        model, _ = load_model("cpu", model_name_or_path="/fake/finetuned")
        mock_model.eval.assert_called_once()
