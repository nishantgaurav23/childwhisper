"""Tests for Whisper-small training script (S2.2).

Tests config loading, argument parsing, model setup, training args,
WER metric computation, dataset building, and dry-run mode.
All model/training operations are mocked to run on CPU without real data.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_file(tmp_path):
    """Create a minimal training config YAML."""
    config = {
        "common": {
            "sample_rate": 16000,
            "max_duration_sec": 30.0,
            "min_duration_sec": 0.3,
            "silence_threshold_db": -40,
            "trim_top_db": 30,
            "spec_augment": {
                "apply": True,
                "mask_time_prob": 0.05,
                "mask_time_length": 10,
                "mask_feature_prob": 0.04,
                "mask_feature_length": 10,
            },
            "validation": {
                "split_ratio": 0.1,
                "stratify_by": "age_bucket",
                "split_by": "child_id",
            },
        },
        "whisper_small": {
            "model_name": "openai/whisper-small",
            "learning_rate": 1.0e-5,
            "warmup_steps": 500,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "gradient_checkpointing": True,
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 3,
            "generation_max_length": 225,
            "dataloader_num_workers": 4,
            "hub_model_id": "test-user/test-whisper-small",
            "hub_private_repo": True,
        },
    }
    config_path = tmp_path / "training_config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


@pytest.fixture
def sample_metadata(tmp_path):
    """Create sample metadata JSONL and dummy audio files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    entries = []
    for i in range(20):
        child_id = f"child_{i % 5}"
        age = ["3-4", "5-7", "8-11", "12+", "unknown"][i % 5]
        entry = {
            "utterance_id": f"utt_{i:03d}",
            "child_id": child_id,
            "session_id": f"sess_{i % 3}",
            "audio_path": f"utt_{i:03d}.flac",
            "audio_duration_sec": 2.0 + (i % 5) * 0.5,
            "age_bucket": age,
            "orthographic_text": f"hello world {i}",
        }
        entries.append(entry)
        # Create a tiny valid WAV-like file (won't be read in mocked tests)
        (audio_dir / f"utt_{i:03d}.flac").write_bytes(b"\x00" * 100)

    meta_path = tmp_path / "metadata.jsonl"
    meta_path.write_text("\n".join(json.dumps(e) for e in entries))
    return meta_path, audio_dir, entries


# ---------------------------------------------------------------------------
# Test: Config Loading
# ---------------------------------------------------------------------------

class TestLoadTrainingConfig:
    def test_loads_yaml_and_merges(self, config_file):
        from src.train_whisper_small import load_training_config

        config = load_training_config(str(config_file))

        # Should have both common and whisper_small keys merged
        assert config["model_name"] == "openai/whisper-small"
        assert config["learning_rate"] == 1.0e-5
        assert config["sample_rate"] == 16000
        assert config["spec_augment"]["apply"] is True
        assert config["validation"]["split_ratio"] == 0.1

    def test_whisper_small_overrides_common(self, config_file):
        from src.train_whisper_small import load_training_config

        config = load_training_config(str(config_file))
        # whisper_small section values should be present
        assert config["warmup_steps"] == 500
        assert config["per_device_train_batch_size"] == 8

    def test_missing_config_raises(self):
        from src.train_whisper_small import load_training_config

        with pytest.raises(FileNotFoundError):
            load_training_config("/nonexistent/config.yaml")


# ---------------------------------------------------------------------------
# Test: Argument Parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        from src.train_whisper_small import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
        ])
        assert args.metadata_path == "/data/meta.jsonl"
        assert args.audio_dir == "/data/audio"
        assert args.dry_run is False
        assert args.push_to_hub is True

    def test_dry_run_flag(self):
        from src.train_whisper_small import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
            "--dry-run",
        ])
        assert args.dry_run is True

    def test_disable_hub_push(self):
        from src.train_whisper_small import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
            "--no-push-to-hub",
        ])
        assert args.push_to_hub is False

    def test_override_epochs(self):
        from src.train_whisper_small import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
            "--num-train-epochs", "5",
        ])
        assert args.num_train_epochs == 5

    def test_override_output_dir(self):
        from src.train_whisper_small import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
            "--output-dir", "/tmp/output",
        ])
        assert args.output_dir == "/tmp/output"


# ---------------------------------------------------------------------------
# Test: Model Setup
# ---------------------------------------------------------------------------

class TestSetupModel:
    @patch("src.train_whisper_small.WhisperForConditionalGeneration")
    @patch("src.train_whisper_small.WhisperProcessor")
    def test_spec_augment_enabled(self, mock_processor_cls, mock_model_cls):
        from src.train_whisper_small import setup_model

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_processor_cls.from_pretrained.return_value = MagicMock()

        config = {
            "model_name": "openai/whisper-small",
            "fp16": True,
            "gradient_checkpointing": True,
            "spec_augment": {
                "apply": True,
                "mask_time_prob": 0.05,
                "mask_time_length": 10,
                "mask_feature_prob": 0.04,
                "mask_feature_length": 10,
            },
        }
        model, processor = setup_model(config)

        assert model.config.apply_spec_augment is True
        assert model.config.mask_time_prob == 0.05
        assert model.config.mask_feature_prob == 0.04
        mock_model.gradient_checkpointing_enable.assert_called_once()

    @patch("src.train_whisper_small.WhisperForConditionalGeneration")
    @patch("src.train_whisper_small.WhisperProcessor")
    def test_forced_decoder_ids_set(self, mock_processor_cls, mock_model_cls):
        from src.train_whisper_small import setup_model

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_processor = MagicMock()
        mock_processor_cls.from_pretrained.return_value = mock_processor

        config = {
            "model_name": "openai/whisper-small",
            "fp16": True,
            "gradient_checkpointing": True,
            "spec_augment": {"apply": True, "mask_time_prob": 0.05,
                             "mask_time_length": 10, "mask_feature_prob": 0.04,
                             "mask_feature_length": 10},
        }
        model, processor = setup_model(config)

        # forced_decoder_ids should be set to None (we use suppress_tokens instead)
        assert model.config.forced_decoder_ids is None
        # suppress_tokens should be set
        assert model.config.suppress_tokens is not None


# ---------------------------------------------------------------------------
# Test: Training Arguments
# ---------------------------------------------------------------------------

class TestSetupTrainingArgs:
    def test_args_match_config(self, tmp_path):
        from src.train_whisper_small import setup_training_args

        config = {
            "learning_rate": 1e-5,
            "warmup_steps": 500,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "gradient_checkpointing": True,
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 3,
            "generation_max_length": 225,
            "dataloader_num_workers": 4,
            "hub_model_id": "test-user/test-whisper-small",
            "hub_private_repo": True,
        }
        args = setup_training_args(
            config, output_dir=str(tmp_path / "output"), push_to_hub=False
        )

        assert args.learning_rate == 1e-5
        assert args.warmup_steps == 500
        assert args.num_train_epochs == 3
        assert args.per_device_train_batch_size == 8
        assert args.gradient_accumulation_steps == 4
        assert args.predict_with_generate is True
        assert args.push_to_hub is False

    def test_dry_run_overrides(self, tmp_path):
        from src.train_whisper_small import setup_training_args

        config = {
            "learning_rate": 1e-5,
            "warmup_steps": 500,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "gradient_checkpointing": True,
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 3,
            "generation_max_length": 225,
            "dataloader_num_workers": 4,
            "hub_model_id": "test-user/test-whisper-small",
            "hub_private_repo": True,
        }
        args = setup_training_args(
            config, output_dir=str(tmp_path / "output"),
            push_to_hub=False, dry_run=True,
        )

        # Dry run should override to minimal training
        assert args.max_steps == 1
        assert args.eval_steps == 1
        assert args.save_steps == 1
        assert args.logging_steps == 1
        assert args.push_to_hub is False


# ---------------------------------------------------------------------------
# Test: WER Metric Computation
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self):
        from src.train_whisper_small import make_compute_metrics

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        # batch_decode returns same text for both pred and label
        mock_tokenizer.batch_decode.return_value = ["hello world", "good morning"]

        compute_metrics = make_compute_metrics(mock_tokenizer)

        pred = MagicMock()
        pred.predictions = np.array([[1, 2, 3]])
        pred.label_ids = np.array([[1, 2, 3]])

        result = compute_metrics(pred)
        assert "wer" in result
        assert result["wer"] == 0.0

    def test_imperfect_predictions(self):
        from src.train_whisper_small import make_compute_metrics

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # First call is predictions, second is labels
        call_count = [0]

        def decode_side_effect(ids, skip_special_tokens=True):
            call_count[0] += 1
            if call_count[0] == 1:
                return ["goodbye world"]  # prediction
            return ["hello world"]  # reference

        mock_tokenizer.batch_decode.side_effect = decode_side_effect

        compute_metrics = make_compute_metrics(mock_tokenizer)

        pred = MagicMock()
        pred.predictions = np.array([[1, 2, 3]])
        pred.label_ids = np.array([[1, 2, 3]])

        result = compute_metrics(pred)
        assert "wer" in result
        assert result["wer"] > 0.0

    def test_replaces_neg100_in_labels(self):
        from src.train_whisper_small import make_compute_metrics

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.batch_decode.return_value = ["hello"]

        compute_metrics = make_compute_metrics(mock_tokenizer)

        pred = MagicMock()
        pred.predictions = np.array([[1, 2, 3]])
        pred.label_ids = np.array([[1, 2, -100]])  # -100 should be replaced

        result = compute_metrics(pred)
        # Should not crash — -100 replaced with pad_token_id before decode
        assert "wer" in result


# ---------------------------------------------------------------------------
# Test: Dataset Building
# ---------------------------------------------------------------------------

class TestBuildDatasets:
    @patch("src.train_whisper_small.WhisperDataset")
    @patch("src.train_whisper_small.create_train_val_split")
    @patch("src.train_whisper_small.load_metadata")
    def test_creates_train_val(self, mock_load, mock_split, mock_dataset):
        from src.train_whisper_small import build_datasets

        mock_load.return_value = [
            {"child_id": "c1", "age_bucket": "5-7", "utterance_id": "u1",
             "audio_path": "u1.flac", "orthographic_text": "hi",
             "audio_duration_sec": 2.0},
        ] * 20

        mock_split.return_value = (
            [{"child_id": "c1"}] * 18,
            [{"child_id": "c2"}] * 2,
        )

        mock_dataset.return_value = MagicMock()

        config = {
            "model_name": "openai/whisper-small",
            "sample_rate": 16000,
            "min_duration_sec": 0.3,
            "max_duration_sec": 30.0,
            "validation": {"split_ratio": 0.1, "split_by": "child_id",
                           "stratify_by": "age_bucket"},
        }

        train_ds, val_ds = build_datasets(
            config, "/data/meta.jsonl", "/data/audio"
        )

        mock_load.assert_called_once_with("/data/meta.jsonl")
        mock_split.assert_called_once()
        assert mock_dataset.call_count == 2  # train + val


# ---------------------------------------------------------------------------
# Test: Hub Push Disabled
# ---------------------------------------------------------------------------

class TestHubPushDisabled:
    def test_no_hub_when_disabled(self, tmp_path):
        from src.train_whisper_small import setup_training_args

        config = {
            "learning_rate": 1e-5,
            "warmup_steps": 500,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "gradient_checkpointing": True,
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 3,
            "generation_max_length": 225,
            "dataloader_num_workers": 4,
            "hub_model_id": "test-user/test-whisper-small",
            "hub_private_repo": True,
        }

        args = setup_training_args(
            config, output_dir=str(tmp_path / "output"), push_to_hub=False
        )
        assert args.push_to_hub is False
