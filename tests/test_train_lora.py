"""Tests for src/train_whisper_lora.py — LoRA configuration & training script.

Tests follow TDD: written BEFORE the implementation.
All model/processor loading is mocked to avoid downloading weights.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_path(tmp_path):
    """Write a minimal training config YAML and return its path."""
    config = {
        "common": {
            "sample_rate": 16000,
            "max_duration_sec": 30.0,
            "min_duration_sec": 0.3,
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
        "whisper_large_v3": {
            "model_name": "openai/whisper-large-v3",
            "learning_rate": 1.0e-3,
            "warmup_steps": 500,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 16,
            "fp16": True,
            "gradient_checkpointing": True,
            "load_in_8bit": True,
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 3,
            "generation_max_length": 225,
            "hub_model_id": "nishantgaurav23/pasketti-whisper-lora",
            "hub_private_repo": True,
            "lora": {
                "r": 32,
                "alpha": 64,
                "target_modules": ["q_proj", "v_proj"],
                "dropout": 0.05,
                "bias": "none",
                "task_type": "SEQ_2_SEQ_LM",
            },
        },
    }
    path = tmp_path / "training_config.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


@pytest.fixture
def metadata_dir(tmp_path):
    """Create a minimal metadata JSONL and a fake audio dir."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    entries = []
    for i in range(6):
        uid = f"utt_{i:03d}"
        child = f"child_{i % 3}"
        age = "5-7" if i % 2 == 0 else "8-11"
        entry = {
            "utterance_id": uid,
            "child_id": child,
            "session_id": "sess_0",
            "audio_path": f"{uid}.flac",
            "audio_duration_sec": 2.0,
            "age_bucket": age,
            "orthographic_text": f"word {i}",
        }
        entries.append(entry)

        # Create a minimal FLAC-like file (will be mocked in dataset)
        (audio_dir / f"{uid}.flac").write_bytes(b"\x00" * 100)

    meta_path = tmp_path / "metadata.jsonl"
    meta_path.write_text("\n".join(json.dumps(e) for e in entries))
    return str(meta_path), str(audio_dir)


# ---------------------------------------------------------------------------
# Test: Config Loading
# ---------------------------------------------------------------------------

class TestLoadLoraConfig:
    def test_merges_common_and_lora_sections(self, config_path):
        from src.train_whisper_lora import load_training_config

        config = load_training_config(config_path)

        # From common section
        assert config["sample_rate"] == 16000
        assert config["spec_augment"]["apply"] is True

        # From whisper_large_v3 section
        assert config["model_name"] == "openai/whisper-large-v3"
        assert config["learning_rate"] == 1e-3
        assert config["load_in_8bit"] is True
        assert config["lora"]["r"] == 32
        assert config["lora"]["alpha"] == 64
        assert config["lora"]["target_modules"] == ["q_proj", "v_proj"]

    def test_file_not_found_raises(self):
        from src.train_whisper_lora import load_training_config

        with pytest.raises(FileNotFoundError):
            load_training_config("/nonexistent/config.yaml")


# ---------------------------------------------------------------------------
# Test: CLI Argument Parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        from src.train_whisper_lora import parse_args

        args = parse_args([
            "--metadata-path", "data/meta.jsonl",
            "--audio-dir", "data/audio",
        ])
        assert args.metadata_path == "data/meta.jsonl"
        assert args.audio_dir == "data/audio"
        assert args.config == "configs/training_config.yaml"
        assert args.output_dir == "./checkpoints/whisper-large-v3-lora"
        assert args.dry_run is False
        assert args.push_to_hub is True

    def test_dry_run_flag(self):
        from src.train_whisper_lora import parse_args

        args = parse_args([
            "--metadata-path", "m.jsonl",
            "--audio-dir", "a/",
            "--dry-run",
            "--no-push-to-hub",
        ])
        assert args.dry_run is True
        assert args.push_to_hub is False


# ---------------------------------------------------------------------------
# Test: LoRA Config Setup
# ---------------------------------------------------------------------------

class TestSetupLoraConfig:
    def test_creates_lora_config_with_correct_params(self, config_path):
        from src.train_whisper_lora import load_training_config, create_lora_config

        config = load_training_config(config_path)
        lora_cfg = create_lora_config(config)

        assert lora_cfg.r == 32
        assert lora_cfg.lora_alpha == 64
        assert set(lora_cfg.target_modules) == {"q_proj", "v_proj"}
        assert lora_cfg.lora_dropout == 0.05
        assert lora_cfg.bias == "none"
        assert lora_cfg.task_type == "SEQ_2_SEQ_LM"


# ---------------------------------------------------------------------------
# Test: Model Setup
# ---------------------------------------------------------------------------

class TestSetupModel:
    @patch("src.train_whisper_lora.get_peft_model")
    @patch("src.train_whisper_lora.prepare_model_for_kbit_training")
    @patch("src.train_whisper_lora.WhisperProcessor.from_pretrained")
    @patch("src.train_whisper_lora.WhisperForConditionalGeneration.from_pretrained")
    def test_spec_augment_enabled(
        self, mock_model_cls, mock_proc_cls, mock_prep_kbit, mock_get_peft, config_path
    ):
        from src.train_whisper_lora import load_training_config, setup_model

        config = load_training_config(config_path)

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.generation_config = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_prep_kbit.return_value = mock_model
        mock_get_peft.return_value = mock_model
        mock_proc_cls.return_value = MagicMock()

        # Don't use INT8 in test (no GPU)
        config["load_in_8bit"] = False

        model, processor = setup_model(config)

        assert mock_model.config.apply_spec_augment is True
        assert mock_model.config.mask_time_prob == 0.05
        assert mock_model.config.mask_feature_prob == 0.04

    @patch("src.train_whisper_lora.get_peft_model")
    @patch("src.train_whisper_lora.prepare_model_for_kbit_training")
    @patch("src.train_whisper_lora.WhisperProcessor.from_pretrained")
    @patch("src.train_whisper_lora.WhisperForConditionalGeneration.from_pretrained")
    def test_forced_decoder_ids_cleared(
        self, mock_model_cls, mock_proc_cls, mock_prep_kbit, mock_get_peft, config_path
    ):
        from src.train_whisper_lora import load_training_config, setup_model

        config = load_training_config(config_path)
        config["load_in_8bit"] = False

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.generation_config = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_prep_kbit.return_value = mock_model
        mock_get_peft.return_value = mock_model
        mock_proc_cls.return_value = MagicMock()

        setup_model(config)

        assert mock_model.config.forced_decoder_ids is None
        assert mock_model.config.suppress_tokens == []
        assert mock_model.generation_config.forced_decoder_ids is None

    @patch("src.train_whisper_lora.get_peft_model")
    @patch("src.train_whisper_lora.prepare_model_for_kbit_training")
    @patch("src.train_whisper_lora.WhisperProcessor.from_pretrained")
    @patch("src.train_whisper_lora.WhisperForConditionalGeneration.from_pretrained")
    def test_lora_applied(
        self, mock_model_cls, mock_proc_cls, mock_prep_kbit, mock_get_peft, config_path
    ):
        from src.train_whisper_lora import load_training_config, setup_model

        config = load_training_config(config_path)
        config["load_in_8bit"] = False

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.generation_config = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_prep_kbit.return_value = mock_model
        mock_get_peft.return_value = mock_model
        mock_proc_cls.return_value = MagicMock()

        setup_model(config)

        # get_peft_model should have been called
        mock_get_peft.assert_called_once()
        # The LoRA config should have correct params
        lora_cfg = mock_get_peft.call_args[0][1]
        assert lora_cfg.r == 32
        assert lora_cfg.lora_alpha == 64


# ---------------------------------------------------------------------------
# Test: Training Arguments
# ---------------------------------------------------------------------------

class TestSetupTrainingArgs:
    def test_lora_defaults(self, config_path):
        from src.train_whisper_lora import load_training_config, setup_training_args

        config = load_training_config(config_path)

        args = setup_training_args(config, output_dir="/tmp/test", push_to_hub=False)

        assert args.learning_rate == 1e-3
        assert args.per_device_train_batch_size == 1
        assert args.gradient_accumulation_steps == 16
        assert args.predict_with_generate is True
        assert args.load_best_model_at_end is True
        assert args.metric_for_best_model == "wer"
        assert args.greater_is_better is False

    def test_dry_run_overrides(self, config_path):
        from src.train_whisper_lora import load_training_config, setup_training_args

        config = load_training_config(config_path)

        args = setup_training_args(
            config, output_dir="/tmp/test", push_to_hub=False, dry_run=True
        )

        assert args.max_steps == 1
        assert args.eval_steps == 1
        assert args.save_steps == 1
        assert args.push_to_hub is False
        assert args.fp16 is False
        assert args.gradient_checkpointing is False


# ---------------------------------------------------------------------------
# Test: Build Datasets
# ---------------------------------------------------------------------------

class TestBuildDatasets:
    @patch("src.train_whisper_lora.WhisperDataset")
    def test_returns_train_and_val(self, mock_ds_cls, config_path, metadata_dir):
        from src.train_whisper_lora import load_training_config, build_datasets

        config = load_training_config(config_path)
        meta_path, audio_dir = metadata_dir

        mock_ds_cls.return_value = MagicMock()

        train_ds, val_ds = build_datasets(config, meta_path, audio_dir)

        # Should create two datasets (train + val)
        assert mock_ds_cls.call_count == 2
        # First call = train, second = val
        train_call_kwargs = mock_ds_cls.call_args_list[0][1]
        val_call_kwargs = mock_ds_cls.call_args_list[1][1]

        assert train_call_kwargs["model_name"] == "openai/whisper-large-v3"
        assert val_call_kwargs["augment_fn"] is None


# ---------------------------------------------------------------------------
# Test: compute_metrics
# ---------------------------------------------------------------------------

class TestMakeComputeMetrics:
    def test_computes_wer(self):
        from src.train_whisper_lora import make_compute_metrics

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.batch_decode = MagicMock(
            side_effect=[
                ["hello world", "testing"],  # predictions
                ["hello world", "testing"],  # labels
            ]
        )

        compute_fn = make_compute_metrics(mock_tokenizer)

        pred = MagicMock()
        pred.predictions = [[1, 2], [3, 4]]
        pred.label_ids = [[1, 2], [3, 4]]

        result = compute_fn(pred)
        assert "wer" in result
        assert isinstance(result["wer"], float)
