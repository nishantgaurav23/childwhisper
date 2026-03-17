"""Tests for S4.3 — Retrain with Augmented Data.

Tests augmentation CLI args, config loading, augment_fn wiring,
error handling, noisy validation reporting, and notebook structure.
All model/training operations are mocked.
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
def augmented_config_file(tmp_path):
    """Create a training config YAML with augmentation section."""
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
            "augmentation": {
                "realclass_min_snr_db": 5.0,
                "realclass_max_snr_db": 20.0,
                "musan_min_snr_db": 0.0,
                "musan_max_snr_db": 15.0,
            },
        },
        "whisper_small": {
            "model_name": "openai/whisper-small",
            "learning_rate": 1.0e-5,
            "warmup_steps": 500,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "fp16": True,
            "gradient_checkpointing": True,
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 3,
            "generation_max_length": 225,
            "hub_model_id": "test-user/test-whisper-small",
            "hub_private_repo": True,
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
            "hub_model_id": "test-user/test-whisper-lora",
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
    config_path = tmp_path / "training_config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


@pytest.fixture
def noise_dirs(tmp_path):
    """Create fake noise directories with dummy audio files."""
    noise_dir = tmp_path / "musan_noise"
    noise_dir.mkdir()
    (noise_dir / "noise_001.wav").write_bytes(b"\x00" * 100)

    realclass_dir = tmp_path / "realclass_noise"
    realclass_dir.mkdir()
    (realclass_dir / "class_001.wav").write_bytes(b"\x00" * 100)

    return str(noise_dir), str(realclass_dir)


# ---------------------------------------------------------------------------
# Test: Augmentation CLI Args — Whisper-small
# ---------------------------------------------------------------------------

class TestSmallAugmentationArgs:
    def test_noise_dir_args_accepted(self):
        from src.train_whisper_small import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
            "--noise-dir", "/data/musan",
            "--realclass-dir", "/data/realclass",
        ])
        assert args.noise_dir == "/data/musan"
        assert args.realclass_dir == "/data/realclass"

    def test_noise_dir_args_default_none(self):
        from src.train_whisper_small import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
        ])
        assert args.noise_dir is None
        assert args.realclass_dir is None

    def test_hub_model_id_override(self):
        from src.train_whisper_small import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
            "--hub-model-id", "user/custom-model",
        ])
        assert args.hub_model_id == "user/custom-model"

    def test_hub_model_id_default_none(self):
        from src.train_whisper_small import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
        ])
        assert args.hub_model_id is None


# ---------------------------------------------------------------------------
# Test: Augmentation CLI Args — Whisper-large-v3 LoRA
# ---------------------------------------------------------------------------

class TestLoraAugmentationArgs:
    def test_noise_dir_args_accepted(self):
        from src.train_whisper_lora import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
            "--noise-dir", "/data/musan",
            "--realclass-dir", "/data/realclass",
        ])
        assert args.noise_dir == "/data/musan"
        assert args.realclass_dir == "/data/realclass"

    def test_noise_dir_args_default_none(self):
        from src.train_whisper_lora import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
        ])
        assert args.noise_dir is None
        assert args.realclass_dir is None

    def test_hub_model_id_override(self):
        from src.train_whisper_lora import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
            "--hub-model-id", "user/custom-lora",
        ])
        assert args.hub_model_id == "user/custom-lora"

    def test_hub_model_id_default_none(self):
        from src.train_whisper_lora import parse_args

        args = parse_args([
            "--metadata-path", "/data/meta.jsonl",
            "--audio-dir", "/data/audio",
        ])
        assert args.hub_model_id is None


# ---------------------------------------------------------------------------
# Test: Augmentation Config Loading
# ---------------------------------------------------------------------------

class TestAugmentationConfig:
    def test_small_config_has_augmentation(self, augmented_config_file):
        from src.train_whisper_small import load_training_config

        config = load_training_config(str(augmented_config_file))
        aug = config.get("augmentation", {})
        assert aug["realclass_min_snr_db"] == 5.0
        assert aug["realclass_max_snr_db"] == 20.0
        assert aug["musan_min_snr_db"] == 0.0
        assert aug["musan_max_snr_db"] == 15.0

    def test_lora_config_has_augmentation(self, augmented_config_file):
        from src.train_whisper_lora import load_training_config

        config = load_training_config(str(augmented_config_file))
        aug = config.get("augmentation", {})
        assert aug["realclass_min_snr_db"] == 5.0
        assert aug["musan_max_snr_db"] == 15.0


# ---------------------------------------------------------------------------
# Test: Augment_fn Wiring — Only one noise dir raises error
# ---------------------------------------------------------------------------

class TestAugmentFnWiring:
    def test_create_augment_fn_both_dirs(self, noise_dirs):
        from src.train_whisper_small import create_augment_fn

        noise_dir, realclass_dir = noise_dirs
        config = {
            "augmentation": {
                "realclass_min_snr_db": 5.0,
                "realclass_max_snr_db": 20.0,
                "musan_min_snr_db": 0.0,
                "musan_max_snr_db": 15.0,
            },
        }
        fn = create_augment_fn(
            noise_dir=noise_dir,
            realclass_dir=realclass_dir,
            config=config,
        )
        assert fn is not None
        assert callable(fn)

    def test_create_augment_fn_no_dirs(self):
        from src.train_whisper_small import create_augment_fn

        config = {"augmentation": {}}
        fn = create_augment_fn(noise_dir=None, realclass_dir=None, config=config)
        assert fn is None

    def test_create_augment_fn_one_dir_raises(self, noise_dirs):
        from src.train_whisper_small import create_augment_fn

        noise_dir, _ = noise_dirs
        config = {"augmentation": {}}
        with pytest.raises(ValueError, match="Both --noise-dir and --realclass-dir"):
            create_augment_fn(noise_dir=noise_dir, realclass_dir=None, config=config)

    def test_create_augment_fn_other_dir_raises(self, noise_dirs):
        from src.train_whisper_small import create_augment_fn

        _, realclass_dir = noise_dirs
        config = {"augmentation": {}}
        with pytest.raises(ValueError, match="Both --noise-dir and --realclass-dir"):
            create_augment_fn(noise_dir=None, realclass_dir=realclass_dir, config=config)

    def test_lora_create_augment_fn_both_dirs(self, noise_dirs):
        from src.train_whisper_lora import create_augment_fn

        noise_dir, realclass_dir = noise_dirs
        config = {
            "augmentation": {
                "realclass_min_snr_db": 5.0,
                "realclass_max_snr_db": 20.0,
                "musan_min_snr_db": 0.0,
                "musan_max_snr_db": 15.0,
            },
        }
        fn = create_augment_fn(
            noise_dir=noise_dir,
            realclass_dir=realclass_dir,
            config=config,
        )
        assert fn is not None
        assert callable(fn)

    def test_lora_create_augment_fn_no_dirs(self):
        from src.train_whisper_lora import create_augment_fn

        config = {"augmentation": {}}
        fn = create_augment_fn(noise_dir=None, realclass_dir=None, config=config)
        assert fn is None

    def test_lora_create_augment_fn_one_dir_raises(self, noise_dirs):
        from src.train_whisper_lora import create_augment_fn

        noise_dir, _ = noise_dirs
        config = {"augmentation": {}}
        with pytest.raises(ValueError, match="Both --noise-dir and --realclass-dir"):
            create_augment_fn(noise_dir=noise_dir, realclass_dir=None, config=config)


# ---------------------------------------------------------------------------
# Test: Hub model ID override wiring
# ---------------------------------------------------------------------------

class TestHubModelIdOverride:
    def test_small_hub_id_override_in_config(self):
        from src.train_whisper_small import setup_training_args

        config = {
            "learning_rate": 1e-5,
            "warmup_steps": 500,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "fp16": True,
            "gradient_checkpointing": True,
            "eval_steps": 500,
            "save_steps": 500,
            "save_total_limit": 3,
            "generation_max_length": 225,
            "hub_model_id": "user/augmented-small",
            "hub_private_repo": True,
        }
        args = setup_training_args(
            config, output_dir="/tmp/test", push_to_hub=True
        )
        assert args.hub_model_id == "user/augmented-small"


# ---------------------------------------------------------------------------
# Test: Augmented training config in YAML
# ---------------------------------------------------------------------------

class TestTrainingConfigYaml:
    def test_augmentation_section_exists(self, project_root):
        config_path = project_root / "configs" / "training_config.yaml"
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        aug = raw["common"]["augmentation"]
        assert "realclass_min_snr_db" in aug
        assert "realclass_max_snr_db" in aug
        assert "musan_min_snr_db" in aug
        assert "musan_max_snr_db" in aug
        assert aug["realclass_min_snr_db"] == 5.0
        assert aug["realclass_max_snr_db"] == 20.0
        assert aug["musan_min_snr_db"] == 0.0
        assert aug["musan_max_snr_db"] == 15.0


# ---------------------------------------------------------------------------
# Test: Notebook existence and structure
# ---------------------------------------------------------------------------

class TestAugmentedNotebook:
    def test_notebook_exists(self, project_root):
        nb_path = project_root / "notebooks" / "04_augmented.ipynb"
        assert nb_path.exists(), "notebooks/04_augmented.ipynb must exist"

    def test_notebook_has_cells(self, project_root):
        nb_path = project_root / "notebooks" / "04_augmented.ipynb"
        nb = json.loads(nb_path.read_text())
        assert "cells" in nb
        assert len(nb["cells"]) >= 5, "Notebook should have at least 5 cells"

    def test_notebook_references_augmentation(self, project_root):
        nb_path = project_root / "notebooks" / "04_augmented.ipynb"
        content = nb_path.read_text()
        assert "noise-dir" in content or "noise_dir" in content
        assert "realclass-dir" in content or "realclass_dir" in content

    def test_notebook_trains_both_models(self, project_root):
        nb_path = project_root / "notebooks" / "04_augmented.ipynb"
        content = nb_path.read_text()
        assert "train_whisper_lora" in content or "train_lora" in content
        assert "train_whisper_small" in content or "train_small" in content

    def test_notebook_has_augmented_hub_ids(self, project_root):
        nb_path = project_root / "notebooks" / "04_augmented.ipynb"
        content = nb_path.read_text()
        assert "augmented" in content.lower()


# ---------------------------------------------------------------------------
# Test: Backward compatibility — no noise args produces same behavior
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    @patch("src.train_whisper_small.WhisperDataset")
    @patch("src.train_whisper_small.create_train_val_split")
    @patch("src.train_whisper_small.load_metadata")
    def test_no_augment_fn_when_no_noise_dirs(
        self, mock_load, mock_split, mock_dataset
    ):
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
            "min_duration_sec": 0.3,
            "max_duration_sec": 30.0,
            "validation": {"split_ratio": 0.1, "split_by": "child_id",
                           "stratify_by": "age_bucket"},
        }

        # No augment_fn passed — should work as before
        train_ds, val_ds = build_datasets(config, "/data/meta.jsonl", "/data/audio")

        # Train dataset should have augment_fn=None
        train_kwargs = mock_dataset.call_args_list[0][1]
        assert train_kwargs["augment_fn"] is None

    @patch("src.train_whisper_small.WhisperDataset")
    @patch("src.train_whisper_small.create_train_val_split")
    @patch("src.train_whisper_small.load_metadata")
    def test_augment_fn_passed_when_provided(
        self, mock_load, mock_split, mock_dataset
    ):
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

        fake_augment = MagicMock()
        config = {
            "model_name": "openai/whisper-small",
            "min_duration_sec": 0.3,
            "max_duration_sec": 30.0,
            "validation": {"split_ratio": 0.1, "split_by": "child_id",
                           "stratify_by": "age_bucket"},
        }

        build_datasets(config, "/data/meta.jsonl", "/data/audio", augment_fn=fake_augment)

        # Train dataset should receive augment_fn
        train_kwargs = mock_dataset.call_args_list[0][1]
        assert train_kwargs["augment_fn"] is fake_augment

        # Val dataset should NOT have augment_fn
        val_kwargs = mock_dataset.call_args_list[1][1]
        assert val_kwargs["augment_fn"] is None
