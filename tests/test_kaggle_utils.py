"""Tests for Kaggle notebook utilities (S2.3)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.kaggle_utils import (
    download_checkpoint,
    get_kaggle_paths,
    get_kaggle_training_args,
    get_latest_checkpoint,
    get_local_paths,
    get_paths,
    is_kaggle,
    setup_hub_auth,
    verify_kaggle_data,
)


class TestIsKaggle:
    """Test Kaggle environment detection."""

    def test_is_kaggle_when_kaggle_env_var_set(self):
        with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}):
            assert is_kaggle() is True

    def test_is_kaggle_when_kaggle_path_exists(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove KAGGLE_KERNEL_RUN_TYPE if present
            env = {k: v for k, v in os.environ.items() if k != "KAGGLE_KERNEL_RUN_TYPE"}
            with patch.dict(os.environ, env, clear=True):
                with patch("pathlib.Path.exists", return_value=True):
                    assert is_kaggle() is True

    def test_not_kaggle_locally(self):
        env = {k: v for k, v in os.environ.items() if k != "KAGGLE_KERNEL_RUN_TYPE"}
        with patch.dict(os.environ, env, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                assert is_kaggle() is False


class TestGetKagglePaths:
    """Test Kaggle path generation."""

    def test_returns_correct_paths(self):
        paths = get_kaggle_paths("pasketti-word-audio")
        assert paths["audio_dir"] == Path("/kaggle/input/pasketti-word-audio/audio")
        assert paths["metadata_path"] == Path(
            "/kaggle/input/pasketti-word-audio/train_word_transcripts.jsonl"
        )
        assert paths["output_dir"] == Path("/kaggle/working/checkpoints/whisper-small")

    def test_custom_dataset_slug(self):
        paths = get_kaggle_paths("my-custom-dataset")
        assert "my-custom-dataset" in str(paths["audio_dir"])


class TestGetLocalPaths:
    """Test local path generation."""

    def test_returns_correct_paths(self):
        paths = get_local_paths("/home/user/data")
        assert paths["audio_dir"] == Path("/home/user/data/audio_sample")
        assert paths["metadata_path"] == Path(
            "/home/user/data/train_word_transcripts.jsonl"
        )
        assert paths["output_dir"] == Path("./checkpoints/whisper-small")

    def test_accepts_path_object(self):
        paths = get_local_paths(Path("/tmp/data"))
        assert paths["audio_dir"] == Path("/tmp/data/audio_sample")


class TestGetPaths:
    """Test auto-detection path routing."""

    def test_returns_kaggle_paths_on_kaggle(self):
        with patch("src.kaggle_utils.is_kaggle", return_value=True):
            paths = get_paths(dataset_slug="pasketti-word-audio")
            assert "/kaggle/" in str(paths["audio_dir"])

    def test_returns_local_paths_locally(self):
        with patch("src.kaggle_utils.is_kaggle", return_value=False):
            paths = get_paths(local_data_dir="/home/user/data")
            assert "audio_sample" in str(paths["audio_dir"])

    def test_raises_without_local_dir_when_not_kaggle(self):
        with patch("src.kaggle_utils.is_kaggle", return_value=False):
            with pytest.raises(ValueError, match="local_data_dir"):
                get_paths()


class TestSetupHubAuth:
    """Test HF Hub authentication setup."""

    def test_auth_with_env_token(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token_123"}):
            with patch("src.kaggle_utils.login") as mock_login:
                setup_hub_auth()
                mock_login.assert_called_once_with(token="hf_test_token_123")

    def test_auth_without_token_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="HF_TOKEN"):
                setup_hub_auth()


class TestGetLatestCheckpoint:
    """Test checkpoint discovery on HF Hub."""

    def test_finds_existing_checkpoint(self):
        mock_api = MagicMock()
        mock_api.list_repo_refs.return_value = MagicMock(
            branches=[MagicMock(name="main")]
        )
        with patch("src.kaggle_utils.HfApi", return_value=mock_api):
            result = get_latest_checkpoint("user/model-repo")
            assert result is True

    def test_no_checkpoint_when_repo_missing(self):
        mock_api = MagicMock()
        mock_api.list_repo_refs.side_effect = Exception("Repository not found")
        with patch("src.kaggle_utils.HfApi", return_value=mock_api):
            result = get_latest_checkpoint("user/nonexistent-repo")
            assert result is False


class TestDownloadCheckpoint:
    """Test checkpoint downloading from HF Hub."""

    def test_download_calls_snapshot(self):
        with patch("src.kaggle_utils.snapshot_download") as mock_dl:
            mock_dl.return_value = "/tmp/downloaded"
            result = download_checkpoint("user/model-repo", "/tmp/local")
            mock_dl.assert_called_once_with(
                repo_id="user/model-repo",
                local_dir="/tmp/local",
            )
            assert result == "/tmp/downloaded"

    def test_download_returns_none_on_error(self):
        with patch("src.kaggle_utils.snapshot_download") as mock_dl:
            mock_dl.side_effect = Exception("Download failed")
            result = download_checkpoint("user/model-repo", "/tmp/local")
            assert result is None


class TestGetKaggleTrainingArgs:
    """Test training CLI args construction."""

    def test_basic_args(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("common:\n  sample_rate: 16000\n")
        args = get_kaggle_training_args(
            config_path=str(config_path),
            metadata_path="/kaggle/input/data/transcripts.jsonl",
            audio_dir="/kaggle/input/data/audio",
            output_dir="/kaggle/working/checkpoints",
        )
        assert "--metadata-path" in args
        assert "/kaggle/input/data/transcripts.jsonl" in args
        assert "--audio-dir" in args
        assert "--config" in args

    def test_args_with_resume(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("common:\n  sample_rate: 16000\n")
        args = get_kaggle_training_args(
            config_path=str(config_path),
            metadata_path="/data/transcripts.jsonl",
            audio_dir="/data/audio",
            output_dir="/checkpoints",
            resume_from="/checkpoints/checkpoint-500",
        )
        assert "--output-dir" in args
        # resume_from should be set as output_dir for Trainer to find
        idx = args.index("--output-dir")
        assert args[idx + 1] == "/checkpoints"

    def test_no_push_without_hf_token(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("common:\n  sample_rate: 16000\n")
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            args = get_kaggle_training_args(
                config_path=str(config_path),
                metadata_path="/data/t.jsonl",
                audio_dir="/data/audio",
                output_dir="/out",
            )
            assert "--no-push-to-hub" in args

    def test_push_with_hf_token(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("common:\n  sample_rate: 16000\n")
        with patch.dict(os.environ, {"HF_TOKEN": "hf_abc"}):
            args = get_kaggle_training_args(
                config_path=str(config_path),
                metadata_path="/data/t.jsonl",
                audio_dir="/data/audio",
                output_dir="/out",
            )
            assert "--no-push-to-hub" not in args


class TestVerifyKaggleData:
    """Test data verification."""

    def test_valid_data(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "utt1.flac").write_text("fake")
        (audio_dir / "utt2.flac").write_text("fake")

        meta_path = tmp_path / "metadata.jsonl"
        entries = [
            {
                "utterance_id": "utt1",
                "audio_path": "utt1.flac",
                "audio_duration_sec": 2.5,
                "child_id": "c1",
                "age_bucket": "5-7",
                "orthographic_text": "hello",
            },
            {
                "utterance_id": "utt2",
                "audio_path": "utt2.flac",
                "audio_duration_sec": 1.0,
                "child_id": "c2",
                "age_bucket": "3-4",
                "orthographic_text": "bye",
            },
        ]
        meta_path.write_text("\n".join(json.dumps(e) for e in entries))

        stats = verify_kaggle_data(str(audio_dir), str(meta_path))
        assert stats["num_utterances"] == 2
        assert stats["num_audio_found"] == 2
        assert stats["num_missing_audio"] == 0
        assert "duration_stats" in stats

    def test_missing_audio_files(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        # Only create one audio file, metadata references two
        (audio_dir / "utt1.flac").write_text("fake")

        meta_path = tmp_path / "metadata.jsonl"
        entries = [
            {
                "utterance_id": "utt1",
                "audio_path": "utt1.flac",
                "audio_duration_sec": 2.5,
                "child_id": "c1",
                "age_bucket": "5-7",
                "orthographic_text": "hello",
            },
            {
                "utterance_id": "utt2",
                "audio_path": "utt2.flac",
                "audio_duration_sec": 1.0,
                "child_id": "c2",
                "age_bucket": "3-4",
                "orthographic_text": "bye",
            },
        ]
        meta_path.write_text("\n".join(json.dumps(e) for e in entries))

        stats = verify_kaggle_data(str(audio_dir), str(meta_path))
        assert stats["num_utterances"] == 2
        assert stats["num_missing_audio"] == 1
        assert stats["num_audio_found"] == 1

    def test_nonexistent_metadata_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            verify_kaggle_data(str(tmp_path / "audio"), str(tmp_path / "missing.jsonl"))
