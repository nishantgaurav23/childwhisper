"""Tests for Kaggle LoRA notebook utilities (S3.2)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.kaggle_utils import (
    check_gpu_memory,
    get_kaggle_paths_lora,
    get_local_paths_lora,
    get_lora_training_args,
    get_paths_lora,
)


class TestGetKagglePathsLora:
    """Test Kaggle LoRA path generation."""

    def test_returns_correct_lora_paths(self):
        paths = get_kaggle_paths_lora("pasketti-word-audio")
        assert paths["audio_dir"] == Path("/kaggle/input/pasketti-word-audio/audio")
        assert paths["metadata_path"] == Path(
            "/kaggle/input/pasketti-word-audio/train_word_transcripts.jsonl"
        )
        assert paths["output_dir"] == Path(
            "/kaggle/working/checkpoints/whisper-large-v3-lora"
        )

    def test_custom_dataset_slug(self):
        paths = get_kaggle_paths_lora("my-custom-dataset")
        assert "my-custom-dataset" in str(paths["audio_dir"])
        assert "whisper-large-v3-lora" in str(paths["output_dir"])


class TestGetLocalPathsLora:
    """Test local LoRA path generation."""

    def test_returns_correct_lora_paths(self):
        paths = get_local_paths_lora("/home/user/data")
        assert paths["audio_dir"] == Path("/home/user/data/audio_sample")
        assert paths["metadata_path"] == Path(
            "/home/user/data/train_word_transcripts.jsonl"
        )
        assert paths["output_dir"] == Path("./checkpoints/whisper-large-v3-lora")

    def test_accepts_path_object(self):
        paths = get_local_paths_lora(Path("/tmp/data"))
        assert paths["audio_dir"] == Path("/tmp/data/audio_sample")
        assert "whisper-large-v3-lora" in str(paths["output_dir"])


class TestGetPathsLora:
    """Test auto-detection routing for LoRA paths."""

    def test_returns_kaggle_paths_on_kaggle(self):
        with patch("src.kaggle_utils.is_kaggle", return_value=True):
            paths = get_paths_lora(dataset_slug="pasketti-word-audio")
            assert "/kaggle/" in str(paths["audio_dir"])
            assert "whisper-large-v3-lora" in str(paths["output_dir"])

    def test_returns_local_paths_locally(self):
        with patch("src.kaggle_utils.is_kaggle", return_value=False):
            paths = get_paths_lora(local_data_dir="/home/user/data")
            assert "audio_sample" in str(paths["audio_dir"])
            assert "whisper-large-v3-lora" in str(paths["output_dir"])

    def test_raises_without_local_dir_when_not_kaggle(self):
        with patch("src.kaggle_utils.is_kaggle", return_value=False):
            with pytest.raises(ValueError, match="local_data_dir"):
                get_paths_lora()


class TestGetLoraTrainingArgs:
    """Test LoRA training CLI args construction."""

    def test_basic_args(self):
        args = get_lora_training_args(
            config_path="/configs/training_config.yaml",
            metadata_path="/kaggle/input/data/transcripts.jsonl",
            audio_dir="/kaggle/input/data/audio",
            output_dir="/kaggle/working/checkpoints",
        )
        assert "--metadata-path" in args
        assert "/kaggle/input/data/transcripts.jsonl" in args
        assert "--audio-dir" in args
        assert "/kaggle/input/data/audio" in args
        assert "--config" in args
        assert "--output-dir" in args

    def test_dry_run_flag(self):
        args = get_lora_training_args(
            config_path="/configs/config.yaml",
            metadata_path="/data/t.jsonl",
            audio_dir="/data/audio",
            output_dir="/out",
            dry_run=True,
        )
        assert "--dry-run" in args

    def test_no_push_without_hf_token(self):
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            args = get_lora_training_args(
                config_path="/configs/config.yaml",
                metadata_path="/data/t.jsonl",
                audio_dir="/data/audio",
                output_dir="/out",
            )
            assert "--no-push-to-hub" in args

    def test_push_with_hf_token(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_abc"}):
            args = get_lora_training_args(
                config_path="/configs/config.yaml",
                metadata_path="/data/t.jsonl",
                audio_dir="/data/audio",
                output_dir="/out",
            )
            assert "--no-push-to-hub" not in args

    def test_num_epochs_override(self):
        args = get_lora_training_args(
            config_path="/configs/config.yaml",
            metadata_path="/data/t.jsonl",
            audio_dir="/data/audio",
            output_dir="/out",
            num_epochs=5,
        )
        assert "--num-train-epochs" in args
        idx = args.index("--num-train-epochs")
        assert args[idx + 1] == "5"


class TestCheckGpuMemory:
    """Test GPU memory verification."""

    def test_with_cuda_available(self):
        mock_props = MagicMock()
        mock_props.name = "Tesla T4"
        mock_props.total_memory = 16 * 1024**3  # 16 GB in bytes

        with patch("src.kaggle_utils.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value = mock_props
            result = check_gpu_memory()
            assert result["gpu_name"] == "Tesla T4"
            assert result["total_memory_gb"] == pytest.approx(16.0, abs=0.5)
            assert result["is_sufficient"] is True

    def test_without_cuda(self):
        with patch("src.kaggle_utils.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            result = check_gpu_memory()
            assert result["gpu_name"] is None
            assert result["total_memory_gb"] == 0.0
            assert result["is_sufficient"] is False

    def test_insufficient_memory(self):
        mock_props = MagicMock()
        mock_props.name = "GTX 1060"
        mock_props.total_memory = 6 * 1024**3  # 6 GB

        with patch("src.kaggle_utils.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value = mock_props
            result = check_gpu_memory()
            assert result["is_sufficient"] is False


class TestNotebookStructure:
    """Test that the LoRA notebook has the correct structure."""

    def test_notebook_exists(self):
        nb_path = Path("notebooks/03_train_lora.ipynb")
        assert nb_path.exists(), "notebooks/03_train_lora.ipynb must exist"

    def test_notebook_has_nine_sections(self):
        nb_path = Path("notebooks/03_train_lora.ipynb")
        with open(nb_path) as f:
            nb = json.load(f)

        # Count markdown cells that start section headers (## N.)
        section_cells = []
        for cell in nb["cells"]:
            if cell["cell_type"] == "markdown":
                source = "".join(cell["source"])
                if source.strip().startswith("## "):
                    section_cells.append(source.strip())

        assert len(section_cells) >= 9, (
            f"Expected at least 9 section headers, found {len(section_cells)}: "
            f"{section_cells}"
        )

    def test_notebook_imports_lora_utils(self):
        nb_path = Path("notebooks/03_train_lora.ipynb")
        with open(nb_path) as f:
            nb = json.load(f)

        all_source = ""
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                all_source += "".join(cell["source"])

        assert "get_paths_lora" in all_source
        assert "get_lora_training_args" in all_source
        assert "train_whisper_lora" in all_source

    def test_notebook_imports_check_gpu(self):
        nb_path = Path("notebooks/03_train_lora.ipynb")
        with open(nb_path) as f:
            nb = json.load(f)

        all_source = ""
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                all_source += "".join(cell["source"])

        assert "check_gpu_memory" in all_source
