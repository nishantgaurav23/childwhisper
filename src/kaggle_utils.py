"""Kaggle notebook utilities for ChildWhisper training.

Provides environment detection, path configuration, HF Hub checkpoint management,
training argument construction, and data verification for Kaggle notebooks.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, login, snapshot_download

logger = logging.getLogger(__name__)


def is_kaggle() -> bool:
    """Detect if running inside a Kaggle kernel."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return Path("/kaggle/working").exists()


def get_kaggle_paths(dataset_slug: str) -> dict[str, Path]:
    """Return standard paths for a Kaggle dataset.

    Args:
        dataset_slug: Kaggle dataset slug (e.g. "pasketti-word-audio").

    Returns:
        Dict with audio_dir, metadata_path, output_dir.
    """
    base = Path(f"/kaggle/input/{dataset_slug}")
    return {
        "audio_dir": base / "audio",
        "metadata_path": base / "train_word_transcripts.jsonl",
        "output_dir": Path("/kaggle/working/checkpoints/whisper-small"),
    }


def get_local_paths(data_dir: str | Path) -> dict[str, Path]:
    """Return standard paths for local development.

    Args:
        data_dir: Local data directory path.

    Returns:
        Dict with audio_dir, metadata_path, output_dir.
    """
    base = Path(data_dir)
    return {
        "audio_dir": base / "audio_sample",
        "metadata_path": base / "train_word_transcripts.jsonl",
        "output_dir": Path("./checkpoints/whisper-small"),
    }


def get_paths(
    dataset_slug: str = "pasketti-word-audio",
    local_data_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Auto-detect environment and return correct paths.

    Uses Kaggle paths if on Kaggle, otherwise local paths.

    Args:
        dataset_slug: Kaggle dataset slug.
        local_data_dir: Local data directory (required when not on Kaggle).

    Returns:
        Dict with audio_dir, metadata_path, output_dir.

    Raises:
        ValueError: If not on Kaggle and local_data_dir is not provided.
    """
    if is_kaggle():
        return get_kaggle_paths(dataset_slug)
    if local_data_dir is None:
        raise ValueError("local_data_dir is required when not running on Kaggle")
    return get_local_paths(local_data_dir)


def setup_hub_auth() -> None:
    """Authenticate with HuggingFace Hub using HF_TOKEN env var.

    Raises:
        ValueError: If HF_TOKEN is not set.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Set it with: export HF_TOKEN=hf_your_token"
        )
    login(token=token)
    logger.info("Authenticated with HuggingFace Hub")


def get_latest_checkpoint(hub_model_id: str) -> bool:
    """Check if a checkpoint exists on HF Hub.

    Args:
        hub_model_id: HF Hub model repository ID.

    Returns:
        True if a checkpoint exists, False otherwise.
    """
    try:
        api = HfApi()
        api.list_repo_refs(hub_model_id)
        return True
    except Exception:
        return False


def download_checkpoint(hub_model_id: str, local_dir: str | Path) -> str | None:
    """Download model checkpoint from HF Hub.

    Args:
        hub_model_id: HF Hub model repository ID.
        local_dir: Local directory to download to.

    Returns:
        Path to downloaded checkpoint, or None on error.
    """
    try:
        result = snapshot_download(
            repo_id=hub_model_id,
            local_dir=str(local_dir),
        )
        logger.info("Downloaded checkpoint to %s", result)
        return result
    except Exception as e:
        logger.warning("Failed to download checkpoint: %s", e)
        return None


def get_kaggle_training_args(
    config_path: str,
    metadata_path: str,
    audio_dir: str,
    output_dir: str,
    resume_from: str | None = None,
    num_epochs: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Build CLI argument list for train_whisper_small.main().

    Args:
        config_path: Path to training config YAML.
        metadata_path: Path to metadata JSONL.
        audio_dir: Path to audio directory.
        output_dir: Output directory for checkpoints.
        resume_from: Path to checkpoint to resume from.
        num_epochs: Override number of epochs.
        dry_run: If True, add --dry-run flag.

    Returns:
        List of CLI argument strings.
    """
    args = [
        "--metadata-path", str(metadata_path),
        "--audio-dir", str(audio_dir),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
    ]

    if num_epochs is not None:
        args.extend(["--num-train-epochs", str(num_epochs)])

    if dry_run:
        args.append("--dry-run")

    if not os.environ.get("HF_TOKEN"):
        args.append("--no-push-to-hub")

    return args


def get_kaggle_paths_lora(dataset_slug: str) -> dict[str, Path]:
    """Return standard paths for a Kaggle dataset with LoRA output dir.

    Args:
        dataset_slug: Kaggle dataset slug (e.g. "pasketti-word-audio").

    Returns:
        Dict with audio_dir, metadata_path, output_dir for LoRA training.
    """
    base = Path(f"/kaggle/input/{dataset_slug}")
    return {
        "audio_dir": base / "audio",
        "metadata_path": base / "train_word_transcripts.jsonl",
        "output_dir": Path("/kaggle/working/checkpoints/whisper-large-v3-lora"),
    }


def get_local_paths_lora(data_dir: str | Path) -> dict[str, Path]:
    """Return standard paths for local development with LoRA output dir.

    Args:
        data_dir: Local data directory path.

    Returns:
        Dict with audio_dir, metadata_path, output_dir for LoRA training.
    """
    base = Path(data_dir)
    return {
        "audio_dir": base / "audio_sample",
        "metadata_path": base / "train_word_transcripts.jsonl",
        "output_dir": Path("./checkpoints/whisper-large-v3-lora"),
    }


def get_paths_lora(
    dataset_slug: str = "pasketti-word-audio",
    local_data_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Auto-detect environment and return correct LoRA paths.

    Args:
        dataset_slug: Kaggle dataset slug.
        local_data_dir: Local data directory (required when not on Kaggle).

    Returns:
        Dict with audio_dir, metadata_path, output_dir for LoRA training.

    Raises:
        ValueError: If not on Kaggle and local_data_dir is not provided.
    """
    if is_kaggle():
        return get_kaggle_paths_lora(dataset_slug)
    if local_data_dir is None:
        raise ValueError("local_data_dir is required when not running on Kaggle")
    return get_local_paths_lora(local_data_dir)


def get_lora_training_args(
    config_path: str,
    metadata_path: str,
    audio_dir: str,
    output_dir: str,
    resume_from: str | None = None,
    num_epochs: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Build CLI argument list for train_whisper_lora.main().

    Args:
        config_path: Path to training config YAML.
        metadata_path: Path to metadata JSONL.
        audio_dir: Path to audio directory.
        output_dir: Output directory for checkpoints.
        resume_from: Path to checkpoint to resume from.
        num_epochs: Override number of epochs.
        dry_run: If True, add --dry-run flag.

    Returns:
        List of CLI argument strings.
    """
    args = [
        "--metadata-path", str(metadata_path),
        "--audio-dir", str(audio_dir),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
    ]

    if num_epochs is not None:
        args.extend(["--num-train-epochs", str(num_epochs)])

    if dry_run:
        args.append("--dry-run")

    if not os.environ.get("HF_TOKEN"):
        args.append("--no-push-to-hub")

    return args


def check_gpu_memory(min_gb: float = 14.0) -> dict:
    """Check GPU memory availability for LoRA training.

    Args:
        min_gb: Minimum GPU memory in GB considered sufficient.

    Returns:
        Dict with gpu_name, total_memory_gb, is_sufficient.
    """
    if not torch.cuda.is_available():
        return {
            "gpu_name": None,
            "total_memory_gb": 0.0,
            "is_sufficient": False,
        }

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_mem / (1024**3)
    return {
        "gpu_name": props.name,
        "total_memory_gb": total_gb,
        "is_sufficient": total_gb >= min_gb,
    }


def verify_kaggle_data(
    audio_dir: str | Path,
    metadata_path: str | Path,
) -> dict:
    """Verify competition data is accessible and return stats.

    Args:
        audio_dir: Directory containing audio files.
        metadata_path: Path to metadata JSONL file.

    Returns:
        Dict with num_utterances, num_audio_found, num_missing_audio, duration_stats.

    Raises:
        FileNotFoundError: If metadata file doesn't exist.
    """
    meta_path = Path(metadata_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    audio_path = Path(audio_dir)
    text = meta_path.read_text().strip()
    entries = [json.loads(line) for line in text.split("\n")] if text else []

    num_found = 0
    num_missing = 0
    durations = []

    for entry in entries:
        ap = audio_path / entry["audio_path"]
        if ap.exists():
            num_found += 1
        else:
            num_missing += 1
        dur = entry.get("audio_duration_sec")
        if dur is not None:
            durations.append(dur)

    duration_stats = {}
    if durations:
        duration_stats = {
            "min": min(durations),
            "max": max(durations),
            "mean": sum(durations) / len(durations),
            "total_hours": sum(durations) / 3600,
        }

    return {
        "num_utterances": len(entries),
        "num_audio_found": num_found,
        "num_missing_audio": num_missing,
        "duration_stats": duration_stats,
    }
