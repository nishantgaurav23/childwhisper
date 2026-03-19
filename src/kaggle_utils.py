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

# Default Kaggle dataset base path — try both standard and datasets/username/ layouts
_KAGGLE_BASE_CANDIDATES = [
    Path("/kaggle/input/pasketti-audio"),
    Path("/kaggle/input/datasets/nishantgaurav23/pasketti-audio"),
]
KAGGLE_DATASET_BASE = next(
    (p for p in _KAGGLE_BASE_CANDIDATES if p.exists()),
    _KAGGLE_BASE_CANDIDATES[0],  # default if none exist
)


def is_kaggle() -> bool:
    """Detect if running inside a Kaggle kernel."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return Path("/kaggle/working").exists()


_unify_cache: dict[str, Path] = {}


def unify_kaggle_audio(
    dataset_base: Path | None = None,
    unified_dir: Path | None = None,
) -> Path:
    """Symlink split audio_part_*/audio/ files into a single unified directory.

    Kaggle dataset has audio split across audio_part_0/, audio_part_1/, etc.
    The JSONL metadata references paths like "audio/U_*.flac". This function
    creates /kaggle/working/unified_audio/audio/ with symlinks to all files
    so the existing code works unchanged.

    Results are cached so repeated calls return instantly.

    Args:
        dataset_base: Base path of the Kaggle dataset.
        unified_dir: Where to create the unified directory.

    Returns:
        The unified base directory (use as audio_dir parent).
    """
    if dataset_base is None:
        dataset_base = KAGGLE_DATASET_BASE
    if unified_dir is None:
        unified_dir = Path("/kaggle/working/unified_audio")

    cache_key = f"{dataset_base}:{unified_dir}"
    if cache_key in _unify_cache:
        return _unify_cache[cache_key]

    audio_out = unified_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    # Symlink audio files from all audio_part_* directories
    part_dirs = sorted(dataset_base.glob("audio_part_*/audio"))
    count = 0
    for part_dir in part_dirs:
        for f in part_dir.iterdir():
            link = audio_out / f.name
            if not link.exists():
                link.symlink_to(f)
                count += 1

    # Also check for audio/audio/ (non-split layout)
    flat_dir = dataset_base / "audio" / "audio"
    if flat_dir.is_dir():
        for f in flat_dir.iterdir():
            link = audio_out / f.name
            if not link.exists():
                link.symlink_to(f)
                count += 1

    logger.info("Unified %d audio files into %s", count, audio_out)
    _unify_cache[cache_key] = unified_dir
    return unified_dir


_noise_cache: dict[str, dict[str, Path | None]] = {}


def get_kaggle_noise_paths(
    dataset_base: Path | None = None,
) -> dict[str, Path | None]:
    """Return paths to noise and MUSAN directories on Kaggle.

    Results are cached so repeated calls return instantly.

    Args:
        dataset_base: Base path of the Kaggle dataset.

    Returns:
        Dict with noise_dir (unified from noise_part_*) and musan_dir.
    """
    if dataset_base is None:
        dataset_base = KAGGLE_DATASET_BASE

    cache_key = str(dataset_base)
    if cache_key in _noise_cache:
        return _noise_cache[cache_key]

    # Unify noise_part_* into a single directory
    noise_parts = sorted(dataset_base.glob("noise_part_*/audio"))
    noise_dir = None
    if noise_parts:
        noise_out = Path("/kaggle/working/unified_noise")
        noise_out.mkdir(parents=True, exist_ok=True)
        for part_dir in noise_parts:
            for f in part_dir.iterdir():
                link = noise_out / f.name
                if not link.exists():
                    link.symlink_to(f)
        noise_dir = noise_out
        logger.info("Unified noise files into %s", noise_out)

    # MUSAN is not split
    musan_dir = dataset_base / "musan" / "musan"
    if not musan_dir.is_dir():
        musan_dir = None

    result = {"noise_dir": noise_dir, "musan_dir": musan_dir}
    _noise_cache[cache_key] = result
    return result


def get_kaggle_paths(dataset_slug: str = "pasketti-audio") -> dict[str, Path]:
    """Return standard paths for a Kaggle dataset.

    Automatically unifies split audio directories into a single location.

    Args:
        dataset_slug: Kaggle dataset slug (unused when KAGGLE_DATASET_BASE exists,
            kept for API compatibility).

    Returns:
        Dict with audio_dir, metadata_path, output_dir, noise_dir, musan_dir.
    """
    base = KAGGLE_DATASET_BASE
    if not base.exists():
        # Fallback to slug-based path
        base = Path(f"/kaggle/input/{dataset_slug}")

    # Unify split audio into a single directory
    unified = unify_kaggle_audio(base)

    # Find metadata (could be at top level, inside a part, or nested deeper)
    metadata_path = base / "train_word_transcripts.jsonl"
    if not metadata_path.exists():
        for candidate in base.glob("**/train_word_transcripts.jsonl"):
            metadata_path = candidate
            break

    # Get noise paths
    noise_paths = get_kaggle_noise_paths(base)

    return {
        "audio_dir": unified,
        "metadata_path": metadata_path,
        "output_dir": Path("/kaggle/working/checkpoints/whisper-small"),
        "noise_dir": noise_paths["noise_dir"],
        "musan_dir": noise_paths["musan_dir"],
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

    Uses HF_TOKEN environment variable directly and skips the slow
    online token validation that huggingface_hub.login() does by default.

    Raises:
        ValueError: If HF_TOKEN is not set.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Set it with: export HF_TOKEN=hf_your_token"
        )
    # Set HUGGING_FACE_HUB_TOKEN so all HF libraries pick it up without
    # a network round-trip for validation. add_to_git_credential=False
    # avoids slow git-credential writes on Kaggle.
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    login(token=token, add_to_git_credential=False, new_session=False)
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

    if resume_from is not None:
        args.extend(["--resume-from", str(resume_from)])

    if dry_run:
        args.append("--dry-run")

    if not os.environ.get("HF_TOKEN"):
        args.append("--no-push-to-hub")

    return args


def get_kaggle_paths_lora(dataset_slug: str = "pasketti-audio") -> dict[str, Path]:
    """Return standard paths for a Kaggle dataset with LoRA output dir.

    Automatically unifies split audio directories into a single location.

    Args:
        dataset_slug: Kaggle dataset slug (unused when KAGGLE_DATASET_BASE exists,
            kept for API compatibility).

    Returns:
        Dict with audio_dir, metadata_path, output_dir, noise_dir, musan_dir.
    """
    # Reuse get_kaggle_paths and just override output_dir
    paths = get_kaggle_paths(dataset_slug)
    paths["output_dir"] = Path("/kaggle/working/checkpoints/whisper-large-v3-lora")
    return paths


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

    if resume_from is not None:
        args.extend(["--resume-from", str(resume_from)])

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
    total_gb = props.total_memory / (1024**3)
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
