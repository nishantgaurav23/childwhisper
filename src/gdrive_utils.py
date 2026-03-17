"""Google Drive data access utilities for ChildWhisper.

Provides functions to download competition data from Google Drive
into Kaggle notebooks or local development environments.

Supports two modes:
1. gdown (shared links) — simplest, requires link sharing enabled
2. PyDrive2 + service account — for private files, no link sharing needed

Usage in Kaggle notebook:
    from src.gdrive_utils import sync_gdrive_to_kaggle
    sync_gdrive_to_kaggle(folder_id="YOUR_FOLDER_ID")
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _ensure_gdown():
    """Install gdown if not available."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        logger.info("Installing gdown...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "gdown"],
        )


def download_file(
    file_id: str,
    output_path: str | Path,
    quiet: bool = False,
) -> Path:
    """Download a single file from Google Drive by file ID.

    The file must have link sharing enabled ("Anyone with the link").

    Args:
        file_id: Google Drive file ID (from the share URL).
        output_path: Local path to save the file.
        quiet: Suppress download progress.

    Returns:
        Path to the downloaded file.
    """
    _ensure_gdown()
    import gdown

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(output_path), quiet=quiet)

    if not output_path.exists():
        raise FileNotFoundError(f"Download failed for file_id={file_id}")

    logger.info("Downloaded %s (%s)", output_path.name, _human_size(output_path))
    return output_path


def download_folder(
    folder_id: str,
    output_dir: str | Path,
    quiet: bool = False,
) -> Path:
    """Download an entire Google Drive folder by folder ID.

    The folder must have link sharing enabled ("Anyone with the link").

    Args:
        folder_id: Google Drive folder ID (from the share URL).
        output_dir: Local directory to download into.
        quiet: Suppress download progress.

    Returns:
        Path to the output directory.
    """
    _ensure_gdown()
    import gdown

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=str(output_dir), quiet=quiet)

    logger.info("Downloaded folder to %s", output_dir)
    return output_dir


def download_from_config(
    config: dict,
    dest_dir: str | Path,
    quiet: bool = False,
) -> Path:
    """Download data files specified in a config dict.

    Config format:
        {
            "folder_id": "GOOGLE_DRIVE_FOLDER_ID",  # for entire folder
            # OR individual files:
            "files": {
                "train_word_transcripts.jsonl": "FILE_ID_1",
                "audio.zip": "FILE_ID_2",
            }
        }

    Args:
        config: Dict with folder_id or files mapping.
        dest_dir: Destination directory.
        quiet: Suppress progress output.

    Returns:
        Path to dest_dir.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if "folder_id" in config:
        download_folder(config["folder_id"], dest_dir, quiet=quiet)
    elif "files" in config:
        for filename, file_id in config["files"].items():
            download_file(file_id, dest_dir / filename, quiet=quiet)
    else:
        raise ValueError("Config must have 'folder_id' or 'files' key")

    return dest_dir


def sync_gdrive_to_kaggle(
    folder_id: str | None = None,
    file_ids: dict[str, str] | None = None,
    dest_dir: str | Path = "/kaggle/working/data",
    quiet: bool = False,
) -> dict[str, Path]:
    """Download Google Drive data and return paths compatible with kaggle_utils.

    This is the main entry point for Kaggle notebooks. Downloads data from
    Google Drive and returns a paths dict matching the format from
    kaggle_utils.get_paths().

    Args:
        folder_id: Google Drive folder ID containing audio/ and metadata.
        file_ids: Dict mapping filenames to Drive file IDs (alternative to folder).
        dest_dir: Local destination directory.
        quiet: Suppress download progress.

    Returns:
        Dict with audio_dir, metadata_path, output_dir — same format as
        kaggle_utils.get_paths().

    Example:
        paths = sync_gdrive_to_kaggle(folder_id="1AbC...xYz")
        # paths["audio_dir"] -> /kaggle/working/data/audio
        # paths["metadata_path"] -> /kaggle/working/data/train_word_transcripts.jsonl
    """
    dest_dir = Path(dest_dir)

    # Skip download if data already exists
    metadata_path = dest_dir / "train_word_transcripts.jsonl"
    audio_dir = dest_dir / "audio"

    if metadata_path.exists() and audio_dir.exists():
        n_audio = len(list(audio_dir.rglob("*.flac"))) + len(list(audio_dir.rglob("*.wav")))
        if n_audio > 0:
            logger.info(
                "Data already exists at %s (%d audio files), skipping download",
                dest_dir, n_audio,
            )
            return _make_paths(dest_dir)

    # Download from Google Drive
    if folder_id is not None:
        logger.info("Downloading data folder from Google Drive...")
        download_folder(folder_id, dest_dir, quiet=quiet)
    elif file_ids is not None:
        logger.info("Downloading data files from Google Drive...")
        for filename, fid in file_ids.items():
            download_file(fid, dest_dir / filename, quiet=quiet)
    else:
        # Check environment variable as fallback
        env_folder_id = os.environ.get("GDRIVE_FOLDER_ID")
        if env_folder_id:
            logger.info("Using GDRIVE_FOLDER_ID from environment...")
            download_folder(env_folder_id, dest_dir, quiet=quiet)
        else:
            raise ValueError(
                "Provide folder_id, file_ids, or set GDRIVE_FOLDER_ID env var"
            )

    # Handle zipped audio
    _extract_archives(dest_dir)

    # Validate
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {metadata_path} after download. "
            "Ensure your Google Drive folder contains train_word_transcripts.jsonl"
        )

    paths = _make_paths(dest_dir)
    logger.info("Data ready: %s", {k: str(v) for k, v in paths.items()})
    return paths


def _make_paths(data_dir: Path) -> dict[str, Path]:
    """Build paths dict from a data directory."""
    return {
        "audio_dir": data_dir / "audio",
        "metadata_path": data_dir / "train_word_transcripts.jsonl",
        "output_dir": Path("/kaggle/working/checkpoints/whisper-small"),
    }


def _extract_archives(dest_dir: Path) -> None:
    """Extract any .zip archives found in dest_dir."""
    import zipfile

    for archive in dest_dir.glob("*.zip"):
        logger.info("Extracting %s...", archive.name)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest_dir)
        archive.unlink()  # Remove zip after extraction


def _human_size(path: Path) -> str:
    """Return human-readable file size."""
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
