"""Submission packaging utilities for DrivenData competition.

Validates submission directory structure, generates file manifests,
computes size budgets, and builds the submission.zip for upload.

Spec: S3.4
"""

from __future__ import annotations

import zipfile
from pathlib import Path

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".DS_Store",
    ".git",
]

REQUIRED_FILES = [
    "main.py",
    "src/preprocess.py",
    "src/utils.py",
]

SIZE_WARNING_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
SIZE_ERROR_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB


def get_excludes() -> list[str]:
    """Return list of patterns to exclude from submission zip."""
    return list(EXCLUDE_PATTERNS)


def _should_exclude(path: Path) -> bool:
    """Check if a file path matches any exclusion pattern."""
    parts = path.parts
    name = path.name
    for pattern in EXCLUDE_PATTERNS:
        if pattern in parts or name == pattern or name.endswith(pattern):
            return True
    return False


def validate_submission_dir(submission_dir: str | Path) -> list[str]:
    """Validate submission directory structure. Returns list of errors (empty = valid)."""
    submission_dir = Path(submission_dir)
    errors: list[str] = []

    if not submission_dir.exists():
        errors.append(f"Submission directory does not exist: {submission_dir}")
        return errors

    if not submission_dir.is_dir():
        errors.append(f"Not a directory: {submission_dir}")
        return errors

    for required in REQUIRED_FILES:
        if not (submission_dir / required).exists():
            errors.append(f"Required file missing: {required}")

    if not (submission_dir / "src").is_dir():
        if not any("src" in e.lower() for e in errors):
            errors.append("Required directory missing: src/")

    if not (submission_dir / "model_weights").is_dir():
        errors.append("model_weights/ directory not found (weights may not be downloaded)")

    return errors


def get_submission_manifest(submission_dir: str | Path) -> list[dict]:
    """List all files to include in submission with sizes.

    Returns list of dicts with 'path' (relative to submission_dir) and 'size' (bytes).
    Excludes __pycache__, .pyc, .DS_Store, .git.
    """
    submission_dir = Path(submission_dir)
    manifest = []

    for file_path in sorted(submission_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(submission_dir)
        if _should_exclude(rel):
            continue
        manifest.append({
            "path": str(rel),
            "size": file_path.stat().st_size,
        })

    return manifest


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def compute_size_budget(submission_dir: str | Path) -> dict:
    """Compute code vs weights size breakdown.

    Returns dict with keys: code_bytes, weights_bytes, total_bytes,
    total_human, warning.
    """
    submission_dir = Path(submission_dir)
    manifest = get_submission_manifest(submission_dir)

    weights_bytes = 0
    code_bytes = 0

    for entry in manifest:
        path = entry["path"]
        size = entry["size"]
        if path.startswith("model_weights"):
            weights_bytes += size
        else:
            code_bytes += size

    total = code_bytes + weights_bytes
    warning = None
    if total > SIZE_ERROR_BYTES:
        warning = f"Package too large ({_human_readable_size(total)}), exceeds 10 GB limit"
    elif total > SIZE_WARNING_BYTES:
        warning = f"Package is large ({_human_readable_size(total)}), may be slow to upload"

    return {
        "code_bytes": code_bytes,
        "weights_bytes": weights_bytes,
        "total_bytes": total,
        "total_human": _human_readable_size(total),
        "warning": warning,
    }


def build_submission_zip(
    submission_dir: str | Path,
    output_path: str | Path,
    dry_run: bool = False,
) -> Path:
    """Build submission.zip from submission directory.

    Args:
        submission_dir: Path to submission/ directory.
        output_path: Where to write the zip file.
        dry_run: If True, validate but don't create zip.

    Returns:
        Path to the output zip file (even in dry-run mode).

    Raises:
        ValueError: If submission directory fails validation.
    """
    submission_dir = Path(submission_dir)
    output_path = Path(output_path)

    errors = validate_submission_dir(submission_dir)
    # Filter out model_weights warnings for build (it's not a hard error)
    hard_errors = [e for e in errors if "model_weights" not in e]
    if hard_errors:
        raise ValueError(
            f"Invalid submission directory: {'; '.join(hard_errors)}"
        )

    if dry_run:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = get_submission_manifest(submission_dir)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in manifest:
            file_path = submission_dir / entry["path"]
            zf.write(file_path, entry["path"])

    return output_path
