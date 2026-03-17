"""Kaggle API wrapper for pushing notebooks, checking status, and pulling outputs.

Spec: S5.5
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

KAGGLE_CRED_PATH = Path.home() / ".kaggle" / "kaggle.json"


def check_kaggle_credentials() -> None:
    """Verify kaggle.json exists. Raises FileNotFoundError if missing."""
    if not KAGGLE_CRED_PATH.exists():
        raise FileNotFoundError(
            f"kaggle.json not found at {KAGGLE_CRED_PATH}. "
            "Download it from kaggle.com -> Settings -> API -> Create New Token"
        )


def create_kernel_metadata(
    kernel_slug: str,
    notebook_filename: str,
    kaggle_username: str,
    dataset_slugs: list[str] | None = None,
    title: str | None = None,
) -> dict:
    """Create kernel-metadata.json dict for Kaggle API."""
    return {
        "id": f"{kaggle_username}/{kernel_slug}",
        "title": title or kernel_slug,
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": dataset_slugs or [],
        "competition_sources": [],
        "kernel_sources": [],
    }


def _run_kaggle_cmd(cmd: list[str], retries: int = 3) -> subprocess.CompletedProcess:
    """Run a kaggle CLI command with retries on failure."""
    for attempt in range(retries):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result
        if attempt < retries - 1:
            wait = 2 ** (attempt + 1)
            logger.warning(
                "Kaggle command failed (attempt %d/%d), retrying in %ds: %s",
                attempt + 1,
                retries,
                wait,
                result.stderr[:200],
            )
            time.sleep(wait)
    return result


def kaggle_push(notebook_dir: str) -> dict:
    """Push a notebook to Kaggle and start the kernel.

    Args:
        notebook_dir: Directory containing kernel-metadata.json and .ipynb file.

    Returns:
        Dict with 'success' bool and 'message' string.
    """
    check_kaggle_credentials()
    cmd = ["kaggle", "kernels", "push", "-p", str(notebook_dir)]
    result = _run_kaggle_cmd(cmd)
    return {
        "success": result.returncode == 0,
        "message": result.stdout.strip() or result.stderr.strip(),
    }


def kaggle_status(kernel_slug: str) -> str:
    """Check the status of a Kaggle kernel.

    Returns one of: 'queued', 'running', 'complete', 'error', 'not_found'.
    """
    cmd = ["kaggle", "kernels", "status", kernel_slug]
    result = _run_kaggle_cmd(cmd, retries=1)

    if result.returncode != 0:
        if "404" in result.stderr or "Not Found" in result.stderr:
            return "not_found"
        return "error"

    try:
        data = json.loads(result.stdout)
        return data.get("status", "error")
    except json.JSONDecodeError:
        # Kaggle CLI sometimes outputs plain text
        stdout = result.stdout.strip().lower()
        for s in ("complete", "running", "queued", "error"):
            if s in stdout:
                return s
        return "error"


def kaggle_pull(kernel_slug: str, output_dir: str) -> dict:
    """Download output files from a completed Kaggle kernel.

    Returns:
        Dict with 'success' bool and 'message' string.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", kernel_slug, "-p", str(output_dir)]
    result = _run_kaggle_cmd(cmd)
    return {
        "success": result.returncode == 0,
        "message": result.stdout.strip() or result.stderr.strip(),
    }


def poll_until_complete(
    kernel_slug: str,
    poll_interval: int = 60,
    timeout: int = 36000,
) -> str:
    """Poll kernel status until complete or timeout.

    Returns final status string.
    """
    start = time.time()
    while time.time() - start < timeout:
        status = kaggle_status(kernel_slug)
        logger.info(
            "Kernel %s: %s (%.0fs elapsed)", kernel_slug, status, time.time() - start
        )
        if status in ("complete", "error", "not_found"):
            return status
        time.sleep(poll_interval)
    logger.warning("Timeout after %ds waiting for %s", timeout, kernel_slug)
    return "timeout"


def main():
    """CLI entrypoint for Kaggle runner."""
    parser = argparse.ArgumentParser(description="Kaggle kernel runner")
    sub = parser.add_subparsers(dest="command")

    push_cmd = sub.add_parser("push", help="Push notebook to Kaggle")
    push_cmd.add_argument("notebook_dir", help="Dir with kernel-metadata.json + .ipynb")

    status_cmd = sub.add_parser("status", help="Check kernel status")
    status_cmd.add_argument("kernel_slug", help="e.g. user/kernel-name")

    pull_cmd = sub.add_parser("pull", help="Pull kernel output")
    pull_cmd.add_argument("kernel_slug")
    pull_cmd.add_argument("-o", "--output-dir", default="output/sweep")

    args = parser.parse_args()

    if args.command == "push":
        result = kaggle_push(args.notebook_dir)
        print(result["message"])
    elif args.command == "status":
        status = kaggle_status(args.kernel_slug)
        print(f"Status: {status}")
    elif args.command == "pull":
        result = kaggle_pull(args.kernel_slug, args.output_dir)
        print(result["message"])
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
