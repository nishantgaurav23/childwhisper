"""Experiment loop orchestrator for AutoWhisper."""

import re
import subprocess
import sys
import time

from src.autowhisper.logger import append_result


def init_run(tag: str, base_branch: str = "main") -> str:
    """Create a new experiment branch and return its name."""
    branch_name = f"autowhisper/{tag}"
    subprocess.run(
        ["git", "checkout", "-b", branch_name],
        capture_output=True,
        text=True,
        check=False,
    )
    return branch_name


def run_experiment(train_script: str, time_budget: int = 900) -> dict:
    """Execute train.py, capture output, parse val_wer and peak_vram_mb.

    Returns dict with val_wer, peak_vram_mb, duration_sec, status, stdout, stderr.
    Timeout: time_budget + 60s grace period.
    """
    timeout = time_budget + 60
    start = time.time()

    try:
        proc = subprocess.run(
            [sys.executable, train_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = int(time.time() - start)
    except subprocess.TimeoutExpired:
        elapsed = int(time.time() - start)
        return {
            "val_wer": -1.0,
            "peak_vram_mb": -1,
            "duration_sec": elapsed,
            "status": "crash",
            "stdout": "",
            "stderr": f"Timeout after {timeout}s",
        }

    if proc.returncode != 0:
        return {
            "val_wer": -1.0,
            "peak_vram_mb": -1,
            "duration_sec": elapsed,
            "status": "crash",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    # Parse val_wer from stdout
    val_wer = -1.0
    wer_match = re.search(r"val_wer:\s*([\d.]+)", proc.stdout)
    if wer_match:
        val_wer = float(wer_match.group(1))

    # Parse peak_vram_mb from stdout
    peak_vram = -1
    vram_match = re.search(r"peak_vram_mb:\s*(\d+)", proc.stdout)
    if vram_match:
        peak_vram = int(vram_match.group(1))

    return {
        "val_wer": val_wer,
        "peak_vram_mb": peak_vram,
        "duration_sec": elapsed,
        "status": "ok",
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def evaluate_and_decide(result: dict, best_wer: float) -> str:
    """Decide whether to keep, discard, or mark as crash.

    Returns 'keep' if val_wer < best_wer, 'discard' if >= best_wer, 'crash' if crashed.
    """
    if result["status"] == "crash" or result["val_wer"] < 0:
        return "crash"
    if result["val_wer"] < best_wer:
        return "keep"
    return "discard"


def keep_experiment(description: str) -> None:
    """Git commit the current changes with description."""
    subprocess.run(
        ["git", "add", "-A"],
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        ["git", "commit", "-m", f"[autowhisper] {description}"],
        capture_output=True,
        text=True,
        check=False,
    )


def revert_experiment() -> None:
    """Revert the last experiment's changes (git checkout on train.py)."""
    subprocess.run(
        ["git", "checkout", "--", "src/autowhisper/train.py"],
        capture_output=True,
        text=True,
        check=False,
    )


def log_result(
    result: dict,
    decision: str,
    description: str,
    experiment_id: str,
    commit_hash: str,
    log_path: str,
) -> None:
    """Append experiment result to the TSV log."""
    row = {
        "experiment_id": experiment_id,
        "commit_hash": commit_hash,
        "val_wer": result["val_wer"],
        "peak_vram_mb": result["peak_vram_mb"],
        "duration_sec": result["duration_sec"],
        "status": decision,
        "description": description,
    }
    append_result(log_path, row)
