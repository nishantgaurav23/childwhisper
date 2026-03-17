"""Final submission validator and pre-submission checklist.

Validates submission JSONL format, checks runtime environment,
verifies size budgets, runs dry-run tests, and performs pre-submission checks.

Spec: S5.4
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

try:
    from src.submission_builder import compute_size_budget
except ModuleNotFoundError:
    from submission_builder import compute_size_budget

# Packages required by the competition runtime
REQUIRED_PACKAGES = ["torch", "transformers", "peft", "librosa", "soundfile"]

# Patterns that indicate hardcoded absolute paths not under /code_execution
_BAD_PATH_RE = re.compile(
    r"""(['"])/(?!code_execution)[a-zA-Z][\w/.-]+\1"""
)

# Network-related imports that are forbidden in offline runtime
NETWORK_IMPORTS = {"requests", "urllib", "http.client", "httpx", "aiohttp", "urllib3"}


def validate_submission_output(
    jsonl_path: str | Path,
    metadata: list[dict],
) -> dict:
    """Validate submission.jsonl format and completeness.

    Args:
        jsonl_path: Path to the submission.jsonl file.
        metadata: List of utterance metadata dicts (must have 'utterance_id').

    Returns:
        Dict with keys: valid (bool), errors (list[str]), count (int).
    """
    jsonl_path = Path(jsonl_path)
    errors: list[str] = []

    if not jsonl_path.exists():
        return {"valid": False, "errors": ["File not found: " + str(jsonl_path)], "count": 0}

    text = jsonl_path.read_text().strip()
    if not text:
        return {"valid": False, "errors": ["File is empty"], "count": 0}

    expected_ids = {m["utterance_id"] for m in metadata}
    seen_ids: list[str] = []
    line_errors: list[str] = []

    for i, line in enumerate(text.split("\n"), start=1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            line_errors.append(f"Line {i}: invalid JSON — cannot parse")
            continue

        if "utterance_id" not in data:
            line_errors.append(f"Line {i}: missing 'utterance_id' field")
            continue

        uid = data["utterance_id"]
        seen_ids.append(uid)

        if "orthographic_text" not in data:
            line_errors.append(f"Line {i}: missing 'orthographic_text' field for {uid}")
            continue

        ot = data["orthographic_text"]
        if not isinstance(ot, str):
            line_errors.append(
                f"Line {i}: 'orthographic_text' must be a string, got {type(ot).__name__}"
            )

    errors.extend(line_errors)

    # Check for duplicates
    seen_set: set[str] = set()
    duplicates: set[str] = set()
    for uid in seen_ids:
        if uid in seen_set:
            duplicates.add(uid)
        seen_set.add(uid)
    if duplicates:
        errors.append(f"Duplicate utterance IDs: {sorted(duplicates)}")

    # Check for missing IDs
    missing = expected_ids - seen_set
    if missing:
        errors.append(f"Missing utterance IDs: {sorted(missing)}")

    # Check for extra IDs
    extra = seen_set - expected_ids
    if extra:
        errors.append(f"Extra/unexpected utterance IDs: {sorted(extra)}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "count": len(seen_ids),
    }


def check_runtime_environment() -> dict:
    """Check runtime environment readiness.

    Returns:
        Dict with keys: python_version (str), packages (dict[str, bool|str]),
        device (str).
    """
    import importlib

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    packages: dict[str, bool | str] = {}
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            packages[pkg] = version
        except ImportError:
            packages[pkg] = False

    # Determine device
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    except ImportError:
        pass

    return {
        "python_version": python_version,
        "packages": packages,
        "device": device,
    }


def validate_size_budget(submission_dir: str | Path) -> dict:
    """Validate submission package size.

    Returns:
        Dict with keys: valid (bool), code_bytes (int), weights_bytes (int),
        total_bytes (int), total_human (str), warning (str|None).
    """
    submission_dir = Path(submission_dir)

    if not submission_dir.exists():
        return {
            "valid": False,
            "code_bytes": 0,
            "weights_bytes": 0,
            "total_bytes": 0,
            "total_human": "0 B",
            "warning": f"Directory not found: {submission_dir}",
        }

    budget = compute_size_budget(submission_dir)

    # 10 GB hard limit
    valid = budget["total_bytes"] <= 10 * 1024 * 1024 * 1024

    return {
        "valid": valid,
        "code_bytes": budget["code_bytes"],
        "weights_bytes": budget["weights_bytes"],
        "total_bytes": budget["total_bytes"],
        "total_human": budget["total_human"],
        "warning": budget["warning"],
    }


def run_dry_run(data_dir: str | Path, output_dir: str | Path) -> dict:
    """Run a mock dry-run of the inference pipeline.

    Reads metadata, writes a dummy submission.jsonl with empty predictions.
    Does NOT load real models — this is a format/pipeline validation only.

    Returns:
        Dict with keys: success (bool), output_path (str|None),
        elapsed_sec (float), error (str|None).
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    t0 = time.time()

    meta_path = data_dir / "train_word_transcripts.jsonl"
    if not meta_path.exists():
        return {
            "success": False,
            "output_path": None,
            "elapsed_sec": time.time() - t0,
            "error": f"Metadata file not found: {meta_path}",
        }

    try:
        text = meta_path.read_text().strip()
        utterances = [json.loads(line) for line in text.split("\n") if line.strip()]
    except Exception as exc:
        return {
            "success": False,
            "output_path": None,
            "elapsed_sec": time.time() - t0,
            "error": f"Failed to parse metadata: {exc}",
        }

    # Write dummy predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.jsonl"

    with open(output_path, "w") as f:
        for u in utterances:
            line = {
                "utterance_id": u["utterance_id"],
                "orthographic_text": "",
            }
            f.write(json.dumps(line) + "\n")

    elapsed = time.time() - t0
    return {
        "success": True,
        "output_path": str(output_path),
        "elapsed_sec": elapsed,
        "error": None,
    }


def _check_entrypoint(main_py: Path) -> bool:
    """Check if main.py has an if __name__ == '__main__' block."""
    content = main_py.read_text()
    return '__name__' in content and '__main__' in content


def _check_hardcoded_paths(main_py: Path) -> list[str]:
    """Check for hardcoded absolute paths outside /code_execution/."""
    content = main_py.read_text()
    issues = []
    for match in _BAD_PATH_RE.finditer(content):
        path_str = match.group(0)
        # Skip common safe patterns
        if "/code_execution/" in path_str:
            continue
        issues.append(f"Hardcoded path found: {path_str}")
    return issues


def _check_network_imports(main_py: Path) -> list[str]:
    """Check for network-related imports in main.py."""
    content = main_py.read_text()
    issues = []
    for net_pkg in NETWORK_IMPORTS:
        # Match 'import requests' or 'from requests import ...'
        pattern = rf"(?:^|\n)\s*(?:import\s+{re.escape(net_pkg)}|from\s+{re.escape(net_pkg)})"
        if re.search(pattern, content):
            issues.append(f"Network import found: {net_pkg}")
    return issues


def _check_pycache(submission_dir: Path) -> list[str]:
    """Check for __pycache__ directories."""
    issues = []
    for p in submission_dir.rglob("__pycache__"):
        if p.is_dir():
            issues.append(f"__pycache__ found: {p.relative_to(submission_dir)}")
    return issues


def run_prechecks(submission_dir: str | Path) -> dict:
    """Run pre-submission checklist on the submission directory.

    Returns:
        Dict with keys: passed (bool), checks (list[dict]), failures (list[str]).
    """
    submission_dir = Path(submission_dir)
    checks: list[dict] = []
    failures: list[str] = []

    if not submission_dir.exists():
        return {
            "passed": False,
            "checks": [{"name": "directory_exists", "passed": False}],
            "failures": [f"Submission directory not found: {submission_dir}"],
        }

    # Check required files
    required_files = {
        "main.py": submission_dir / "main.py",
        "src/preprocess.py": submission_dir / "src" / "preprocess.py",
        "src/utils.py": submission_dir / "src" / "utils.py",
    }
    for name, path in required_files.items():
        exists = path.exists()
        checks.append({"name": f"file_{name}", "passed": exists})
        if not exists:
            failures.append(f"Required file missing: {name}")

    main_py = submission_dir / "main.py"
    if main_py.exists():
        # Entrypoint check
        has_entrypoint = _check_entrypoint(main_py)
        checks.append({"name": "entrypoint", "passed": has_entrypoint})
        if not has_entrypoint:
            failures.append("main.py missing __main__ entrypoint")

        # Hardcoded paths check
        path_issues = _check_hardcoded_paths(main_py)
        no_hardcoded = len(path_issues) == 0
        checks.append({"name": "no_hardcoded_paths", "passed": no_hardcoded})
        if not no_hardcoded:
            for issue in path_issues:
                failures.append(issue)

        # Network imports check
        net_issues = _check_network_imports(main_py)
        no_network = len(net_issues) == 0
        checks.append({"name": "no_network_imports", "passed": no_network})
        if not no_network:
            for issue in net_issues:
                failures.append(issue)

    # __pycache__ check
    cache_issues = _check_pycache(submission_dir)
    no_pycache = len(cache_issues) == 0
    checks.append({"name": "no_pycache", "passed": no_pycache})

    # model_weights check
    weights_dir = submission_dir / "model_weights"
    has_weights = weights_dir.is_dir()
    checks.append({"name": "model_weights_dir", "passed": has_weights})

    return {
        "passed": len(failures) == 0,
        "checks": checks,
        "failures": failures,
    }
