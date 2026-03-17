"""Tests for S3.4 — Submission zip builder (src/submission_builder.py).

Tests the submission packaging pipeline: directory validation, manifest
generation, size budgeting, and zip creation. All filesystem-based, no
external services or model loading.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.submission_builder import (
    build_submission_zip,
    compute_size_budget,
    get_excludes,
    get_submission_manifest,
    validate_submission_dir,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_submission(tmp_path):
    """Create a minimal valid submission directory structure."""
    sub = tmp_path / "submission"
    sub.mkdir()

    # main.py
    (sub / "main.py").write_text("# entrypoint\nif __name__ == '__main__': pass\n")

    # __init__.py
    (sub / "__init__.py").write_text("")

    # src/ with required modules
    src = sub / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "preprocess.py").write_text("# audio preprocessing\n")
    (src / "utils.py").write_text("# text normalization\n")

    # model_weights/ directory
    weights = sub / "model_weights"
    weights.mkdir()

    # utils/ directory
    utils = sub / "utils"
    utils.mkdir()
    (utils / "__init__.py").write_text("")

    return sub


@pytest.fixture
def submission_with_weights(valid_submission):
    """Valid submission with fake model weight files."""
    lora = valid_submission / "model_weights" / "lora_large_v3"
    lora.mkdir()
    (lora / "adapter_config.json").write_text('{"r": 32}')
    (lora / "adapter_model.safetensors").write_bytes(b"\x00" * 1024)

    small = valid_submission / "model_weights" / "whisper_small_ft"
    small.mkdir()
    (small / "config.json").write_text('{"model_type": "whisper"}')
    (small / "model.safetensors").write_bytes(b"\x00" * 2048)

    return valid_submission


@pytest.fixture
def submission_with_pycache(valid_submission):
    """Valid submission with __pycache__ and .pyc files."""
    cache = valid_submission / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-311.pyc").write_bytes(b"\x00" * 100)
    (valid_submission / "src" / "__pycache__").mkdir()
    (valid_submission / "src" / "__pycache__" / "utils.cpython-311.pyc").write_bytes(
        b"\x00" * 50
    )
    (valid_submission / ".DS_Store").write_bytes(b"\x00" * 10)
    return valid_submission


# ---------------------------------------------------------------------------
# validate_submission_dir
# ---------------------------------------------------------------------------

class TestValidateSubmissionDir:
    def test_valid_dir_returns_no_errors(self, valid_submission):
        errors = validate_submission_dir(valid_submission)
        assert errors == []

    def test_missing_main_py(self, valid_submission):
        (valid_submission / "main.py").unlink()
        errors = validate_submission_dir(valid_submission)
        assert any("main.py" in e for e in errors)

    def test_missing_src_dir(self, valid_submission):
        import shutil
        shutil.rmtree(valid_submission / "src")
        errors = validate_submission_dir(valid_submission)
        assert any("src" in e.lower() for e in errors)

    def test_missing_preprocess_py(self, valid_submission):
        (valid_submission / "src" / "preprocess.py").unlink()
        errors = validate_submission_dir(valid_submission)
        assert any("preprocess.py" in e for e in errors)

    def test_missing_utils_py(self, valid_submission):
        (valid_submission / "src" / "utils.py").unlink()
        errors = validate_submission_dir(valid_submission)
        assert any("utils.py" in e for e in errors)

    def test_missing_model_weights_warns(self, valid_submission):
        import shutil
        shutil.rmtree(valid_submission / "model_weights")
        errors = validate_submission_dir(valid_submission)
        # model_weights missing is a warning, not a hard error
        # (can test without weights)
        assert any("model_weights" in e for e in errors)

    def test_nonexistent_dir_returns_error(self, tmp_path):
        errors = validate_submission_dir(tmp_path / "nonexistent")
        assert len(errors) > 0
        assert any("not found" in e.lower() or "does not exist" in e.lower() for e in errors)

    def test_accepts_string_path(self, valid_submission):
        errors = validate_submission_dir(str(valid_submission))
        assert errors == []


# ---------------------------------------------------------------------------
# get_submission_manifest
# ---------------------------------------------------------------------------

class TestGetSubmissionManifest:
    def test_returns_list_of_dicts(self, valid_submission):
        manifest = get_submission_manifest(valid_submission)
        assert isinstance(manifest, list)
        assert all(isinstance(entry, dict) for entry in manifest)

    def test_entries_have_path_and_size(self, valid_submission):
        manifest = get_submission_manifest(valid_submission)
        for entry in manifest:
            assert "path" in entry
            assert "size" in entry

    def test_includes_main_py(self, valid_submission):
        manifest = get_submission_manifest(valid_submission)
        paths = [entry["path"] for entry in manifest]
        assert any("main.py" in p for p in paths)

    def test_includes_src_files(self, valid_submission):
        manifest = get_submission_manifest(valid_submission)
        paths = [entry["path"] for entry in manifest]
        assert any("preprocess.py" in p for p in paths)
        assert any("utils.py" in p for p in paths)

    def test_excludes_pycache(self, submission_with_pycache):
        manifest = get_submission_manifest(submission_with_pycache)
        paths = [entry["path"] for entry in manifest]
        assert not any("__pycache__" in p for p in paths)
        assert not any(".pyc" in p for p in paths)

    def test_excludes_ds_store(self, submission_with_pycache):
        manifest = get_submission_manifest(submission_with_pycache)
        paths = [entry["path"] for entry in manifest]
        assert not any(".DS_Store" in p for p in paths)

    def test_sizes_are_nonnegative_integers(self, valid_submission):
        manifest = get_submission_manifest(valid_submission)
        for entry in manifest:
            assert isinstance(entry["size"], int)
            assert entry["size"] >= 0

    def test_includes_weight_files(self, submission_with_weights):
        manifest = get_submission_manifest(submission_with_weights)
        paths = [entry["path"] for entry in manifest]
        assert any("adapter_config.json" in p for p in paths)
        assert any("model.safetensors" in p for p in paths)


# ---------------------------------------------------------------------------
# compute_size_budget
# ---------------------------------------------------------------------------

class TestComputeSizeBudget:
    def test_returns_dict_with_required_keys(self, valid_submission):
        budget = compute_size_budget(valid_submission)
        assert "code_bytes" in budget
        assert "weights_bytes" in budget
        assert "total_bytes" in budget

    def test_total_equals_code_plus_weights(self, submission_with_weights):
        budget = compute_size_budget(submission_with_weights)
        assert budget["total_bytes"] == budget["code_bytes"] + budget["weights_bytes"]

    def test_weights_bytes_counts_model_weights_dir(self, submission_with_weights):
        budget = compute_size_budget(submission_with_weights)
        # We wrote 1024 + 2048 bytes of fake weight data + json config files
        assert budget["weights_bytes"] > 0

    def test_code_bytes_counts_py_files(self, valid_submission):
        budget = compute_size_budget(valid_submission)
        assert budget["code_bytes"] > 0

    def test_empty_weights_dir_returns_zero(self, valid_submission):
        budget = compute_size_budget(valid_submission)
        assert budget["weights_bytes"] == 0

    def test_has_warning_field(self, valid_submission):
        budget = compute_size_budget(valid_submission)
        assert "warning" in budget

    def test_no_warning_for_small_package(self, valid_submission):
        budget = compute_size_budget(valid_submission)
        assert budget["warning"] is None

    def test_has_human_readable_total(self, submission_with_weights):
        budget = compute_size_budget(submission_with_weights)
        assert "total_human" in budget
        assert isinstance(budget["total_human"], str)


# ---------------------------------------------------------------------------
# get_excludes
# ---------------------------------------------------------------------------

class TestGetExcludes:
    def test_returns_list(self):
        excludes = get_excludes()
        assert isinstance(excludes, list)

    def test_excludes_pycache(self):
        excludes = get_excludes()
        assert any("__pycache__" in e for e in excludes)

    def test_excludes_pyc(self):
        excludes = get_excludes()
        assert any(".pyc" in e for e in excludes)

    def test_excludes_ds_store(self):
        excludes = get_excludes()
        assert any(".DS_Store" in e for e in excludes)

    def test_excludes_git(self):
        excludes = get_excludes()
        assert any(".git" in e for e in excludes)


# ---------------------------------------------------------------------------
# build_submission_zip
# ---------------------------------------------------------------------------

class TestBuildSubmissionZip:
    def test_creates_zip_file(self, valid_submission, tmp_path):
        output = tmp_path / "output" / "submission.zip"
        result = build_submission_zip(valid_submission, output)
        assert result.exists()
        assert result.suffix == ".zip"

    def test_zip_contains_main_py(self, valid_submission, tmp_path):
        output = tmp_path / "submission.zip"
        build_submission_zip(valid_submission, output)
        with zipfile.ZipFile(output, "r") as zf:
            names = zf.namelist()
            assert "main.py" in names

    def test_zip_contains_src_files(self, valid_submission, tmp_path):
        output = tmp_path / "submission.zip"
        build_submission_zip(valid_submission, output)
        with zipfile.ZipFile(output, "r") as zf:
            names = zf.namelist()
            assert any("src/preprocess.py" in n for n in names)
            assert any("src/utils.py" in n for n in names)

    def test_zip_excludes_pycache(self, submission_with_pycache, tmp_path):
        output = tmp_path / "submission.zip"
        build_submission_zip(submission_with_pycache, output)
        with zipfile.ZipFile(output, "r") as zf:
            names = zf.namelist()
            assert not any("__pycache__" in n for n in names)
            assert not any(".pyc" in n for n in names)

    def test_zip_excludes_ds_store(self, submission_with_pycache, tmp_path):
        output = tmp_path / "submission.zip"
        build_submission_zip(submission_with_pycache, output)
        with zipfile.ZipFile(output, "r") as zf:
            names = zf.namelist()
            assert not any(".DS_Store" in n for n in names)

    def test_dry_run_does_not_create_file(self, valid_submission, tmp_path):
        output = tmp_path / "submission.zip"
        result = build_submission_zip(valid_submission, output, dry_run=True)
        assert not output.exists()
        assert result == output  # still returns the path it would create

    def test_includes_weight_files(self, submission_with_weights, tmp_path):
        output = tmp_path / "submission.zip"
        build_submission_zip(submission_with_weights, output)
        with zipfile.ZipFile(output, "r") as zf:
            names = zf.namelist()
            assert any("adapter_config.json" in n for n in names)
            assert any("model.safetensors" in n for n in names)

    def test_raises_on_invalid_submission(self, tmp_path):
        """Should raise ValueError if submission dir is invalid."""
        bad_dir = tmp_path / "empty"
        bad_dir.mkdir()
        output = tmp_path / "submission.zip"
        with pytest.raises(ValueError):
            build_submission_zip(bad_dir, output)

    def test_creates_parent_dirs(self, valid_submission, tmp_path):
        output = tmp_path / "deep" / "nested" / "submission.zip"
        result = build_submission_zip(valid_submission, output)
        assert result.exists()

    def test_returns_path_object(self, valid_submission, tmp_path):
        output = tmp_path / "submission.zip"
        result = build_submission_zip(valid_submission, output)
        assert isinstance(result, Path)

    def test_zip_is_valid(self, valid_submission, tmp_path):
        output = tmp_path / "submission.zip"
        build_submission_zip(valid_submission, output)
        assert zipfile.is_zipfile(output)
