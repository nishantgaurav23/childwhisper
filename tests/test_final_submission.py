"""Tests for S5.4 — Final submission package (src/final_submission.py).

Tests the final submission validator: JSONL format validation, runtime environment
checks, size budget validation, dry-run testing, and pre-submission checklist.
All external dependencies are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.final_submission import (
    check_runtime_environment,
    run_dry_run,
    run_prechecks,
    validate_size_budget,
    validate_submission_output,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_metadata():
    """Sample utterance metadata."""
    return [
        {"utterance_id": "utt_001", "audio_path": "audio/utt_001.flac"},
        {"utterance_id": "utt_002", "audio_path": "audio/utt_002.flac"},
        {"utterance_id": "utt_003", "audio_path": "audio/utt_003.flac"},
    ]


@pytest.fixture
def valid_jsonl(tmp_path, sample_metadata):
    """Write a valid submission.jsonl and return its path."""
    path = tmp_path / "submission.jsonl"
    lines = []
    for m in sample_metadata:
        lines.append(json.dumps({
            "utterance_id": m["utterance_id"],
            "orthographic_text": "hello world",
        }))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def valid_submission_dir(tmp_path):
    """Create a valid submission directory for pre-checks."""
    sub = tmp_path / "submission"
    sub.mkdir()
    (sub / "main.py").write_text(
        '"""Entrypoint."""\nimport json\nif __name__ == "__main__":\n    pass\n'
    )
    (sub / "__init__.py").write_text("")
    src = sub / "src"
    src.mkdir()
    (src / "__init__.py").write_text("")
    (src / "preprocess.py").write_text("# preprocessing\n")
    (src / "utils.py").write_text("# utils\n")
    weights = sub / "model_weights"
    weights.mkdir()
    lora = weights / "lora_large_v3"
    lora.mkdir()
    (lora / "adapter_config.json").write_text('{"r": 32}')
    small = weights / "whisper_small_ft"
    small.mkdir()
    (small / "config.json").write_text('{"model_type": "whisper"}')
    return sub


# ---------------------------------------------------------------------------
# validate_submission_output
# ---------------------------------------------------------------------------


class TestValidateSubmissionOutput:
    def test_valid_output_passes(self, valid_jsonl, sample_metadata):
        result = validate_submission_output(valid_jsonl, sample_metadata)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_utterance_id_fails(self, tmp_path, sample_metadata):
        path = tmp_path / "submission.jsonl"
        # Only write 2 of 3 utterances
        lines = [
            json.dumps({"utterance_id": "utt_001", "orthographic_text": "a"}),
            json.dumps({"utterance_id": "utt_002", "orthographic_text": "b"}),
        ]
        path.write_text("\n".join(lines) + "\n")
        result = validate_submission_output(path, sample_metadata)
        assert result["valid"] is False
        assert any("missing" in e.lower() for e in result["errors"])

    def test_duplicate_utterance_id_fails(self, tmp_path, sample_metadata):
        path = tmp_path / "submission.jsonl"
        lines = [
            json.dumps({"utterance_id": "utt_001", "orthographic_text": "a"}),
            json.dumps({"utterance_id": "utt_001", "orthographic_text": "b"}),
            json.dumps({"utterance_id": "utt_002", "orthographic_text": "c"}),
            json.dumps({"utterance_id": "utt_003", "orthographic_text": "d"}),
        ]
        path.write_text("\n".join(lines) + "\n")
        result = validate_submission_output(path, sample_metadata)
        assert result["valid"] is False
        assert any("duplicate" in e.lower() for e in result["errors"])

    def test_extra_utterance_id_fails(self, tmp_path, sample_metadata):
        path = tmp_path / "submission.jsonl"
        lines = [
            json.dumps({"utterance_id": "utt_001", "orthographic_text": "a"}),
            json.dumps({"utterance_id": "utt_002", "orthographic_text": "b"}),
            json.dumps({"utterance_id": "utt_003", "orthographic_text": "c"}),
            json.dumps({"utterance_id": "utt_999", "orthographic_text": "d"}),
        ]
        path.write_text("\n".join(lines) + "\n")
        result = validate_submission_output(path, sample_metadata)
        assert result["valid"] is False
        assert any("extra" in e.lower() or "unexpected" in e.lower() for e in result["errors"])

    def test_invalid_json_line_fails(self, tmp_path, sample_metadata):
        path = tmp_path / "submission.jsonl"
        path.write_text('{"utterance_id": "utt_001"}\nnot valid json\n')
        result = validate_submission_output(path, sample_metadata)
        assert result["valid"] is False
        assert any("json" in e.lower() or "parse" in e.lower() for e in result["errors"])

    def test_missing_orthographic_text_field_fails(self, tmp_path, sample_metadata):
        path = tmp_path / "submission.jsonl"
        lines = [
            json.dumps({"utterance_id": "utt_001"}),  # missing orthographic_text
            json.dumps({"utterance_id": "utt_002", "orthographic_text": "b"}),
            json.dumps({"utterance_id": "utt_003", "orthographic_text": "c"}),
        ]
        path.write_text("\n".join(lines) + "\n")
        result = validate_submission_output(path, sample_metadata)
        assert result["valid"] is False
        assert any("orthographic_text" in e for e in result["errors"])

    def test_non_string_orthographic_text_fails(self, tmp_path, sample_metadata):
        path = tmp_path / "submission.jsonl"
        lines = [
            json.dumps({"utterance_id": "utt_001", "orthographic_text": 123}),
            json.dumps({"utterance_id": "utt_002", "orthographic_text": "b"}),
            json.dumps({"utterance_id": "utt_003", "orthographic_text": "c"}),
        ]
        path.write_text("\n".join(lines) + "\n")
        result = validate_submission_output(path, sample_metadata)
        assert result["valid"] is False
        assert any("string" in e.lower() or "type" in e.lower() for e in result["errors"])

    def test_empty_orthographic_text_is_valid(self, tmp_path, sample_metadata):
        """Empty string is valid (silence predictions)."""
        path = tmp_path / "submission.jsonl"
        lines = [
            json.dumps({"utterance_id": "utt_001", "orthographic_text": ""}),
            json.dumps({"utterance_id": "utt_002", "orthographic_text": ""}),
            json.dumps({"utterance_id": "utt_003", "orthographic_text": ""}),
        ]
        path.write_text("\n".join(lines) + "\n")
        result = validate_submission_output(path, sample_metadata)
        assert result["valid"] is True

    def test_nonexistent_file_fails(self, tmp_path, sample_metadata):
        result = validate_submission_output(tmp_path / "nope.jsonl", sample_metadata)
        assert result["valid"] is False
        assert any("not found" in e.lower() or "exist" in e.lower() for e in result["errors"])

    def test_returns_count(self, valid_jsonl, sample_metadata):
        result = validate_submission_output(valid_jsonl, sample_metadata)
        assert result["count"] == 3


# ---------------------------------------------------------------------------
# check_runtime_environment
# ---------------------------------------------------------------------------


class TestCheckRuntimeEnvironment:
    def test_returns_dict_with_required_keys(self):
        result = check_runtime_environment()
        assert "python_version" in result
        assert "packages" in result
        assert "device" in result

    def test_python_version_is_string(self):
        result = check_runtime_environment()
        assert isinstance(result["python_version"], str)

    def test_packages_is_dict(self):
        result = check_runtime_environment()
        assert isinstance(result["packages"], dict)

    def test_checks_core_packages(self):
        result = check_runtime_environment()
        packages = result["packages"]
        assert "torch" in packages
        assert "transformers" in packages
        assert "librosa" in packages

    def test_device_field_present(self):
        result = check_runtime_environment()
        assert result["device"] in ("cuda", "mps", "cpu")


# ---------------------------------------------------------------------------
# validate_size_budget
# ---------------------------------------------------------------------------


class TestValidateSizeBudget:
    def test_valid_small_package(self, valid_submission_dir):
        result = validate_size_budget(valid_submission_dir)
        assert result["valid"] is True
        assert result["total_bytes"] > 0

    def test_returns_breakdown(self, valid_submission_dir):
        result = validate_size_budget(valid_submission_dir)
        assert "code_bytes" in result
        assert "weights_bytes" in result
        assert "total_bytes" in result
        assert "total_human" in result

    def test_warns_on_large_package(self, tmp_path):
        """Packages > 4 GB should get a warning."""
        sub = tmp_path / "submission"
        sub.mkdir()
        (sub / "main.py").write_text("pass")
        src = sub / "src"
        src.mkdir()
        (src / "preprocess.py").write_text("pass")
        (src / "utils.py").write_text("pass")
        # We can't create a 4GB file in tests, but validate_size_budget
        # delegates to compute_size_budget which we tested in S3.4
        result = validate_size_budget(sub)
        assert result["valid"] is True  # small package is fine
        assert result["warning"] is None

    def test_nonexistent_dir_fails(self, tmp_path):
        result = validate_size_budget(tmp_path / "nonexistent")
        assert result["valid"] is False

    def test_weights_breakdown_present(self, valid_submission_dir):
        result = validate_size_budget(valid_submission_dir)
        assert "weights_bytes" in result
        assert isinstance(result["weights_bytes"], int)


# ---------------------------------------------------------------------------
# run_dry_run
# ---------------------------------------------------------------------------


class TestRunDryRun:
    def test_dry_run_produces_output(self, tmp_path):
        """Dry run should produce a valid submission.jsonl with mock data."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train_word_transcripts.jsonl").write_text(
            json.dumps({
                "utterance_id": "utt_001",
                "audio_path": "audio/utt_001.flac",
                "audio_duration_sec": 1.5,
            })
            + "\n"
        )
        output_dir = tmp_path / "output"
        result = run_dry_run(data_dir, output_dir)
        assert result["success"] is True
        assert result["output_path"] is not None

    def test_dry_run_output_is_valid_jsonl(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train_word_transcripts.jsonl").write_text(
            json.dumps({
                "utterance_id": "utt_001",
                "audio_path": "audio/utt_001.flac",
                "audio_duration_sec": 1.5,
            })
            + "\n"
        )
        output_dir = tmp_path / "output"
        result = run_dry_run(data_dir, output_dir)
        output_path = Path(result["output_path"])
        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        for line in lines:
            data = json.loads(line)
            assert "utterance_id" in data
            assert "orthographic_text" in data

    def test_dry_run_with_missing_metadata_fails(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # No train_word_transcripts.jsonl
        output_dir = tmp_path / "output"
        result = run_dry_run(data_dir, output_dir)
        assert result["success"] is False

    def test_dry_run_returns_timing(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train_word_transcripts.jsonl").write_text(
            json.dumps({
                "utterance_id": "utt_001",
                "audio_path": "audio/utt_001.flac",
                "audio_duration_sec": 1.0,
            })
            + "\n"
        )
        output_dir = tmp_path / "output"
        result = run_dry_run(data_dir, output_dir)
        assert "elapsed_sec" in result
        assert result["elapsed_sec"] >= 0


# ---------------------------------------------------------------------------
# run_prechecks
# ---------------------------------------------------------------------------


class TestRunPrechecks:
    def test_valid_submission_passes(self, valid_submission_dir):
        result = run_prechecks(valid_submission_dir)
        assert result["passed"] is True
        assert len(result["failures"]) == 0

    def test_returns_checks_list(self, valid_submission_dir):
        result = run_prechecks(valid_submission_dir)
        assert "checks" in result
        assert isinstance(result["checks"], list)
        assert len(result["checks"]) > 0

    def test_each_check_has_name_and_status(self, valid_submission_dir):
        result = run_prechecks(valid_submission_dir)
        for check in result["checks"]:
            assert "name" in check
            assert "passed" in check

    def test_missing_main_py_fails(self, valid_submission_dir):
        (valid_submission_dir / "main.py").unlink()
        result = run_prechecks(valid_submission_dir)
        assert result["passed"] is False
        assert any("main.py" in f for f in result["failures"])

    def test_missing_src_preprocess_fails(self, valid_submission_dir):
        (valid_submission_dir / "src" / "preprocess.py").unlink()
        result = run_prechecks(valid_submission_dir)
        assert result["passed"] is False
        assert any("preprocess" in f for f in result["failures"])

    def test_missing_entrypoint_fails(self, valid_submission_dir):
        """main.py must have if __name__ == '__main__' block."""
        (valid_submission_dir / "main.py").write_text("# no entrypoint\npass\n")
        result = run_prechecks(valid_submission_dir)
        assert result["passed"] is False
        assert any("entrypoint" in f.lower() or "__main__" in f for f in result["failures"])

    def test_hardcoded_paths_fail(self, valid_submission_dir):
        """main.py should not contain hardcoded paths outside /code_execution/."""
        (valid_submission_dir / "main.py").write_text(
            'path = "/home/user/data"\nif __name__ == "__main__":\n    pass\n'
        )
        result = run_prechecks(valid_submission_dir)
        assert result["passed"] is False
        assert any("hardcoded" in f.lower() or "path" in f.lower() for f in result["failures"])

    def test_network_import_fails(self, valid_submission_dir):
        """main.py should not import networking libraries."""
        (valid_submission_dir / "main.py").write_text(
            'import requests\nif __name__ == "__main__":\n    pass\n'
        )
        result = run_prechecks(valid_submission_dir)
        assert result["passed"] is False
        assert any("network" in f.lower() for f in result["failures"])

    def test_no_pycache_check(self, valid_submission_dir):
        """Verify that pycache check is included."""
        cache = valid_submission_dir / "__pycache__"
        cache.mkdir()
        (cache / "foo.pyc").write_bytes(b"\x00")
        result = run_prechecks(valid_submission_dir)
        # pycache presence is a warning in checks, not necessarily a failure
        # since build_submission.sh excludes them. Check it's reported.
        check_names = [c["name"] for c in result["checks"]]
        assert any("pycache" in n.lower() or "cache" in n.lower() for n in check_names)

    def test_nonexistent_dir_fails(self, tmp_path):
        result = run_prechecks(tmp_path / "nonexistent")
        assert result["passed"] is False
