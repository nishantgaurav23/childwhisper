"""Tests for S1.1 — Project Structure & Dependencies."""

import os
import stat
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDirectoryStructure:
    """Verify all expected directories exist."""

    @pytest.mark.parametrize(
        "dir_path",
        [
            "src",
            "tests",
            "configs",
            "scripts",
            "notebooks",
            "submission",
            "submission/model_weights",
            "submission/utils",
            "data",
            "specs",
        ],
    )
    def test_directory_exists(self, dir_path):
        assert (PROJECT_ROOT / dir_path).is_dir(), f"Directory {dir_path}/ missing"


class TestPythonPackages:
    """Verify Python packages are importable via __init__.py."""

    @pytest.mark.parametrize(
        "init_path",
        [
            "src/__init__.py",
            "tests/__init__.py",
            "submission/__init__.py",
            "submission/utils/__init__.py",
        ],
    )
    def test_init_file_exists(self, init_path):
        assert (PROJECT_ROOT / init_path).is_file(), f"{init_path} missing"


class TestRequirements:
    """Verify requirements.txt exists and contains expected packages."""

    def test_requirements_file_exists(self):
        assert (PROJECT_ROOT / "requirements.txt").is_file()

    def test_requirements_not_empty(self):
        content = (PROJECT_ROOT / "requirements.txt").read_text()
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert len(lines) > 0, "requirements.txt is empty"

    @pytest.mark.parametrize(
        "package",
        [
            "torch",
            "torchaudio",
            "transformers",
            "peft",
            "accelerate",
            "bitsandbytes",
            "datasets",
            "librosa",
            "soundfile",
            "jiwer",
            "audiomentations",
            "pytest",
            "ruff",
        ],
    )
    def test_required_package_listed(self, package):
        content = (PROJECT_ROOT / "requirements.txt").read_text().lower()
        assert package in content, f"Package '{package}' not found in requirements.txt"


class TestTrainingConfig:
    """Verify training configuration is valid YAML with expected structure."""

    @pytest.fixture
    def config(self):
        config_path = PROJECT_ROOT / "configs" / "training_config.yaml"
        assert config_path.is_file(), "configs/training_config.yaml missing"
        return yaml.safe_load(config_path.read_text())

    def test_config_has_whisper_small_section(self, config):
        assert "whisper_small" in config, "Missing 'whisper_small' section"

    def test_config_has_whisper_large_section(self, config):
        assert "whisper_large_v3" in config, "Missing 'whisper_large_v3' section"

    def test_whisper_small_has_learning_rate(self, config):
        assert "learning_rate" in config["whisper_small"]

    def test_whisper_large_has_lora_config(self, config):
        assert "lora" in config["whisper_large_v3"]

    def test_whisper_large_lora_rank(self, config):
        assert config["whisper_large_v3"]["lora"]["r"] == 32

    def test_whisper_large_lora_alpha(self, config):
        assert config["whisper_large_v3"]["lora"]["alpha"] == 64

    def test_config_has_common_section(self, config):
        assert "common" in config, "Missing 'common' section"

    def test_common_sample_rate(self, config):
        assert config["common"]["sample_rate"] == 16000

    def test_common_has_spec_augment(self, config):
        assert "spec_augment" in config["common"]


class TestShellScripts:
    """Verify shell scripts exist and are executable."""

    @pytest.mark.parametrize(
        "script",
        [
            "scripts/download_data.sh",
            "scripts/download_weights.sh",
            "scripts/build_submission.sh",
        ],
    )
    def test_script_exists(self, script):
        assert (PROJECT_ROOT / script).is_file(), f"{script} missing"

    @pytest.mark.parametrize(
        "script",
        [
            "scripts/download_data.sh",
            "scripts/download_weights.sh",
            "scripts/build_submission.sh",
        ],
    )
    def test_script_is_executable(self, script):
        path = PROJECT_ROOT / script
        mode = os.stat(path).st_mode
        assert mode & stat.S_IXUSR, f"{script} is not executable"

    @pytest.mark.parametrize(
        "script",
        [
            "scripts/download_data.sh",
            "scripts/download_weights.sh",
            "scripts/build_submission.sh",
        ],
    )
    def test_script_has_shebang(self, script):
        content = (PROJECT_ROOT / script).read_text()
        assert content.startswith("#!/"), f"{script} missing shebang line"


class TestGitignore:
    """Verify .gitignore covers critical patterns."""

    @pytest.fixture
    def gitignore(self):
        return (PROJECT_ROOT / ".gitignore").read_text()

    @pytest.mark.parametrize(
        "pattern",
        ["data/", ".env", "__pycache__", "*.py[cod]", ".venv/", "submission/model_weights/"],
    )
    def test_critical_pattern_present(self, gitignore, pattern):
        assert pattern in gitignore, f"Pattern '{pattern}' missing from .gitignore"


class TestConftest:
    """Verify tests/conftest.py exists with shared fixtures."""

    def test_conftest_exists(self):
        assert (PROJECT_ROOT / "tests" / "conftest.py").is_file()

    def test_conftest_has_project_root_fixture(self):
        content = (PROJECT_ROOT / "tests" / "conftest.py").read_text()
        assert "project_root" in content
