"""Tests for Kaggle API wrapper (push/status/pull)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class TestKagglePush:
    """Tests for FR-3: Kaggle API push."""

    def test_kaggle_push_calls_cli(self, tmp_path):
        """Mock subprocess — verify kaggle kernels push is called correctly."""
        from src.kaggle_runner import kaggle_push

        # Create a fake notebook dir with kernel-metadata.json
        nb_dir = tmp_path / "trial_001"
        nb_dir.mkdir()
        (nb_dir / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": "user/kernel-name",
                    "code_file": "notebook.ipynb",
                }
            )
        )
        (nb_dir / "notebook.ipynb").write_text("{}")

        with (
            patch("subprocess.run") as mock_run,
            patch("src.kaggle_runner.check_kaggle_credentials"),
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Kernel push successful", stderr=""
            )
            result = kaggle_push(str(nb_dir))

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0] if call_args[0] else call_args.kwargs.get("args", [])
        assert "kaggle" in cmd
        assert "kernels" in cmd
        assert "push" in cmd
        assert result["success"] is True


class TestKaggleStatus:
    """Tests for FR-3: Kaggle API status."""

    @pytest.mark.parametrize(
        "status_text,expected",
        [
            ("running", "running"),
            ("complete", "complete"),
            ("error", "error"),
            ("queued", "queued"),
        ],
    )
    def test_kaggle_status_parses_output(self, status_text, expected):
        """Mock subprocess — verify status parsing for various states."""
        from src.kaggle_runner import kaggle_status

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=f'{{"status": "{status_text}"}}',
                stderr="",
            )
            status = kaggle_status("user/kernel-name")

        assert status == expected

    def test_kaggle_status_kernel_not_found(self):
        """Kernel not found returns 'not_found'."""
        from src.kaggle_runner import kaggle_status

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="404 - Not Found",
            )
            status = kaggle_status("user/nonexistent-kernel")

        assert status == "not_found"


class TestKagglePull:
    """Tests for FR-3: Kaggle API pull."""

    def test_kaggle_pull_downloads_files(self, tmp_path):
        """Mock subprocess — verify output files are saved to correct dir."""
        from src.kaggle_runner import kaggle_pull

        output_dir = tmp_path / "output"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="Output downloaded", stderr=""
            )
            result = kaggle_pull("user/kernel-name", str(output_dir))

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0] if call_args[0] else call_args.kwargs.get("args", [])
        assert "kaggle" in cmd
        assert "kernels" in cmd
        assert "output" in cmd
        assert result["success"] is True


class TestKaggleCredentials:
    """Tests for credential handling."""

    def test_kaggle_missing_credentials(self):
        """Raises clear error when kaggle.json is missing."""
        from src.kaggle_runner import check_kaggle_credentials

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="kaggle.json"):
                check_kaggle_credentials()


class TestKernelMetadata:
    """Tests for kernel metadata generation."""

    def test_kernel_metadata_format(self):
        """Generated kernel-metadata.json matches Kaggle schema."""
        from src.kaggle_runner import create_kernel_metadata

        meta = create_kernel_metadata(
            kernel_slug="childwhisper-sweep-trial-003",
            notebook_filename="sweep_trial_003.ipynb",
            kaggle_username="nishantgaurav23",
            dataset_slugs=["nishantgaurav23/pasketti-audio"],
            title="ChildWhisper Sweep Trial 003",
        )
        assert meta["id"] == "nishantgaurav23/childwhisper-sweep-trial-003"
        assert meta["title"] == "ChildWhisper Sweep Trial 003"
        assert meta["code_file"] == "sweep_trial_003.ipynb"
        assert meta["language"] == "python"
        assert meta["kernel_type"] == "notebook"
        assert meta["is_private"] is True
        assert meta["enable_gpu"] is True
        assert meta["enable_internet"] is True
        assert meta["dataset_sources"] == ["nishantgaurav23/pasketti-audio"]
        assert meta["competition_sources"] == []
        assert meta["kernel_sources"] == []
