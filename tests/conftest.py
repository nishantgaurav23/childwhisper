"""Shared test fixtures for ChildWhisper."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def configs_dir(project_root):
    """Return the configs directory."""
    return project_root / "configs"


@pytest.fixture
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def sample_audio_dir(data_dir):
    """Return the sample audio directory for local testing."""
    return data_dir / "audio_sample"
