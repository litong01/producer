"""Pytest fixtures for pipeline tests."""
import os
from pathlib import Path

import pytest

# Project root (parent of tests/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# 童年.pdf is the example file for lyrics/title tests
TONGNIAN_PDF = REPO_ROOT / "童年.pdf"


def tongnian_pdf_path():
    """Path to 童年.pdf; None if file is not present."""
    if TONGNIAN_PDF.exists():
        return str(TONGNIAN_PDF)
    return None


@pytest.fixture
def repo_root():
    """Project root directory."""
    return REPO_ROOT


@pytest.fixture
def tongnian_pdf():
    """Path to 童年.pdf. Skips the test if the file is missing."""
    p = tongnian_pdf_path()
    if p is None:
        pytest.skip("童年.pdf not found (place it in the project root to run this test)")
    return p


@pytest.fixture
def temp_work_dir(tmp_path):
    """Temporary directory for pipeline output."""
    return str(tmp_path)
