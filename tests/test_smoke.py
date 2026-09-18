import re
from pathlib import Path

import pytest

import qftppy


def test_version_is_valid():
    assert re.fullmatch(r"\d+\.\d+\.\d+", qftppy.__version__)


def test_installed_version_matches_pyproject():
    """Catches a stale install after release-please bumps the version in pyproject.toml."""
    tomllib = pytest.importorskip("tomllib")
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("not running from a source checkout")
    with pyproject.open("rb") as f:
        assert qftppy.__version__ == tomllib.load(f)["project"]["version"]
