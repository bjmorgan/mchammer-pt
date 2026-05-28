"""Version metadata tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

import mchammer_pt


def test_version_matches_pyproject():
    """``mchammer_pt.__version__`` matches the version declared in
    pyproject.toml.

    ``__version__`` is derived from the installed distribution
    metadata (``importlib.metadata``), so this guards against the
    metadata drifting from the source declaration. A local failure
    here usually means the editable install is stale; re-run
    ``pip install -e .``.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert mchammer_pt.__version__ == declared
