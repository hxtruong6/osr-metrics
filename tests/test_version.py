"""Tests for osr_metrics.__version__ wiring."""
import re

import osr_metrics


def test_version_is_string():
    assert isinstance(osr_metrics.__version__, str)
    assert osr_metrics.__version__  # non-empty


def test_version_matches_pyproject():
    """__version__ must equal the version declared in pyproject.toml.

    When the package is installed (editable or wheel), importlib.metadata
    returns the version setuptools recorded at install time. Mismatch means
    someone bumped pyproject.toml without reinstalling.
    """
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert osr_metrics.__version__ == declared, (
        f"__version__={osr_metrics.__version__!r} but pyproject declares "
        f"{declared!r}. Reinstall with `pip install -e .` after bumping."
    )


def test_version_looks_like_pep440():
    # Loose check: digits.digits.digits with optional pre/dev/post suffix.
    assert re.match(r"^\d+\.\d+\.\d+", osr_metrics.__version__)
