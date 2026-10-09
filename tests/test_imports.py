"""Dependency boundary guards.

manipy sits on top of kernellib and gaussx and must stay usable without
scikit-learn or NumPyro. Importing the package in a fresh interpreter must not
pull in either, nor pyrox or the optional-extra libraries.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


FORBIDDEN_ON_IMPORT = (
    "numpyro",
    "pyrox",
    "sklearn",
    "pynndescent",
    "pooch",
)


@pytest.mark.slow
def test_import_does_not_load_forbidden_dependencies() -> None:
    code = (
        "import sys, manipy; "
        f"loaded = sorted(set({FORBIDDEN_ON_IMPORT!r}) & set(sys.modules)); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.slow
def test_import_does_not_load_sklearn() -> None:
    code = "import sys, manipy; assert 'sklearn' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
