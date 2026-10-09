"""Guards on the public API surface.

These tests are deliberately about *contracts*, not behaviour: if a symbol is
removed from ``__all__`` or stops being importable from the top level, that is
a breaking change and should fail loudly here.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

import manipy


# Every non-package module directly under ``manipy``. A new module must be
# added here (and given an API doc page) for ``test_no_unexpected_submodules``
# to pass.
SUBMODULES: list[str] = ["_out_of_sample", "datasets", "hsi", "metrics"]


def _isort_order(names: list[str]) -> list[str]:
    """The order ruff's RUF022 fixer gives ``__all__``: CONSTANTS, then
    CamelCase, then everything else, each alphabetical. Matching it keeps the
    test and ``ruff check --fix`` from fighting over names like ``RBF``."""

    def key(name: str) -> tuple[int, str]:
        if name.isupper():
            return (0, name)
        if name[:1].isupper():
            return (1, name)
        return (2, name)

    return sorted(names, key=key)


def test_version_is_a_dotted_string() -> None:
    assert isinstance(manipy.__version__, str)
    major, minor, patch = manipy.__version__.split(".")[:3]
    assert all(part.isdigit() for part in (major, minor, patch))


def test_all_is_sorted_and_unique() -> None:
    assert manipy.__all__ == _isort_order(manipy.__all__)
    assert len(manipy.__all__) == len(set(manipy.__all__))


@pytest.mark.parametrize("name", manipy.__all__)
def test_every_exported_name_is_importable(name: str) -> None:
    assert hasattr(manipy, name), f"{name} is in __all__ but not defined"


def test_no_unexpected_submodules() -> None:
    found = {
        info.name for info in pkgutil.iter_modules(manipy.__path__) if not info.ispkg
    }
    assert found == set(SUBMODULES)


def test_submodules_import_cleanly_and_declare_all() -> None:
    for module in SUBMODULES:
        imported = importlib.import_module(f"manipy.{module}")
        assert hasattr(imported, "__all__"), f"manipy.{module} is missing __all__"
        assert imported.__all__ == _isort_order(imported.__all__)


def test_package_is_typed() -> None:
    """PEP 561 marker must ship so downstream type checkers see annotations."""
    marker = Path(manipy.__path__[0]) / "py.typed"
    assert marker.is_file(), f"missing PEP 561 marker at {marker}"
