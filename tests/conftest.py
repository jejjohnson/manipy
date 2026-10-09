"""Shared pytest configuration.

float64 is enabled for the whole suite, as in kernellib and gaussx: tests
compare against dense references at float64 tolerances.
"""

from __future__ import annotations

import equinox.internal as eqxi
import jax
import pytest


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def getkey() -> eqxi.GetKey:
    """Fresh PRNG keys, seeded from ``EQX_GETKEY_SEED`` when set."""
    return eqxi.GetKey()


# Doctests cannot take decorators, so they are tiered here. The scikit-learn
# adapters' examples are integration tests; add the qualified names of any
# doctest that measures over a second (mostly jit compilation) to the set.
_SLOW_DOCTESTS: frozenset[str] = frozenset(
    {
        # ~10 s cold: graphs, fit, compile
        "manipy._alignment._linear.ManifoldAlignment",
        "manipy._embeddings._isomap.Isomap",  # ~5 s: two swiss-roll fits
        "manipy._embeddings._lle.LocallyLinearEmbedding",  # ~6 s: two fits
        "manipy._embeddings._diffusion_maps.DiffusionMaps",  # ~4 s: graph + fit
        "manipy._out_of_sample.NystromExtension",  # ~10 s: two fits + extensions
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if not isinstance(item, pytest.DoctestItem):
            continue
        if item.name.startswith("manipy.sklearn."):
            item.add_marker(pytest.mark.integration)
        elif item.name in _SLOW_DOCTESTS:
            item.add_marker(pytest.mark.slow)
