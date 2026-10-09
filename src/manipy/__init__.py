"""manipy: manifold alignment, dimensionality reduction and HSI workflows in JAX.

Built on kernellib (graphs, Laplacians, eigenmaps, kernel PCA) and gaussx
(linear-algebra primitives). manipy uses these and never re-exports them:
import ``kernellib.LaplacianEigenmaps`` directly.

The public API is flat and ``__all__`` is the contract. It is empty at the
M0 scaffold; the algorithms land in later phases of the roadmap.
"""

from __future__ import annotations


__version__ = "0.0.0"  # x-release-please-version

__all__: list[str] = []
