"""manipy: manifold alignment, dimensionality reduction and HSI workflows in JAX.

Built on kernellib (graphs, Laplacians, eigenmaps, kernel PCA) and gaussx
(linear-algebra primitives). manipy uses these and never re-exports them:
import ``kernellib.LaplacianEigenmaps`` directly.

The public API is flat and ``__all__`` is the contract:

- `ManifoldAlignment`: Wang, SSMA and SEMA manifold alignment of several
  domains with partial labels.
- `Isomap`: Isomap and landmark Isomap.
- `DiffusionMaps`: diffusion maps with anisotropic normalisation.
- `NystromExtension`: Nyström out-of-sample extension of kernellib's
  Laplacian / Schrödinger eigenmaps and of diffusion maps.
- `LocallyLinearEmbedding`: LLE, modified LLE, Hessian LLE and LTSA.

The scikit-learn adapters live in ``manipy.sklearn`` (extra
``manipy-jax[sklearn]``).
"""

from __future__ import annotations

from manipy import datasets, hsi, metrics
from manipy._alignment import ManifoldAlignment
from manipy._embeddings import DiffusionMaps, Isomap, LocallyLinearEmbedding
from manipy._out_of_sample import NystromExtension


__version__ = "0.0.0"  # x-release-please-version

__all__ = [
    "DiffusionMaps",
    "Isomap",
    "LocallyLinearEmbedding",
    "ManifoldAlignment",
    "NystromExtension",
    "datasets",
    "hsi",
    "metrics",
]
