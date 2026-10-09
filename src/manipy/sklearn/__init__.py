"""scikit-learn adapters for manipy (optional: ``pip install manipy-jax[sklearn]``).

The core library never imports scikit-learn; this module is imported only
when asked for. Each adapter wraps a manipy object in scikit-learn's
estimator conventions (mutable, ``fit`` returns ``self``, fitted attributes
end in ``_``, NumPy in and out), so ``clone``, ``get_params`` and
``set_params`` work. The fitted manipy object is kept as ``model_``.

- `DiffusionMaps`: diffusion maps (``fit`` / ``fit_transform``).
- `Isomap`: Isomap and landmark Isomap (``fit`` / ``fit_transform``).
- `LocallyLinearEmbedding`: LLE, modified LLE, Hessian LLE and LTSA
  (``fit`` / ``fit_transform``).
- `NystromExtension`: wraps diffusion maps or kernellib's Laplacian /
  Schrödinger eigenmaps into a transformer with a Nyström ``transform``.
- `ManifoldAlignment`: multi-domain alignment. Its ``fit`` takes *lists* of
  per-domain arrays and its ``transform`` needs ``domain=``, so it is a
  partial fit of the scikit-learn contract (see its docstring).
"""

from manipy.sklearn._alignment import ManifoldAlignment
from manipy.sklearn._embeddings import DiffusionMaps, Isomap, LocallyLinearEmbedding
from manipy.sklearn._out_of_sample import NystromExtension


__all__ = [
    "DiffusionMaps",
    "Isomap",
    "LocallyLinearEmbedding",
    "ManifoldAlignment",
    "NystromExtension",
]
