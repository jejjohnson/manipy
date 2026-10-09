"""scikit-learn adapters for manipy (optional: ``pip install manipy-jax[sklearn]``).

The core library never imports scikit-learn; this module is imported only
when asked for. Each adapter wraps a manipy object in scikit-learn's
estimator conventions (mutable, ``fit`` returns ``self``, fitted attributes
end in ``_``, NumPy in and out), so ``clone``, ``get_params`` and
``set_params`` work. The fitted manipy object is kept as ``model_``.

- `ManifoldAlignment`: multi-domain alignment. Its ``fit`` takes *lists* of
  per-domain arrays and its ``transform`` needs ``domain=``, so it is a
  partial fit of the scikit-learn contract (see its docstring).
"""

from manipy.sklearn._alignment import ManifoldAlignment


__all__ = ["ManifoldAlignment"]
