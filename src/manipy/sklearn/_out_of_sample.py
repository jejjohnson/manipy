"""scikit-learn adapter for `manipy.NystromExtension`."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    clone,
)
from sklearn.utils.validation import check_is_fitted, validate_data

from manipy import _out_of_sample
from manipy.sklearn._embeddings import DiffusionMaps


__all__ = ["NystromExtension"]


class NystromExtension(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Turns a transductive spectral embedding into a transformer.

    Fits ``estimator`` and extends its embedding to new points with the
    Nyström formula (`manipy.NystromExtension`). ``estimator`` is
    `manipy.sklearn.DiffusionMaps` (the default) or kernellib's
    ``kernellib.sklearn.LaplacianEigenmaps`` /
    ``kernellib.sklearn.SchrodingerEigenmaps``; ``y`` is passed on (the
    labels of Schrödinger eigenmaps). ``fit_transform`` returns the fitted
    embedding; ``transform`` of the training points reproduces it (for
    Schrödinger eigenmaps, at points the potential leaves alone).

    Args:
        estimator: The embedding to extend; ``None`` for
            ``DiffusionMaps()``.

    Attributes:
        estimator_: The fitted clone of ``estimator``.
        extension_: The fitted `manipy.NystromExtension`.
        embedding_: The training embedding, ``(n_samples, n_components)``.
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> from manipy.sklearn import DiffusionMaps, NystromExtension
        >>> rng = np.random.default_rng(0)
        >>> X, X_new = rng.normal(size=(80, 3)), rng.normal(size=(5, 3))
        >>> ny = NystromExtension(DiffusionMaps(n_components=2, t=2)).fit(X)
        >>> ny.transform(X_new).shape
        (5, 2)
        >>> bool(np.allclose(ny.transform(X), ny.embedding_))
        True
    """

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Any = None) -> NystromExtension:
        """Fit the embedding and set up its extension.

        Args:
            X: ``(n_samples, n_features)``.
            y: Passed to ``estimator.fit``.

        Returns:
            ``self``.
        """
        X = validate_data(self, X, dtype=np.float64)
        est = DiffusionMaps() if self.estimator is None else clone(self.estimator)
        self.estimator_ = est.fit(X, y)
        self.extension_ = _out_of_sample.NystromExtension().fit(
            self.estimator_.model_, jnp.asarray(X)
        )
        self.embedding_ = np.asarray(self.estimator_.embedding_)
        self._n_features_out = self.embedding_.shape[1]
        return self

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> np.ndarray:
        """Fit and return the training embedding.

        Args:
            X: ``(n_samples, n_features)``.
            y: Passed to ``estimator.fit``.
            **fit_params: Unused.

        Returns:
            ``(n_samples, n_components)``.
        """
        return self.fit(X, y).embedding_

    def transform(self, X: Any) -> np.ndarray:
        """Embed new points by the Nyström extension.

        Args:
            X: ``(n_samples, n_features)``.

        Returns:
            ``(n_samples, n_components)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.asarray(self.extension_.transform(jnp.asarray(X)))
