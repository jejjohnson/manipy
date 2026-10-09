"""scikit-learn adapters for manipy's embeddings."""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

from manipy import _embeddings as emb


__all__ = ["Isomap"]

_FLOAT = (np.float64, np.float32)


def _check_n_samples(n: int, minimum: int, name: str) -> None:
    if n < minimum:
        raise ValueError(
            f"{name} needs at least {minimum} samples; got n_samples = {n}."
        )


class _Embedding(BaseEstimator):
    """Transductive embedding: ``fit`` / ``fit_transform``, no ``transform``,
    like ``sklearn.manifold.SpectralEmbedding``."""

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit and return the embedding of ``X``.

        Args:
            X: ``(n_samples, n_features)``.
            y: Ignored.

        Returns:
            ``(n_samples, n_components)``.
        """
        return self.fit(X, y).embedding_  # ty: ignore[unresolved-attribute]


class Isomap(_Embedding):
    """Isomap and landmark Isomap (`manipy.Isomap`) for scikit-learn.

    Like ``sklearn.manifold.Isomap`` without ``transform``: ``fit`` /
    ``fit_transform`` only. On small inputs ``n_neighbors`` is capped at
    ``n_samples - 1`` and ``n_landmarks`` and ``n_components`` at
    ``n_samples``.

    Args:
        n_components: Embedding dimension.
        n_neighbors: Neighbours per point in the k-NN graph.
        n_landmarks: Number of landmarks, or ``None`` for exact Isomap.
        neighbors_backend: ``"exact"``, ``"pynndescent"`` or ``"sklearn"``.
        random_state: Seed for the landmarks and approximate neighbours.

    Attributes:
        embedding_: ``(n_samples, n_components)``.
        eigenvalues_: The largest eigenvalues of the double-centred squared
            geodesic distances, descending.
        landmarks_: Landmark indices, or ``None`` for exact Isomap.
        model_: The fitted `manipy.Isomap`.
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> from manipy.sklearn import Isomap
        >>> X = np.random.default_rng(0).normal(size=(60, 3))
        >>> Isomap(n_components=2, n_neighbors=8).fit_transform(X).shape
        (60, 2)
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 10,
        n_landmarks: int | None = None,
        neighbors_backend: Literal["exact", "pynndescent", "sklearn"] = "exact",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_landmarks = n_landmarks
        self.neighbors_backend = neighbors_backend
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> Isomap:
        """Embed ``X``.

        Args:
            X: ``(n_samples, n_features)``.
            y: Ignored.

        Returns:
            ``self``.
        """
        X = validate_data(self, X, dtype=_FLOAT)
        n = X.shape[0]
        _check_n_samples(n, 2, type(self).__name__)
        n_landmarks = None if self.n_landmarks is None else min(self.n_landmarks, n)
        n_components = min(self.n_components, n if n_landmarks is None else n_landmarks)
        self.model_ = emb.Isomap(
            n_components=n_components,
            n_neighbors=min(self.n_neighbors, n - 1),
            n_landmarks=n_landmarks,
            neighbors_backend=self.neighbors_backend,
            random_state=self.random_state,
        ).fit(jnp.asarray(X))
        self.embedding_ = np.asarray(self.model_.embedding)
        self.eigenvalues_ = np.asarray(self.model_.eigenvalues)
        self.landmarks_ = (
            None if self.model_.landmarks is None else np.asarray(self.model_.landmarks)
        )
        return self
