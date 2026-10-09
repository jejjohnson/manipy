"""scikit-learn adapters for manipy's embeddings."""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import kernellib as kl
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data

from manipy import _embeddings as emb


__all__ = ["DiffusionMaps", "Isomap", "LocallyLinearEmbedding"]

_FLOAT = (np.float64, np.float32)
# Diffusion maps solve inputs this small densely, whatever eigen_solver says.
_SMALL = 200


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


def _hessian_size(n: int) -> int:
    """Hessian LLE needs more than ``n (n + 3) / 2`` neighbours."""
    return n * (n + 3) // 2


class LocallyLinearEmbedding(_Embedding):
    """LLE, modified LLE, Hessian LLE and LTSA (`manipy.LocallyLinearEmbedding`)
    for scikit-learn.

    Like ``sklearn.manifold.LocallyLinearEmbedding`` without ``transform``:
    ``fit`` / ``fit_transform`` only. On small inputs ``n_neighbors`` is
    capped at ``n_samples - 1`` and ``n_components`` at
    ``min(n_features, n_samples - 2)``, and for ``"hessian"`` further until
    ``n_neighbors > n_components (n_components + 3) / 2``.

    Args:
        n_components: Embedding dimension.
        n_neighbors: Neighbours per point.
        method: ``"standard"``, ``"modified"``, ``"hessian"`` or ``"ltsa"``.
        reg: Relative regularisation of the local Gram matrices.
        modified_tol: Householder tolerance of ``"modified"``.
        eigen_solver: ``"dense"`` or ``"arpack"``.
        neighbors_backend: ``"exact"``, ``"pynndescent"`` or ``"sklearn"``.
        random_state: Seed for approximate neighbours and ARPACK.

    Attributes:
        embedding_: ``(n_samples, n_components)``.
        eigenvalues_: The eigenvalues of the alignment matrix used.
        reconstruction_error_: Their sum.
        model_: The fitted `manipy.LocallyLinearEmbedding`.
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> from manipy.sklearn import LocallyLinearEmbedding
        >>> X = np.random.default_rng(0).normal(size=(60, 3))
        >>> LocallyLinearEmbedding(n_components=2, n_neighbors=8).fit_transform(
        ...     X
        ... ).shape
        (60, 2)
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        n_neighbors: int = 10,
        method: Literal["standard", "modified", "hessian", "ltsa"] = "standard",
        reg: float = 1e-3,
        modified_tol: float = 1e-12,
        eigen_solver: Literal["dense", "arpack"] = "dense",
        neighbors_backend: Literal["exact", "pynndescent", "sklearn"] = "exact",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.method = method
        self.reg = reg
        self.modified_tol = modified_tol
        self.eigen_solver = eigen_solver
        self.neighbors_backend = neighbors_backend
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> LocallyLinearEmbedding:
        """Embed ``X``.

        Args:
            X: ``(n_samples, n_features)``.
            y: Ignored.

        Returns:
            ``self``.
        """
        X = validate_data(self, X, dtype=_FLOAT)
        n, d = X.shape
        _check_n_samples(n, 3, type(self).__name__)
        n_neighbors = min(self.n_neighbors, n - 1)
        n_components = min(self.n_components, d, n - 2)
        if self.method == "hessian":
            while n_components > 1 and n_neighbors <= _hessian_size(n_components):
                n_components -= 1
            if n_neighbors <= _hessian_size(n_components):
                raise ValueError(
                    'method="hessian" needs n_neighbors > 2 even for one '
                    f"component; got n_samples = {n}."
                )
        self.model_ = emb.LocallyLinearEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors,
            method=self.method,
            reg=self.reg,
            modified_tol=self.modified_tol,
            eigen_solver=self.eigen_solver,
            neighbors_backend=self.neighbors_backend,
            random_state=self.random_state,
        ).fit(jnp.asarray(X))
        self.embedding_ = np.asarray(self.model_.embedding)
        self.eigenvalues_ = np.asarray(self.model_.eigenvalues)
        self.reconstruction_error_ = float(np.asarray(self.model_.reconstruction_error))
        return self


class DiffusionMaps(_Embedding):
    """Diffusion maps (`manipy.DiffusionMaps`) for scikit-learn.

    ``fit`` / ``fit_transform`` only. On small inputs ``n_neighbors`` and
    ``n_components`` are capped at ``n_samples - 1``.

    Args:
        n_components: Embedding dimension.
        alpha: Anisotropy in ``[0, 1]`` (``1``: Laplace-Beltrami).
        t: Diffusion time.
        kernel: A kernellib kernel, or ``None`` for the heat kernel.
        n_neighbors: ``None`` for the dense Gram matrix, else the k-NN graph.
        bandwidth: Heat-kernel width (``kernel=None``); ``None`` for the
            median heuristic.
        eigen_solver: ``"dense"`` or ``"lanczos"``. Like
            ``sklearn.manifold.SpectralEmbedding``'s iterative solvers,
            ``"lanczos"`` falls back to ``"dense"`` for ``n_samples <= 200``,
            where its Krylov rank (at most ``n_samples / 2``) is too small.
        neighbors_backend: ``"exact"``, ``"pynndescent"`` or ``"sklearn"``.
        random_state: Seed for approximate neighbours and Lanczos.

    Attributes:
        embedding_: ``(n_samples, n_components)``.
        eigenvalues_: Eigenvalues of the Markov matrix, descending.
        model_: The fitted `manipy.DiffusionMaps`.
        n_features_in_: Number of input features.

    Examples:
        >>> import numpy as np
        >>> from manipy.sklearn import DiffusionMaps
        >>> X = np.random.default_rng(0).normal(size=(60, 3))
        >>> DiffusionMaps(n_components=2, alpha=1.0, t=2).fit_transform(X).shape
        (60, 2)
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        alpha: float = 1.0,
        t: float = 1.0,
        kernel: kl.AbstractKernel | None = None,
        n_neighbors: int | None = None,
        bandwidth: float | None = None,
        eigen_solver: Literal["dense", "lanczos"] = "dense",
        neighbors_backend: Literal["exact", "pynndescent", "sklearn"] = "exact",
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.alpha = alpha
        self.t = t
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.eigen_solver = eigen_solver
        self.neighbors_backend = neighbors_backend
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> DiffusionMaps:
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
        self.model_ = emb.DiffusionMaps(
            n_components=min(self.n_components, n - 1),
            alpha=self.alpha,
            t=self.t,
            kernel=self.kernel,
            n_neighbors=None
            if self.n_neighbors is None
            else min(self.n_neighbors, n - 1),
            bandwidth=self.bandwidth,
            eigen_solver="dense" if n <= _SMALL else self.eigen_solver,
            neighbors_backend=self.neighbors_backend,
            random_state=self.random_state,
        ).fit(jnp.asarray(X))
        self.embedding_ = np.asarray(self.model_.embedding)
        self.eigenvalues_ = np.asarray(self.model_.eigenvalues)
        return self
