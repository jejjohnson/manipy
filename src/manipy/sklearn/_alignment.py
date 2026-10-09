"""scikit-learn adapter for `manipy.ManifoldAlignment`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import jax.numpy as jnp
import kernellib as kl
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from manipy._alignment import _linear


__all__ = ["ManifoldAlignment"]

_FLOAT = (np.float64, np.float32)


class ManifoldAlignment(BaseEstimator):
    """Manifold alignment (`manipy.ManifoldAlignment`) for scikit-learn.

    A partial fit of the scikit-learn estimator contract. Alignment needs
    several domains with different numbers of features, which a single
    ``(n_samples, n_features)`` ``X`` cannot hold, so:

    - ``fit(X, y)`` takes ``X`` as a *list* of per-domain arrays and ``y`` as
      a list of per-domain label arrays (``-1`` for unlabelled);
    - ``transform(X, domain=i)`` needs the domain of the samples;
    - there is no ``fit_transform``, and the adapter cannot sit inside a
      ``Pipeline`` or pass ``check_estimator``.

    ``clone``, ``get_params`` / ``set_params`` (so a hand-written parameter
    search) and the fitted-attribute conventions work. Put the classifier
    after the alignment yourself, as in the example.

    Args:
        method: ``"wang"``, ``"ssma"`` or ``"sema"``.
        n_components: Dimension of the shared space.
        mu: Weight of the same-label term.
        alpha: Weight of the potential (``"sema"`` only).
        normalize_potential: Trace-normalise the potential (``"sema"`` only).
        ridge: Relative ridge, see `manipy.ManifoldAlignment`.
        n_neighbors: Neighbours per sample in each domain's graph.
        weighting: ``"heat"``, ``"connectivity"`` or ``"cosine"``.
        bandwidth: Heat-kernel width, ``None`` for the median distance.
        standardize: z-score each domain's embedding on its labelled
            samples.

    Attributes:
        model_: The fitted `manipy.ManifoldAlignment`.
        projections_: Per-domain projections, ``(n_features_i, n_components)``.
        eigenvalues_: The generalised eigenvalues, ascending.
        n_domains_: Number of domains seen in ``fit``.
        n_features_per_domain_: Number of features of each domain.

    Examples:
        Two sensors (say HyMap and AVIRIS) image the same three classes
        through 6 and 4 bands. Align them, train an SVM on the first sensor's
        labelled pixels and classify the second sensor's pixels.

        >>> import numpy as np
        >>> from sklearn.svm import SVC
        >>> from manipy.sklearn import ManifoldAlignment
        >>> rng = np.random.default_rng(0)
        >>> y = np.repeat(np.arange(3), 40)
        >>> latent = np.array([[0, 0], [4, 0], [0, 4]])[y] + rng.normal(
        ...     size=(120, 2)
        ... )
        >>> X_hymap = latent @ rng.normal(size=(2, 6)) + 0.1 * rng.normal(
        ...     size=(120, 6)
        ... )
        >>> X_aviris = latent @ rng.normal(size=(2, 4)) + 0.1 * rng.normal(
        ...     size=(120, 4)
        ... )
        >>> partial = np.where(np.arange(120) % 3 == 0, y, -1)  # a third labelled
        >>> ma = ManifoldAlignment(n_components=2).fit(
        ...     [X_hymap, X_aviris], [partial, partial]
        ... )
        >>> labelled = partial >= 0
        >>> svm = SVC().fit(ma.transform(X_hymap[labelled], domain=0), y[labelled])
        >>> pred = svm.predict(ma.transform(X_aviris, domain=1))
        >>> bool(np.mean(pred == y) > 0.85)
        True
    """

    def __init__(
        self,
        method: Literal["wang", "ssma", "sema"] = "ssma",
        n_components: int = 10,
        *,
        mu: float = 0.5,
        alpha: float = 1.0,
        normalize_potential: bool = True,
        ridge: float = 1e-6,
        n_neighbors: int = 10,
        weighting: Literal["heat", "connectivity", "cosine"] = "heat",
        bandwidth: float | None = None,
        standardize: bool = False,
    ) -> None:
        self.method = method
        self.n_components = n_components
        self.mu = mu
        self.alpha = alpha
        self.normalize_potential = normalize_potential
        self.ridge = ridge
        self.n_neighbors = n_neighbors
        self.weighting = weighting
        self.bandwidth = bandwidth
        self.standardize = standardize

    def fit(
        self,
        X: Sequence[Any],
        y: Sequence[Any],
        spatial_graphs: Sequence[kl.AbstractGraph] | None = None,
    ) -> ManifoldAlignment:
        """Fit the per-domain projections.

        Args:
            X: List of per-domain arrays, ``(n_samples_i, n_features_i)``.
            y: List of per-domain integer labels, ``-1`` for unlabelled.
            spatial_graphs: One graph per domain (``"sema"`` only).

        Returns:
            ``self``.

        Raises:
            ValueError: If ``X`` or ``y`` is not a list of per-domain arrays,
                or for anything `manipy.ManifoldAlignment.fit` rejects.
        """
        if isinstance(X, np.ndarray) or isinstance(y, np.ndarray):
            raise ValueError(
                "ManifoldAlignment.fit takes lists of per-domain arrays, "
                "X=[X_0, X_1, ...] and y=[y_0, y_1, ...]."
            )
        Xs = [check_array(Xi, dtype=_FLOAT) for Xi in X]
        ys = [np.asarray(yi).astype(int) for yi in y]
        self.model_ = _linear.ManifoldAlignment(
            method=self.method,
            n_components=self.n_components,
            mu=self.mu,
            alpha=self.alpha,
            normalize_potential=self.normalize_potential,
            ridge=self.ridge,
            n_neighbors=self.n_neighbors,
            weighting=self.weighting,
            bandwidth=self.bandwidth,
            standardize=self.standardize,
        ).fit(
            [jnp.asarray(Xi) for Xi in Xs],
            [jnp.asarray(yi) for yi in ys],
            spatial_graphs,
        )
        assert self.model_.projections is not None
        assert self.model_.eigenvalues is not None
        self.projections_ = [np.asarray(P) for P in self.model_.projections]
        self.eigenvalues_ = np.asarray(self.model_.eigenvalues)
        self.n_domains_ = len(Xs)
        self.n_features_per_domain_ = [Xi.shape[1] for Xi in Xs]
        return self

    def transform(self, X: Any, *, domain: int) -> np.ndarray:
        """Map samples of one domain into the shared space.

        Args:
            X: Samples of domain ``domain``, ``(n_samples, n_features_domain)``.
            domain: Index of the domain, in the order given to ``fit``.

        Returns:
            ``(n_samples, n_components)``.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=_FLOAT)
        return np.asarray(self.model_.transform(jnp.asarray(X), domain=domain))
