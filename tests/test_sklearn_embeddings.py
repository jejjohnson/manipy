"""scikit-learn's estimator checks on the embedding adapters."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import parametrize_with_checks

import manipy
from manipy.sklearn import DiffusionMaps, Isomap, LocallyLinearEmbedding


pytestmark = pytest.mark.integration


ESTIMATORS = [
    Isomap(n_components=2, n_neighbors=5),
    Isomap(n_components=2, n_neighbors=5, n_landmarks=8, random_state=0),
    LocallyLinearEmbedding(n_components=2, n_neighbors=5),
    LocallyLinearEmbedding(n_components=2, n_neighbors=5, method="modified"),
    LocallyLinearEmbedding(n_components=2, n_neighbors=5, eigen_solver="arpack"),
    LocallyLinearEmbedding(n_components=2, n_neighbors=6, method="hessian"),
    LocallyLinearEmbedding(n_components=2, n_neighbors=5, method="ltsa"),
    DiffusionMaps(n_components=2, alpha=1.0, t=2),
    DiffusionMaps(n_components=2, alpha=0.5, n_neighbors=5),
    DiffusionMaps(n_components=2, alpha=0.0, eigen_solver="lanczos", random_state=0),
]

_MODELS = {
    "DiffusionMaps": manipy.DiffusionMaps,
    "Isomap": manipy.Isomap,
    "LocallyLinearEmbedding": manipy.LocallyLinearEmbedding,
}


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible(estimator, check) -> None:
    check(estimator)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_adapter_matches_core_model(estimator) -> None:
    X = np.random.default_rng(0).normal(size=(40, 3))
    est = clone(estimator)
    Y = est.fit_transform(X)
    assert isinstance(Y, np.ndarray) and Y.shape == (40, 2)
    assert isinstance(est.model_, _MODELS[type(est).__name__])
    np.testing.assert_allclose(Y, np.asarray(est.model_.embedding))
    assert est.n_features_in_ == 3


def test_hessian_rejects_too_few_samples() -> None:
    X = np.random.default_rng(0).normal(size=(3, 2))
    with pytest.raises(ValueError, match="hessian"):
        LocallyLinearEmbedding(method="hessian").fit(X)


def test_diffusion_maps_lanczos_above_the_dense_cutoff() -> None:
    X = np.random.default_rng(0).normal(size=(300, 3))
    dense = DiffusionMaps(n_neighbors=10).fit_transform(X)
    lanczos = DiffusionMaps(n_neighbors=10, eigen_solver="lanczos").fit(X)
    assert lanczos.model_.eigen_solver == "lanczos"
    np.testing.assert_allclose(np.abs(lanczos.embedding_), np.abs(dense), atol=1e-6)
