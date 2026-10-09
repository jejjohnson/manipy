"""scikit-learn's estimator checks on the Nyström extension adapter."""

from __future__ import annotations

import kernellib.sklearn as kls
import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import manipy
from manipy.sklearn import DiffusionMaps, NystromExtension


pytestmark = pytest.mark.integration


ESTIMATORS = [
    NystromExtension(),
    NystromExtension(DiffusionMaps(n_components=2, alpha=0.5, n_neighbors=5)),
    NystromExtension(kls.LaplacianEigenmaps(n_components=2, n_neighbors=5)),
]


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible(estimator, check) -> None:
    check(estimator)


@pytest.mark.parametrize(
    "inner",
    [
        DiffusionMaps(n_components=2, t=2),
        kls.LaplacianEigenmaps(n_components=2, n_neighbors=6),
    ],
)
def test_training_points_come_back(inner) -> None:
    X = np.random.default_rng(0).normal(size=(50, 3))
    ny = NystromExtension(inner).fit(X)
    assert isinstance(ny.extension_, manipy.NystromExtension)
    np.testing.assert_allclose(ny.transform(X), ny.embedding_, atol=1e-10)
    np.testing.assert_array_equal(ny.fit_transform(X), ny.embedding_)
    assert ny.get_feature_names_out().tolist() == [
        "nystromextension0",
        "nystromextension1",
    ]


def test_labels_reach_schrodinger_eigenmaps() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = np.full(60, -1)
    y[:5], y[5:10] = 0, 1
    ny = NystromExtension(kls.SchrodingerEigenmaps(n_neighbors=6, alpha=5.0))
    ny.fit(X, y)
    # Exact at the unlabelled points, where the potential row is zero.
    np.testing.assert_allclose(ny.transform(X[10:]), ny.embedding_[10:], atol=1e-10)
