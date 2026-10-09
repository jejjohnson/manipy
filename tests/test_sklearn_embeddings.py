"""scikit-learn's estimator checks on the embedding adapters."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import parametrize_with_checks

import manipy
from manipy.sklearn import Isomap


pytestmark = pytest.mark.integration


ESTIMATORS = [
    Isomap(n_components=2, n_neighbors=5),
    Isomap(n_components=2, n_neighbors=5, n_landmarks=8, random_state=0),
]


@parametrize_with_checks(ESTIMATORS)
def test_sklearn_compatible(estimator, check) -> None:
    check(estimator)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_adapter_matches_core_model(estimator) -> None:
    X = np.random.default_rng(0).normal(size=(40, 3))
    est = clone(estimator)
    Y = est.fit_transform(X)
    assert isinstance(Y, np.ndarray) and Y.shape == (40, 2)
    assert isinstance(est.model_, manipy.Isomap)
    np.testing.assert_allclose(Y, np.asarray(est.model_.embedding))
    assert est.n_features_in_ == 3
