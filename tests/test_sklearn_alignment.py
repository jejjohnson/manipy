"""The scikit-learn adapter for manifold alignment, tested directly.

Multi-domain input cannot satisfy ``check_estimator``, so the parts of the
contract the adapter does keep are checked one by one.
"""

from __future__ import annotations

import kernellib as kl
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

import manipy
from manipy.sklearn import ManifoldAlignment


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def domains() -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 20)
    latent = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])[y] + rng.normal(
        size=(60, 2)
    )
    X_a = latent @ rng.normal(size=(2, 5)) + 0.1 * rng.normal(size=(60, 5))
    X_b = latent @ rng.normal(size=(2, 3)) + 0.1 * rng.normal(size=(60, 3))
    partial = np.where(np.arange(60) % 3 == 0, y, -1)
    return [X_a, X_b], [partial, partial]


def test_fit_returns_self_and_sets_fitted_attributes(domains) -> None:
    X, y = domains
    est = ManifoldAlignment(n_components=2, n_neighbors=5)
    assert est.fit(X, y) is est
    assert est.n_domains_ == 2
    assert est.n_features_per_domain_ == [5, 3]
    assert [P.shape for P in est.projections_] == [(5, 2), (3, 2)]
    assert est.eigenvalues_.shape == (2,)
    assert isinstance(est.model_, manipy.ManifoldAlignment)


def test_transform_matches_core_model(domains) -> None:
    X, y = domains
    est = ManifoldAlignment(n_components=2, n_neighbors=5, standardize=True).fit(X, y)
    for i in range(2):
        out = est.transform(X[i], domain=i)
        assert isinstance(out, np.ndarray)
        assert out.shape == (60, 2)
        np.testing.assert_allclose(
            out, np.asarray(est.model_.transform(X[i], domain=i))
        )


def test_params_and_clone(domains) -> None:
    X, y = domains
    est = ManifoldAlignment("wang", 3, mu=0.2, n_neighbors=5)
    assert est.get_params()["mu"] == 0.2
    est.set_params(mu=0.7)
    copy = clone(est)
    assert copy.get_params() == est.get_params()
    assert not hasattr(copy, "model_")
    copy.fit(X, y)
    assert copy.model_.method == "wang" and copy.model_.mu == 0.7


def test_unfitted_transform_raises() -> None:
    with pytest.raises(NotFittedError):
        ManifoldAlignment().transform(np.ones((2, 2)), domain=0)


def test_fit_rejects_a_single_array(domains) -> None:
    X, y = domains
    with pytest.raises(ValueError, match="lists of per-domain"):
        ManifoldAlignment().fit(X[0], y[0])


def test_sema_through_the_adapter(domains) -> None:
    X, y = domains
    graphs = [kl.grid_graph((6, 10))] * 2
    est = ManifoldAlignment("sema", 2, n_neighbors=5).fit(X, y, graphs)
    assert est.transform(X[1], domain=1).shape == (60, 2)
