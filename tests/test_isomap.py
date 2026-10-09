"""Isomap and landmark Isomap, against scikit-learn and against each other."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.manifold import Isomap as SkIsomap

import manipy
from manipy._embeddings._isomap import _double_centre, _geodesic_distances


@pytest.fixture(scope="module")
def roll() -> jax.Array:
    X, _ = manipy.datasets.swiss_roll(120, key=jax.random.key(0))
    return X


def test_matches_sklearn_isomap(roll) -> None:
    iso = manipy.Isomap(n_components=2, n_neighbors=8).fit(roll)
    sk = SkIsomap(n_neighbors=8, n_components=2).fit(np.asarray(roll))
    np.testing.assert_allclose(iso.eigenvalues, sk.kernel_pca_.eigenvalues_, rtol=1e-8)
    # Eigenvectors are defined up to sign.
    np.testing.assert_allclose(
        np.abs(iso.embedding), np.abs(sk.embedding_), rtol=1e-6, atol=1e-6
    )


def test_geodesics_match_sklearn(roll) -> None:
    D = _geodesic_distances(roll, 8, None, "exact", None)
    sk = SkIsomap(n_neighbors=8).fit(np.asarray(roll))
    np.testing.assert_allclose(D, sk.dist_matrix_, rtol=1e-10)


def test_double_centre_is_classical_mds_gram() -> None:
    # For Euclidean distances, -1/2 J D^2 J is the Gram matrix of centred points.
    X = jax.random.normal(jax.random.key(1), (10, 3))
    Xc = einx.subtract("n d, d -> n d", X, einx.mean("[n] d -> d", X))
    D2 = einx.sum("i j [d]", einx.subtract("i d, j d -> i j d", X, X) ** 2)
    np.testing.assert_allclose(
        _double_centre(D2), einx.dot("i d, j d -> i j", Xc, Xc), atol=1e-10
    )


def test_every_point_a_landmark_is_isomap(roll) -> None:
    exact = manipy.Isomap(n_components=2, n_neighbors=8).fit(roll)
    lm = manipy.Isomap(n_components=2, n_neighbors=8, n_landmarks=120).fit(roll)
    np.testing.assert_allclose(lm.embedding, exact.embedding, atol=1e-8)
    np.testing.assert_allclose(lm.eigenvalues, exact.eigenvalues, rtol=1e-10)
    np.testing.assert_array_equal(lm.landmarks, np.arange(120))


def test_landmarks_land_on_their_mds_coordinates(roll) -> None:
    lm = manipy.Isomap(
        n_components=2, n_neighbors=8, n_landmarks=30, random_state=3
    ).fit(roll)
    idx = np.asarray(lm.landmarks)
    assert idx.shape == (30,) and len(set(idx.tolist())) == 30
    D = jnp.asarray(_geodesic_distances(roll, 8, None, "exact", None))[idx][:, idx]
    lam, U = jnp.linalg.eigh(_double_centre(D**2))
    mds = einx.multiply("m n, n -> m n", U[:, ::-1][:, :2], jnp.sqrt(lam[::-1][:2]))
    np.testing.assert_allclose(jnp.abs(lm.embedding[idx]), jnp.abs(mds), atol=1e-8)


@pytest.mark.slow
def test_disconnected_and_duplicate_points_stay_finite() -> None:
    a = jax.random.normal(jax.random.key(2), (15, 2))
    X = jnp.concatenate([a, a + 50.0, a[:3]])  # two far clusters + duplicates
    iso = manipy.Isomap(n_components=2, n_neighbors=3).fit(X)
    assert bool(jnp.all(jnp.isfinite(iso.embedding)))


def test_unfitted_fields_are_none() -> None:
    iso = manipy.Isomap()
    assert iso.embedding is None and iso.eigenvalues is None and iso.landmarks is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_components": 0}, "n_components"),
        ({"n_neighbors": 0}, "n_neighbors"),
        ({"n_components": 3, "n_landmarks": 2}, "n_landmarks"),
    ],
)
def test_invalid_config(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        manipy.Isomap(**kwargs)


def test_invalid_inputs() -> None:
    X = jnp.ones((5, 2))
    with pytest.raises(ValueError, match="2-D"):
        manipy.Isomap().fit(jnp.ones(5))
    with pytest.raises(ValueError, match="below N"):
        manipy.Isomap(n_neighbors=5).fit(X)
    with pytest.raises(ValueError, match="n_landmarks <= N"):
        manipy.Isomap(n_neighbors=2, n_landmarks=6).fit(X)


@pytest.mark.slow
def test_swiss_roll_quality_matches_sklearn() -> None:
    X, _ = manipy.datasets.swiss_roll(600, key=jax.random.key(4))
    Xn = np.asarray(X)
    sk = SkIsomap(n_neighbors=10, n_components=2).fit_transform(Xn)
    iso = manipy.Isomap(n_components=2, n_neighbors=10).fit(X).embedding
    lm = manipy.Isomap(n_components=2, n_neighbors=10, n_landmarks=80).fit(X).embedding
    t_sk = float(manipy.metrics.trustworthiness(X, sk, n_neighbors=10))
    t_iso = float(manipy.metrics.trustworthiness(X, iso, n_neighbors=10))
    t_lm = float(manipy.metrics.trustworthiness(X, lm, n_neighbors=10))
    assert abs(t_iso - t_sk) < 1e-6
    assert t_lm > t_sk - 0.02
