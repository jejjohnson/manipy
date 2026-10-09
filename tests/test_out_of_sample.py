"""Nyström out-of-sample extension: in-sample points reproduce their embedding."""

from __future__ import annotations

import dataclasses

import einx
import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import pytest

import manipy


@pytest.fixture(scope="module")
def roll() -> jax.Array:
    X, _ = manipy.datasets.swiss_roll(240, key=jax.random.key(0))
    return X


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"weighting": "connectivity"}, {"bandwidth": 2.5}],
    ids=["heat-median", "connectivity", "heat-fixed"],
)
def test_laplacian_eigenmaps_in_sample(roll, kwargs) -> None:
    le = kl.LaplacianEigenmaps(n_components=3, n_neighbors=8, **kwargs).fit(roll)
    ext = manipy.NystromExtension().fit(le, roll)
    # The affinities of the training points are the training graph, exactly.
    W = kl.adjacency_matrix(le.graph, **kwargs)
    np.testing.assert_allclose(ext.affinities(roll), W, atol=1e-12)
    np.testing.assert_allclose(ext.transform(roll), le.embedding, atol=1e-10)


def test_schrodinger_eigenmaps_unlabelled_points_in_sample(roll) -> None:
    labels = jnp.full(roll.shape[0], -1).at[:10].set(0).at[10:20].set(1)
    se = kl.SchrodingerEigenmaps(n_components=2, n_neighbors=8, alpha=5.0)
    se = se.fit(roll, kl.label_potential(labels))
    Y = manipy.NystromExtension().fit(se, roll).transform(roll)
    # New points carry no potential: exact where the potential row is zero.
    np.testing.assert_allclose(Y[20:], se.embedding[20:], atol=1e-10)


@pytest.mark.parametrize("n_neighbors", [None, 8])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_diffusion_maps_in_sample(roll, n_neighbors, alpha) -> None:
    dm = manipy.DiffusionMaps(
        n_components=3, alpha=alpha, t=3, n_neighbors=n_neighbors
    ).fit(roll)
    ext = manipy.NystromExtension().fit(dm, roll)
    np.testing.assert_allclose(ext.transform(roll), dm.embedding, atol=1e-10)


def test_new_points_extend_smoothly(roll) -> None:
    # With a dense kernel the extension is smooth: a small perturbation of a
    # training point lands near its embedding. (On a k-NN graph it is only
    # piecewise smooth: a perturbation can change the neighbour set.) A query
    # equal to a training point is that point.
    dm = manipy.DiffusionMaps(n_components=2, alpha=1.0, bandwidth=3.0).fit(roll)
    ext = manipy.NystromExtension().fit(dm, roll)
    step = 1e-4 * jax.random.normal(jax.random.key(1), roll.shape)
    near = ext.transform(roll + step)
    spread = jnp.max(jnp.abs(dm.embedding))
    assert float(jnp.max(jnp.abs(near - dm.embedding))) < 1e-2 * float(spread)
    np.testing.assert_allclose(ext.transform(roll[:3]), dm.embedding[:3], atol=1e-10)


@pytest.mark.slow
def test_held_out_points_keep_their_neighbourhoods() -> None:
    X, _ = manipy.datasets.swiss_roll(1000, key=jax.random.key(2))
    train, test = X[:700], X[700:]
    le = kl.LaplacianEigenmaps(n_components=2, n_neighbors=12).fit(train)
    Y_test = manipy.NystromExtension().fit(le, train).transform(test)
    t_train = float(manipy.metrics.trustworthiness(train, le.embedding, n_neighbors=10))
    t_test = float(manipy.metrics.trustworthiness(test, Y_test, n_neighbors=10))
    assert t_test > t_train - 0.05


def test_affinities_follow_the_symmetrised_knn_rule(roll) -> None:
    le = kl.LaplacianEigenmaps(n_neighbors=5, weighting="connectivity").fit(roll)
    ext = manipy.NystromExtension().fit(le, roll)
    x = einx.id("d -> 1 d", roll[0] + 0.3)
    A = np.asarray(ext.affinities(x))[0]
    d = np.asarray(jnp.sqrt(einx.sum("n [d]", (roll - x) ** 2)))
    own = set(np.argsort(d)[:5].tolist())  # its 5 nearest training points
    listed_by = set(np.flatnonzero(d <= np.asarray(ext.radii)).tolist())
    assert set(np.flatnonzero(A).tolist()) == own | listed_by
    assert set(np.unique(A).tolist()) <= {0.0, 1.0}


def test_errors(roll) -> None:
    le = kl.LaplacianEigenmaps(n_neighbors=8)
    with pytest.raises(ValueError, match="not fitted"):
        manipy.NystromExtension().fit(le, roll)
    with pytest.raises(ValueError, match="not fitted"):
        manipy.NystromExtension().fit(manipy.DiffusionMaps(), roll)
    with pytest.raises(TypeError, match="extends"):
        manipy.NystromExtension().fit(manipy.Isomap(), roll)  # ty: ignore[invalid-argument-type]
    fitted = le.fit(roll)
    with pytest.raises(ValueError, match="training points"):
        manipy.NystromExtension().fit(fitted, roll[:10])
    W = kl.adjacency_matrix(kl.nearest_neighbors(roll, 8))
    on_graph = kl.LaplacianEigenmaps(n_neighbors=8).fit(roll, graph=W)
    with pytest.raises(ValueError, match="precomputed"):
        manipy.NystromExtension().fit(on_graph, roll)
    identity = kl.LaplacianEigenmaps(n_neighbors=8, constraint="identity").fit(roll)
    with pytest.raises(ValueError, match="degree"):
        manipy.NystromExtension().fit(identity, roll)
    with pytest.raises(ValueError, match="not fitted"):
        manipy.NystromExtension().transform(roll)
    ext = manipy.NystromExtension().fit(fitted, roll)
    with pytest.raises(ValueError, match="features"):
        ext.transform(jnp.ones((2, 2)))


def test_zero_eigenvalue_is_rejected(roll) -> None:
    le = kl.LaplacianEigenmaps(n_neighbors=8).fit(roll)
    broken = dataclasses.replace(le, eigenvalues=jnp.ones_like(le.eigenvalues))
    with pytest.raises(ValueError, match="eigenvalue"):
        manipy.NystromExtension().fit(broken, roll)
