"""LLE and modified LLE, against scikit-learn and against their definitions."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import pytest
import scipy.linalg as sla
from sklearn.manifold import LocallyLinearEmbedding as SkLLE

import manipy
from manipy._embeddings import _lle, _spectral


@pytest.fixture(scope="module")
def roll() -> jax.Array:
    X, _ = manipy.datasets.swiss_roll(150, key=jax.random.key(0))
    return X


@pytest.mark.parametrize("method", ["standard", "modified"])
def test_matches_sklearn_subspace(roll, method) -> None:
    ours = manipy.LocallyLinearEmbedding(
        n_components=2, n_neighbors=10, method=method
    ).fit(roll)
    sk = SkLLE(n_components=2, n_neighbors=10, method=method, eigen_solver="dense").fit(
        np.asarray(roll)
    )
    # Same subspace (principal angles ~ 0), same reconstruction error.
    angles = sla.subspace_angles(np.asarray(ours.embedding), sk.embedding_)
    assert np.max(angles) < 1e-6
    np.testing.assert_allclose(
        ours.reconstruction_error, sk.reconstruction_error_, rtol=1e-6
    )


def test_weights_reconstruct_and_sum_to_one(roll) -> None:
    idx = kl.nearest_neighbors(roll, 6).indices
    Z = _lle._local_offsets(roll, idx)
    W = jax.vmap(_lle.barycenter_weights, in_axes=(0, None))(Z, 1e-3)
    np.testing.assert_allclose(einx.sum("N [k]", W), 1.0, rtol=1e-12)
    # Against the direct constrained least-squares solution (unregularised,
    # where k > D makes it well posed up to the regulariser's effect).
    i = 7
    G = einx.dot("a d, b d -> a b", Z[i], Z[i])
    G = G + 1e-3 * jnp.trace(G) * jnp.eye(6)
    w = jnp.linalg.solve(G, jnp.ones(6))
    np.testing.assert_allclose(W[i], w / jnp.sum(w), rtol=1e-10)


def test_standard_operator_is_i_minus_w_squared(roll) -> None:
    N, k = 40, 5
    X = roll[:N]
    idx = kl.nearest_neighbors(X, k).indices
    blocks = _lle._standard_blocks(X, idx, 1e-3)
    nodes = jnp.concatenate([einx.id("N -> N 1", jnp.arange(N)), idx], axis=1)
    M = _spectral.alignment_operator(nodes, blocks, N).as_matrix()
    W = jax.vmap(_lle.barycenter_weights, in_axes=(0, None))(
        _lle._local_offsets(X, idx), 1e-3
    )
    Wd = jnp.zeros((N, N)).at[einx.id("N -> N k", jnp.arange(N), k=k), idx].set(W)
    I_W = jnp.eye(N) - Wd
    np.testing.assert_allclose(M, einx.dot("i a, i b -> a b", I_W, I_W), atol=1e-12)


@pytest.mark.parametrize("method", ["standard", "modified"])
def test_arpack_matches_dense(roll, method) -> None:
    dense = manipy.LocallyLinearEmbedding(n_neighbors=10, method=method).fit(roll)
    arpack = manipy.LocallyLinearEmbedding(
        n_neighbors=10, method=method, eigen_solver="arpack"
    ).fit(roll)
    np.testing.assert_allclose(arpack.eigenvalues, dense.eigenvalues, atol=1e-9)
    angles = sla.subspace_angles(np.asarray(arpack.embedding), dense.embedding)
    assert np.max(angles) < 1e-6


def test_embedding_is_orthonormal_and_centred(roll) -> None:
    Y = manipy.LocallyLinearEmbedding(n_neighbors=10).fit(roll).embedding
    np.testing.assert_allclose(
        einx.dot("N a, N b -> a b", Y, Y), jnp.eye(2), atol=1e-10
    )
    # Orthogonal to the constant null vector.
    np.testing.assert_allclose(einx.sum("[N] n", Y), 0.0, atol=1e-6)


def test_residual_check_raises(roll) -> None:
    idx = kl.nearest_neighbors(roll[:30], 4).indices
    blocks = _lle._standard_blocks(roll[:30], idx, 1e-3)
    nodes = jnp.concatenate([einx.id("N -> N 1", jnp.arange(30)), idx], axis=1)
    M = _spectral.alignment_operator(nodes, blocks, 30)
    lam, U = _spectral.smallest_eigpairs(M, 2)
    assert bool(jnp.all(_spectral.residuals(M, lam, U) < 1e-10))
    with pytest.raises(ValueError, match="eigen_solver"):
        _spectral.smallest_eigpairs(M, 2, solver="lobpcg")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="< N"):
        _spectral.smallest_eigpairs(M, 29)


def test_unfitted_fields_are_none() -> None:
    lle = manipy.LocallyLinearEmbedding()
    assert lle.embedding is None and lle.eigenvalues is None
    assert lle.reconstruction_error is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "ltsa-ish"}, "method"),
        ({"eigen_solver": "lanczos"}, "eigen_solver"),
        ({"n_components": 0}, "n_components"),
        ({"n_neighbors": 0}, "n_neighbors"),
        ({"method": "modified", "n_neighbors": 1, "n_components": 2}, "modified"),
    ],
)
def test_invalid_config(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        manipy.LocallyLinearEmbedding(**kwargs)


def test_invalid_inputs() -> None:
    X = einx.id("(n d) -> n d", jnp.arange(10.0) ** 2, d=2)
    with pytest.raises(ValueError, match="2-D"):
        manipy.LocallyLinearEmbedding().fit(jnp.ones(5))
    with pytest.raises(ValueError, match="input dimension"):
        manipy.LocallyLinearEmbedding(n_components=3, n_neighbors=2).fit(X)
    with pytest.raises(ValueError, match="below N"):
        manipy.LocallyLinearEmbedding(n_neighbors=5).fit(X)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["standard", "modified"])
def test_swiss_roll_trustworthiness_matches_sklearn(method) -> None:
    X, _ = manipy.datasets.swiss_roll(600, key=jax.random.key(4))
    sk = SkLLE(
        n_components=2, n_neighbors=12, method=method, eigen_solver="dense"
    ).fit_transform(np.asarray(X))
    ours = manipy.LocallyLinearEmbedding(
        n_components=2, n_neighbors=12, method=method, eigen_solver="arpack"
    ).fit(X)
    t_sk = float(manipy.metrics.trustworthiness(X, sk, n_neighbors=10))
    t_ours = float(manipy.metrics.trustworthiness(X, ours.embedding, n_neighbors=10))
    assert abs(t_ours - t_sk) < 1e-3
