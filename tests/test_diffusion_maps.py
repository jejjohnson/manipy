"""Diffusion maps, against their definition (Coifman & Lafon, 2006)."""

from __future__ import annotations

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import pytest
import scipy.linalg as sla

import manipy


@pytest.fixture(scope="module")
def roll() -> jax.Array:
    X, _ = manipy.datasets.swiss_roll(120, key=jax.random.key(0))
    return X


def _markov(K: jax.Array, alpha: float) -> tuple[jax.Array, jax.Array]:
    """P = D^{-1} K^(α) and the stationary distribution, from the definition."""
    q = einx.sum("i [j]", K)
    Ka = einx.divide(
        "i j, j -> i j", einx.divide("i j, i -> i j", K, q**alpha), q**alpha
    )
    d = einx.sum("i [j]", Ka)
    return einx.divide("i j, i -> i j", Ka, d), d / jnp.sum(d)


def _dense_adjacency(graph, N: int) -> jax.Array:
    s, r = graph.topology.senders, graph.topology.receivers
    W = jnp.zeros((N, N)).at[s, r].set(graph.weights)
    return W + einx.id("i j -> j i", W)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_right_eigenvectors_of_the_markov_matrix(roll, alpha) -> None:
    dm = manipy.DiffusionMaps(n_components=3, alpha=alpha, t=2).fit(roll)
    P, pi = _markov(dm.fitted_kernel(roll, roll), alpha)
    Psi, lam = dm.eigenvectors, dm.eigenvalues
    PPsi = einx.dot("i j, j n -> i n", P, Psi)
    np.testing.assert_allclose(
        PPsi, einx.multiply("i n, n -> i n", Psi, lam), atol=1e-10
    )
    # π-orthonormal and π-orthogonal to ψ0 = 1.
    gram = einx.dot("i a, i b -> a b", einx.multiply("i a, i -> i a", Psi, pi), Psi)
    np.testing.assert_allclose(gram, jnp.eye(3), atol=1e-10)
    np.testing.assert_allclose(einx.dot("i n, i -> n", Psi, pi), 0.0, atol=1e-10)
    # Coordinates are λ^t ψ, eigenvalues descending below 1.
    np.testing.assert_allclose(
        dm.embedding, einx.multiply("i n, n -> i n", Psi, lam**2), rtol=1e-12
    )
    assert bool(jnp.all(jnp.diff(lam) <= 0)) and float(lam[0]) < 1.0


def test_graph_mode_uses_the_knn_graph(roll) -> None:
    dm = manipy.DiffusionMaps(n_components=2, alpha=0.5, n_neighbors=8).fit(roll)
    graph = kl.knn_graph(roll, 8, weighting=dm.fitted_kernel, ensure_connected=True)
    K = _dense_adjacency(graph, roll.shape[0])
    np.testing.assert_allclose(dm.degrees, einx.sum("i [j]", K), rtol=1e-12)
    P, _ = _markov(K, 0.5)
    PPsi = einx.dot("i j, j n -> i n", P, dm.eigenvectors)
    expected = einx.multiply("i n, n -> i n", dm.eigenvectors, dm.eigenvalues)
    np.testing.assert_allclose(PPsi, expected, atol=1e-9)
    # The default bandwidth is the median k-NN distance, as in kernellib.
    knn = kl.nearest_neighbors(roll, 8)
    np.testing.assert_allclose(
        dm.fitted_kernel.lengthscale, jnp.median(knn.distances), rtol=1e-12
    )


def test_full_embedding_gives_diffusion_distances() -> None:
    X = jax.random.normal(jax.random.key(1), (25, 2))
    t = 3
    dm = manipy.DiffusionMaps(n_components=24, alpha=0.5, t=t).fit(X)
    P, pi = _markov(dm.fitted_kernel(X, X), 0.5)
    Pt = jnp.linalg.matrix_power(P, t)
    diff = einx.subtract("i z, j z -> i j z", Pt, Pt)
    D2 = einx.sum("i j [z]", einx.divide("i j z, z -> i j z", diff**2, pi))
    Y = dm.embedding
    E2 = einx.sum("i j [n]", einx.subtract("i n, j n -> i j n", Y, Y) ** 2)
    np.testing.assert_allclose(E2, D2, atol=1e-10)


def test_kernel_and_bandwidth_agree(roll) -> None:
    a = manipy.DiffusionMaps(bandwidth=3.0).fit(roll)
    b = manipy.DiffusionMaps(kernel=kl.RBF(3.0)).fit(roll)
    np.testing.assert_allclose(a.embedding, b.embedding, atol=1e-12)
    # The default dense bandwidth is the median pairwise distance.
    c = manipy.DiffusionMaps().fit(roll)
    np.testing.assert_allclose(
        c.fitted_kernel.lengthscale, kl.estimate_lengthscale(roll, "median")
    )


@pytest.mark.slow
@pytest.mark.parametrize("n_neighbors", [None, 10])
def test_lanczos_matches_dense(n_neighbors) -> None:
    X, _ = manipy.datasets.swiss_roll(500, key=jax.random.key(2))
    dense = manipy.DiffusionMaps(alpha=0.0, n_neighbors=n_neighbors).fit(X)
    lanczos = manipy.DiffusionMaps(
        alpha=0.0, n_neighbors=n_neighbors, eigen_solver="lanczos"
    ).fit(X)
    np.testing.assert_allclose(lanczos.eigenvalues, dense.eigenvalues, rtol=1e-8)
    angles = sla.subspace_angles(np.asarray(lanczos.embedding), dense.embedding)
    assert np.max(angles) < 1e-6


def test_lanczos_residual_check_raises(roll, monkeypatch) -> None:
    def wrong(A, *, rank, key):
        N = A.in_size()
        U = jnp.linalg.qr(jax.random.normal(key, (N, rank)))[0]
        return jnp.linspace(1.0, 0.0, rank), U

    monkeypatch.setattr(gx, "eig", wrong)
    with pytest.raises(RuntimeError, match="lanczos"):
        manipy.DiffusionMaps(eigen_solver="lanczos").fit(roll)


@pytest.mark.slow
def test_alpha_one_removes_the_sampling_density() -> None:
    # A unit circle sampled with a density varying ~5x around it. With α = 1
    # (and a fixed bandwidth, the setting of Coifman & Lafon's limit) the first
    # two diffusion coordinates are still cos θ and sin θ, the Laplace-Beltrami
    # eigenfunctions; α = 1/2 and α = 0 are increasingly distorted by the
    # density. Measured angles: 0.07, 0.50, 0.93.
    u = jax.random.uniform(jax.random.key(3), (400,))
    theta = 2 * jnp.pi * u + 0.8 * jnp.sin(2 * jnp.pi * u)
    X = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

    def misfit(alpha: float) -> float:
        dm = manipy.DiffusionMaps(alpha=alpha, bandwidth=0.2).fit(X)
        angles = sla.subspace_angles(np.asarray(dm.eigenvectors), np.asarray(X))
        return float(np.max(angles))

    m0, m_half, m1 = misfit(0.0), misfit(0.5), misfit(1.0)
    assert m1 < 0.15
    assert m1 < m_half < m0


def test_unfitted_fields_are_none() -> None:
    dm = manipy.DiffusionMaps()
    assert dm.embedding is None and dm.eigenvalues is None
    assert dm.eigenvectors is None and dm.degrees is None and dm.fitted_kernel is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"eigen_solver": "arpack"}, "eigen_solver"),
        ({"n_components": 0}, "n_components"),
        ({"alpha": 1.5}, "alpha"),
        ({"t": -1.0}, "t must"),
        ({"n_neighbors": 0}, "n_neighbors"),
        ({"kernel": kl.RBF(1.0), "bandwidth": 1.0}, "bandwidth"),
    ],
)
def test_invalid_config(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        manipy.DiffusionMaps(**kwargs)


def test_invalid_inputs() -> None:
    X = jax.random.normal(jax.random.key(4), (5, 2))
    with pytest.raises(ValueError, match="2-D"):
        manipy.DiffusionMaps().fit(jnp.ones(5))
    with pytest.raises(ValueError, match="n_components"):
        manipy.DiffusionMaps(n_components=5).fit(X)
    with pytest.raises(ValueError, match="n_neighbors"):
        manipy.DiffusionMaps(n_neighbors=5).fit(X)
