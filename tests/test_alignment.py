"""Manifold alignment: maths against dense references, and behaviour."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import kernellib as kl
import numpy as np
import pytest
import scipy.linalg

import manipy
from manipy._alignment._class_graphs import class_quadratic_forms, ssma_scales
from manipy._alignment._domains import labelled_rows, make_domains


GRID = (6, 10)  # every domain is a 6 x 10 "image" of 60 pixels
N = GRID[0] * GRID[1]
DIMS = (2, 3, 4)  # different numbers of features per domain
K = 5  # neighbours


def _labels(key: jax.Array, every: int) -> tuple[jax.Array, jax.Array]:
    """Three classes; every ``every``-th sample labelled, the rest ``-1``."""
    y = jax.random.randint(key, (N,), 0, 3)
    return y, jnp.where(jnp.arange(N) % every == 0, y, -1)


@pytest.fixture(scope="module")
def three_domains() -> tuple[list[jax.Array], list[jax.Array]]:
    """Three unrelated domains with ``DIMS`` features and partial labels."""
    keys = jax.random.split(jax.random.key(0), 2 * len(DIMS))
    X, y = [], []
    for i, d in enumerate(DIMS):
        X.append(jax.random.normal(keys[2 * i], (N, d)))
        y.append(_labels(keys[2 * i + 1], 4)[1])
    return X, y


def _quad(Z: jax.Array, M: jax.Array) -> jax.Array:
    return einx.dot("n a, n m, m b -> a b", Z, M, Z)


def _dense_problem(
    method: str,
    X: list[jax.Array],
    y: list[jax.Array],
    *,
    mu: float,
    alpha: float = 1.0,
    ridge: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """``(A, B)`` built densely, straight from roadmap §4.1 and the MATLAB.

    Every graph is an ``ΣN × ΣN`` matrix and the class graphs are the dense
    label comparisons, so this shares nothing with the closed forms.
    """
    Xc = [einx.subtract("n d, d -> n d", Xi, einx.mean("[n] d -> d", Xi)) for Xi in X]
    Z = jsl.block_diag(*Xc)
    Wg = jsl.block_diag(*[kl.knn_graph(Xi, K).to_dense() for Xi in X])
    Lg = kl.graph_laplacian(Wg)
    Dg = jnp.diag(einx.sum("n [m]", Wg))
    labels = jnp.concatenate(y)
    lab = (labels >= 0).astype(Z.dtype)
    both = einx.multiply("a, b -> a b", lab, lab)
    same = einx.equal("a, b -> a b", labels, labels).astype(Z.dtype)
    Ws = both * same
    Wd = both * (1.0 - same)
    if method == "ssma":
        # Tuia's reference code (manifoldalignment.m, 'ssma' branch).
        eye = jnp.eye(Ws.shape[0], dtype=Z.dtype)
        Ws = (Ws + eye) / jnp.sum(Ws + eye) * jnp.sum(Wg)
        Wd = (Wd + eye) / jnp.sum(Wd + eye) * jnp.sum(Wg)
    Ls, Ld = kl.graph_laplacian(Ws), kl.graph_laplacian(Wd)
    if method == "wang":
        A, B = _quad(Z, Lg + mu * Ls), _quad(Z, Dg)
    elif method == "ssma":
        A, B = _quad(Z, (1 - mu) * Lg + mu * Ls), _quad(Z, Ld)
    else:
        grid = kl.grid_graph(GRID)
        P = jsl.block_diag(
            *[
                kl.graph_laplacian(kl.spatial_spectral_graph(Xi, grid).to_dense())
                for Xi in X
            ]
        )
        alpha_t = alpha * jnp.trace(Lg) / jnp.trace(P)
        A, B = _quad(Z, (1 - mu) * (Lg + alpha_t * P) + mu * Ls), _quad(Z, Dg)
    gram = einx.dot("n a, n b -> a b", Z, Z)
    lam = ridge * jnp.trace(B) / jnp.trace(gram)
    return A + lam * gram, B + lam * gram


def _match_signs(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Flip the columns of ``G`` to agree in sign with ``F``."""
    signs = np.sign(einx.sum("[n] k", F * G))
    return G * signs


# -- class graphs --------------------------------------------------------------


def test_closed_form_class_terms_match_dense(three_domains) -> None:
    X, y = three_domains
    domains = make_domains(X, y, n_neighbors=K, weighting="heat", bandwidth=None)
    Z_l, y_l = labelled_rows(domains)
    Ls, Ld, counts = class_quadratic_forms(Z_l, y_l)

    Z = jsl.block_diag(*[d.X for d in domains])
    labels = jnp.concatenate(y)
    lab = (labels >= 0).astype(Z.dtype)
    both = einx.multiply("a, b -> a b", lab, lab)
    same = einx.equal("a, b -> a b", labels, labels).astype(Z.dtype)
    assert jnp.allclose(Ls, _quad(Z, kl.graph_laplacian(both * same)), atol=1e-9)
    assert jnp.allclose(Ld, _quad(Z, kl.graph_laplacian(both * (1 - same))), atol=1e-9)
    assert int(jnp.sum(counts)) == int(jnp.sum(lab))


def test_labelled_rows_sit_at_cumulative_offsets(three_domains) -> None:
    X, y = three_domains
    domains = make_domains(X, y, n_neighbors=K, weighting="heat", bandwidth=None)
    Z_l, _ = labelled_rows(domains)
    n_lab = [int(jnp.sum(yi >= 0)) for yi in y]
    # The third domain's rows: columns 5:9 hold its data, the rest are zero.
    rows = Z_l[n_lab[0] + n_lab[1] :]
    assert jnp.allclose(rows[:, 5:], domains[2].X[np.asarray(y[2]) >= 0])
    assert jnp.all(rows[:, :5] == 0)


def test_ssma_scales_match_matlab_formula() -> None:
    # Hand fixture: 6 samples, labels [0, 0, 1, -1, 1, 1]; W_g of total 10.
    labels = jnp.array([0, 0, 1, -1, 1, 1])
    lab = (labels >= 0).astype(float)
    both = einx.multiply("a, b -> a b", lab, lab)
    same = einx.equal("a, b -> a b", labels, labels).astype(float)
    eye = jnp.eye(6)
    # manifoldalignment.m: W = W + eye; W = W / sum(sum(W)) * sum(sum(Wg)).
    matlab_s = 10.0 / jnp.sum(both * same + eye)
    matlab_d = 10.0 / jnp.sum(both * (1 - same) + eye)
    scale_s, scale_d = ssma_scales(jnp.array([2, 3]), 6, 10.0)
    # By hand: sum(W_s + I) = 2² + 3² + 6 = 19, sum(W_d + I) = 25 - 13 + 6 = 18.
    assert jnp.allclose(scale_s, 10.0 / 19.0)
    assert jnp.allclose(scale_d, 10.0 / 18.0)
    assert jnp.allclose(scale_s, matlab_s)
    assert jnp.allclose(scale_d, matlab_d)


# -- the full problem ------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("method", ["wang", "ssma", "sema"])
def test_fit_matches_dense_reference(three_domains, method: str) -> None:
    """Every method against a dense ``ΣN × ΣN`` assembly and SciPy's solver.

    With three domains of 2, 3 and 4 features, the projections must be rows
    0:2, 2:5 and 5:9 of the stacked eigenvectors: the MATLAB's
    non-cumulative offsets (2:3 / 3:5 / 4:7, 1-based) fail this.
    """
    X, y = three_domains
    n = 4
    graphs = [kl.grid_graph(GRID)] * 3 if method == "sema" else None
    ma = manipy.ManifoldAlignment(
        method=method, n_components=n, mu=0.3, n_neighbors=K, ridge=1e-3
    ).fit(X, y, graphs)
    A, B = _dense_problem(method, X, y, mu=0.3, ridge=1e-3)
    evals, F = scipy.linalg.eigh(
        np.asarray(A), np.asarray(B), subset_by_index=[0, n - 1]
    )

    assert ma.eigenvalues is not None and ma.projections is not None
    assert np.allclose(ma.eigenvalues, evals, rtol=1e-6, atol=1e-9)
    stacked = np.concatenate([np.asarray(P) for P in ma.projections])
    assert np.allclose(_match_signs(F, stacked), F, atol=1e-6)
    assert [P.shape for P in ma.projections] == [(d, n) for d in DIMS]


def test_transform_shapes_three_domains(three_domains) -> None:
    X, y = three_domains
    ma = manipy.ManifoldAlignment(n_components=3, n_neighbors=K).fit(X, y)
    assert ma.projections is not None and ma.means is not None
    for i, d in enumerate(DIMS):
        assert ma.projections[i].shape == (d, 3)
        assert ma.means[i].shape == (d,)
        assert ma.transform(X[i][:7], domain=i).shape == (7, 3)


@pytest.mark.slow
def test_sema_without_potential_is_scaled_wang(three_domains) -> None:
    """``A_sema = (1 - μ) A_wang(μ / (1 - μ))`` and ``B`` is the same."""
    X, y = three_domains
    mu = 0.4
    graphs = [kl.grid_graph(GRID)] * 3
    common = {"n_components": 4, "n_neighbors": K, "ridge": 0.0}
    sema = manipy.ManifoldAlignment(method="sema", mu=mu, alpha=0.0, **common)
    sema = sema.fit(X, y, graphs)
    wang = manipy.ManifoldAlignment(method="wang", mu=mu / (1 - mu), **common)
    wang = wang.fit(X, y)
    assert sema.eigenvalues is not None and wang.eigenvalues is not None
    assert sema.projections is not None and wang.projections is not None
    assert jnp.allclose(sema.eigenvalues, (1 - mu) * wang.eigenvalues, rtol=1e-6)
    for Ps, Pw in zip(sema.projections, wang.projections, strict=True):
        Ps, Pw = np.asarray(Ps), np.asarray(Pw)
        assert np.allclose(_match_signs(Pw, Ps), Pw, atol=1e-6)


@pytest.mark.slow
def test_normalize_potential_scales_alpha(three_domains) -> None:
    """``normalize_potential=False`` with α̃ passed in matches the default."""
    X, y = three_domains
    graphs = [kl.grid_graph(GRID)] * 3
    common = {"method": "sema", "n_components": 3, "n_neighbors": K}
    normalised = manipy.ManifoldAlignment(alpha=0.5, **common).fit(X, y, graphs)
    domains = make_domains(X, y, n_neighbors=K, weighting="heat", bandwidth=None)
    trace_l = sum(float(jnp.sum(d.graph.degree())) for d in domains)
    trace_p = sum(
        float(jnp.sum(kl.spatial_spectral_graph(d.X, graphs[0]).degree()))
        for d in domains
    )
    raw = manipy.ManifoldAlignment(
        alpha=0.5 * trace_l / trace_p, normalize_potential=False, **common
    ).fit(X, y, graphs)
    assert normalised.eigenvalues is not None and raw.eigenvalues is not None
    assert jnp.allclose(normalised.eigenvalues, raw.eigenvalues, rtol=1e-6)


# -- behaviour -------------------------------------------------------------------


def _rotated_pair() -> tuple[jax.Array, jax.Array, jax.Array]:
    """One 3-class point cloud and a rotated, slightly noisy copy of it."""
    k1, k2, k3 = jax.random.split(jax.random.key(1), 3)
    y = jnp.repeat(jnp.arange(3), N // 3)
    centres = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    X = centres[y] + jax.random.normal(k1, (N, 3))
    Q, _ = jnp.linalg.qr(jax.random.normal(k2, (3, 3)))
    X_rot = einx.dot("n a, a b -> n b", X, Q) + 0.05 * jax.random.normal(k3, (N, 3))
    return X, X_rot, y


@pytest.mark.slow
def test_matching_points_meet_as_mu_grows() -> None:
    """The Procrustes error between the two embeddings of the same points
    (no post-hoc rotation: the alignment is the map) falls as ``μ`` grows."""
    X, X_rot, y = _rotated_pair()
    partial = jnp.where(jnp.arange(N) % 10 == 0, y, -1)  # 6 labels per domain

    def error(mu: float) -> float:
        ma = manipy.ManifoldAlignment(n_components=2, mu=mu).fit(
            [X, X_rot], [partial, partial]
        )
        E, E_rot = ma.transform(X, domain=0), ma.transform(X_rot, domain=1)
        return float(
            jnp.linalg.norm(E - E_rot)
            / jnp.sqrt(jnp.linalg.norm(E) * jnp.linalg.norm(E_rot))
        )

    # Weakly coupled (small μ), the smallest generalised eigenvectors of
    # this symmetric pair can be anti-aligned across the domains (error
    # about √2); enough label weight switches them to aligned ones.
    errors = [error(mu) for mu in (0.1, 0.5, 0.9)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.1


def test_standardize_zscores_labelled_embeddings(three_domains) -> None:
    X, y = three_domains
    ma = manipy.ManifoldAlignment(n_components=3, n_neighbors=K, standardize=True)
    ma = ma.fit(X, y)
    for i in range(3):
        E = ma.transform(X[i][np.asarray(y[i]) >= 0], domain=i)
        assert jnp.allclose(einx.mean("[n] k", E), 0.0, atol=1e-9)
        var = einx.sum("[n] k", E**2) / (E.shape[0] - 1)
        assert jnp.allclose(var, 1.0)


def test_standardize_falls_back_to_all_samples(three_domains) -> None:
    X, y = three_domains
    y = [y[0], jnp.full(N, -1), y[2]]  # domain 1 has no labels
    ma = manipy.ManifoldAlignment(n_components=3, n_neighbors=K, standardize=True)
    E = ma.fit(X, y).transform(X[1], domain=1)
    assert jnp.allclose(einx.mean("[n] k", E), 0.0, atol=1e-9)


def test_unfitted_fields_are_none() -> None:
    ma = manipy.ManifoldAlignment()
    assert ma.projections is None and ma.eigenvalues is None
    with pytest.raises(ValueError, match="not fitted"):
        ma.transform(jnp.ones((2, 2)), domain=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "lda"}, "method"),
        ({"n_components": 0}, "n_components"),
        ({"ridge": -1.0}, "ridge"),
        ({"mu": 1.5}, "mu"),
    ],
)
def test_invalid_configuration(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        manipy.ManifoldAlignment(**kwargs)


def test_wang_accepts_mu_above_one(three_domains) -> None:
    X, y = three_domains
    ma = manipy.ManifoldAlignment(method="wang", mu=2.0, n_components=2, n_neighbors=K)
    assert ma.fit(X, y).eigenvalues is not None


def test_invalid_fit_inputs(three_domains) -> None:
    X, y = three_domains
    ma = manipy.ManifoldAlignment(n_components=2, n_neighbors=K)
    with pytest.raises(ValueError, match="label arrays"):
        ma.fit(X, y[:2])
    with pytest.raises(ValueError, match="At least one"):
        ma.fit([], [])
    with pytest.raises(ValueError, match="2-D"):
        ma.fit([X[0][:, 0]], [y[0]])
    with pytest.raises(ValueError, match="shape"):
        ma.fit([X[0]], [y[0][:-1]])
    with pytest.raises(ValueError, match="integer"):
        ma.fit([X[0]], [y[0].astype(float)])
    with pytest.raises(ValueError, match="n_neighbors"):
        ma.fit([X[0][:K]], [y[0][:K]])
    with pytest.raises(ValueError, match="exceeds"):
        manipy.ManifoldAlignment(n_components=10, n_neighbors=K).fit(X, y)
    with pytest.raises(ValueError, match="No labelled"):
        ma.fit(X, [jnp.full(N, -1)] * 3)
    with pytest.raises(ValueError, match="two classes"):
        ma.fit(X, [jnp.where(yi >= 0, 0, -1) for yi in y])


def test_sema_needs_matching_spatial_graphs(three_domains) -> None:
    X, y = three_domains
    ma = manipy.ManifoldAlignment(method="sema", n_components=2, n_neighbors=K)
    with pytest.raises(ValueError, match="spatial_graphs"):
        ma.fit(X, y)
    with pytest.raises(ValueError, match="spatial graphs"):
        ma.fit(X, y, [kl.grid_graph(GRID)])
    with pytest.raises(ValueError, match="nodes"):
        ma.fit(X, y, [kl.grid_graph((5, 5))] * 3)


def test_transform_rejects_bad_domain_and_shape(three_domains) -> None:
    X, y = three_domains
    ma = manipy.ManifoldAlignment(n_components=2, n_neighbors=K).fit(X, y)
    with pytest.raises(ValueError, match="domain must be"):
        ma.transform(X[0], domain=3)
    with pytest.raises(ValueError, match="features"):
        ma.transform(X[1], domain=0)


def test_rank_deficient_domain_raises() -> None:
    key = jax.random.key(3)
    latent = jax.random.normal(key, (N, 2))
    # Four features, each a copy of one of two: the centred Gram is singular.
    copies = jnp.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    X_bad = einx.dot("n a, a b -> n b", latent, copies)
    X_ok = jax.random.normal(jax.random.key(4), (N, 2))
    y = _labels(jax.random.key(5), 3)[1]
    with pytest.raises(ValueError, match="singular"):
        manipy.ManifoldAlignment(n_components=2, n_neighbors=K).fit(
            [X_bad, X_ok], [y, y]
        )
