"""Classification and embedding-quality metrics, against sklearn references."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.manifold import trustworthiness as sk_trustworthiness

from manipy import metrics


@pytest.fixture
def labels() -> tuple[jax.Array, jax.Array]:
    k1, k2 = jax.random.split(jax.random.key(0))
    y = jax.random.randint(k1, (200,), 0, 4)
    flip = jax.random.bernoulli(k2, 0.3, (200,))
    return y, jnp.where(flip, (y + 1) % 4, y)


def test_confusion_and_accuracies(labels) -> None:
    y, p = labels
    C = metrics.confusion_matrix(y, p)
    np.testing.assert_array_equal(C, skm.confusion_matrix(y, p))
    np.testing.assert_allclose(metrics.overall_accuracy(y, p), skm.accuracy_score(y, p))
    np.testing.assert_allclose(
        metrics.average_accuracy(y, p), skm.balanced_accuracy_score(y, p)
    )
    np.testing.assert_allclose(
        metrics.per_class_accuracy(y, p), skm.recall_score(y, p, average=None)
    )


def test_kappa_matches_sklearn(labels) -> None:
    y, p = labels
    np.testing.assert_allclose(metrics.cohen_kappa(y, p), skm.cohen_kappa_score(y, p))


def test_kappa_variance_matches_a_loop_reference(labels) -> None:
    y, p = labels
    C = np.asarray(skm.confusion_matrix(y, p), dtype=float)
    n, k = C.sum(), C.shape[0]
    r, c = C.sum(1), C.sum(0)
    t1 = np.trace(C) / n
    t2 = sum(r[i] * c[i] for i in range(k)) / n**2
    t3 = sum(C[i, i] * (r[i] + c[i]) for i in range(k)) / n**2
    t4 = sum(C[i, j] * (r[j] + c[i]) ** 2 for i in range(k) for j in range(k)) / n**3
    ref = (
        t1 * (1 - t1) / (1 - t2) ** 2
        + 2 * (1 - t1) * (2 * t1 * t2 - t3) / (1 - t2) ** 3
        + (1 - t1) ** 2 * (t4 - 4 * t2**2) / (1 - t2) ** 4
    ) / n
    np.testing.assert_allclose(metrics.cohen_kappa_variance(y, p), ref, rtol=1e-10)
    assert 0.0 < float(metrics.cohen_kappa_variance(y, p)) < 0.01


def test_class_absent_from_truth_is_skipped_by_aa() -> None:
    y = jnp.array([0, 0, 1, 1])
    p = jnp.array([0, 1, 1, 2])  # class 2 is predicted but never true
    pca = metrics.per_class_accuracy(y, p)
    assert pca.shape == (3,) and bool(jnp.isnan(pca[2]))
    np.testing.assert_allclose(metrics.average_accuracy(y, p), 0.5)


def test_jit_with_n_classes(labels) -> None:
    y, p = labels
    oa = jax.jit(lambda a, b: metrics.cohen_kappa(a, b, n_classes=4))(y, p)
    np.testing.assert_allclose(oa, skm.cohen_kappa_score(y, p))


@pytest.fixture
def embedding() -> tuple[jax.Array, jax.Array]:
    kx, ky = jax.random.split(jax.random.key(1))
    X = jax.random.normal(kx, (60, 6))
    return X, X[:, :2] + 0.3 * jax.random.normal(ky, (60, 2))


@pytest.mark.parametrize("k", [3, 8])
def test_trustworthiness_matches_sklearn(embedding, k: int) -> None:
    X, Y = embedding
    np.testing.assert_allclose(
        metrics.trustworthiness(X, Y, n_neighbors=k),
        sk_trustworthiness(np.asarray(X), np.asarray(Y), n_neighbors=k),
        rtol=1e-8,
    )


def test_continuity_is_trustworthiness_with_swapped_arguments(embedding) -> None:
    X, Y = embedding
    np.testing.assert_allclose(
        metrics.continuity(X, Y, n_neighbors=5),
        sk_trustworthiness(np.asarray(Y), np.asarray(X), n_neighbors=5),
        rtol=1e-8,
    )


def test_perfect_and_random_embeddings(embedding) -> None:
    X, Y = embedding
    n, k = X.shape[0], 5
    assert float(metrics.trustworthiness(X, X, n_neighbors=k)) == 1.0
    assert float(metrics.continuity(X, X, n_neighbors=k)) == 1.0
    assert float(metrics.knn_preservation(X, X, n_neighbors=k)) == 1.0
    np.testing.assert_allclose(metrics.lcmc(X, X, n_neighbors=k), 1 - k / (n - 1))
    junk = jax.random.normal(jax.random.key(9), (n, 2))
    assert float(metrics.lcmc(X, Y, n_neighbors=k)) > float(
        metrics.lcmc(X, junk, n_neighbors=k)
    )
    assert float(metrics.knn_preservation(X, junk, n_neighbors=k)) < 0.3


def test_knn_preservation_against_brute_force(embedding) -> None:
    X, Y = embedding
    k = 4

    def knn(Z):
        d = np.linalg.norm(np.asarray(Z)[:, None] - np.asarray(Z)[None], axis=-1)
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1)[:, :k]

    a, b = knn(X), knn(Y)
    ref = np.mean([len(set(a[i]) & set(b[i])) / k for i in range(len(a))])
    np.testing.assert_allclose(metrics.knn_preservation(X, Y, n_neighbors=k), ref)


def test_quality_argument_errors(embedding) -> None:
    X, Y = embedding
    with pytest.raises(ValueError, match="same number"):
        metrics.lcmc(X, Y[:-1])
    with pytest.raises(ValueError, match="n_neighbors"):
        metrics.knn_preservation(X, Y, n_neighbors=0)
    with pytest.raises(ValueError, match=r"N / 1\.5"):
        metrics.trustworthiness(X, Y, n_neighbors=50)
