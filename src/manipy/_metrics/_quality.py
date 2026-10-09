"""Embedding-quality metrics: how well neighbourhoods survive a projection.

All of them compare the ``k`` nearest neighbours of every point in the
original space ``X`` with those in the embedding ``Y``. They build the dense
``(N, N)`` distance and rank matrices, so they suit evaluation sets of up to
a few thousand points; subsample larger images.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _ranks(X: Float[Array, "n d"]) -> Int[Array, "n n"]:
    """``R[i, j]``: the rank of ``j`` among the neighbours of ``i`` (nearest is
    1); each point has rank 0 to itself."""
    X = jnp.asarray(X, dtype=float)
    sq = einx.sum("n [d] -> n", X**2)
    d2 = einx.add("i, j -> i j", sq, sq) - 2.0 * einx.dot("i d, j d -> i j", X, X)
    d2 = jnp.where(jnp.eye(X.shape[0], dtype=bool), -1.0, d2)  # self is first
    order = jnp.argsort(d2)
    return jnp.argsort(order)


def _check(X, Y, k: int, *, trust: bool = False) -> int:
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(
            f"X and Y need the same number of points; got {n} and {Y.shape[0]}"
        )
    if not 1 <= k < n:
        raise ValueError(f"n_neighbors must be in [1, N); got {k} for N = {n}")
    if trust and 2 * n - 3 * k - 1 <= 0:
        raise ValueError(
            f"n_neighbors must be below N / 1.5 for this metric; got {k}, N = {n}"
        )
    return n


def _rank_penalty(R_from, R_to, k: int, n: int) -> Float[Array, ""]:
    # Points that are k-neighbours in `R_to` but not in `R_from`, penalised by
    # how far outside the k-neighbourhood of `R_from` they fall.
    intruder = (R_to >= 1) & (R_to <= k) & (R_from > k)
    penalty = jnp.sum(jnp.where(intruder, R_from - k, 0))
    return 1.0 - 2.0 / (n * k * (2 * n - 3 * k - 1)) * penalty


def trustworthiness(
    X: Float[Array, "n d"], Y: Float[Array, "n e"], *, n_neighbors: int = 5
) -> Float[Array, ""]:
    r"""Trustworthiness of an embedding (Venna & Kaski, 2001).

    The penalty for *intruders*: points that are among the $k$ nearest
    neighbours of $i$ in the embedding ``Y`` but not in the original ``X``,
    weighted by their rank $r(i, j)$ in ``X``,

    $$T(k) = 1 - \frac{2}{nk(2n-3k-1)}\sum_{i}\sum_{j\in U_i(k)}\big(r(i,j)-k\big).$$

    ``1`` means no false neighbours. Matches
    ``sklearn.manifold.trustworthiness`` with the Euclidean metric.

    Args:
        X: Original points ``(N, D)``.
        Y: Embedded points ``(N, E)``.
        n_neighbors: Neighbourhood size $k$, below $N/1.5$.

    Returns:
        A scalar in ``[0, 1]``.

    Examples:
        >>> import jax
        >>> from manipy import metrics
        >>> X = jax.random.normal(jax.random.key(0), (30, 4))
        >>> float(metrics.trustworthiness(X, X))  # the identity embedding
        1.0
    """
    n = _check(X, Y, n_neighbors, trust=True)
    return _rank_penalty(_ranks(X), _ranks(Y), n_neighbors, n)


def continuity(
    X: Float[Array, "n d"], Y: Float[Array, "n e"], *, n_neighbors: int = 5
) -> Float[Array, ""]:
    r"""Continuity of an embedding (Venna & Kaski, 2001).

    The dual of `trustworthiness`: it penalises *extrusions*, true neighbours
    in ``X`` that the embedding ``Y`` pushed out of the $k$-neighbourhood,
    weighted by their rank in ``Y``. Equal to ``trustworthiness(Y, X)``.

    Args:
        X: Original points ``(N, D)``.
        Y: Embedded points ``(N, E)``.
        n_neighbors: Neighbourhood size $k$, below $N/1.5$.

    Returns:
        A scalar in ``[0, 1]``.

    Examples:
        >>> import jax
        >>> from manipy import metrics
        >>> X = jax.random.normal(jax.random.key(0), (30, 4))
        >>> float(metrics.continuity(X, X))
        1.0
    """
    n = _check(X, Y, n_neighbors, trust=True)
    return _rank_penalty(_ranks(Y), _ranks(X), n_neighbors, n)


def _overlap(X, Y, k: int) -> Float[Array, ""]:
    n = _check(X, Y, k)
    RX, RY = _ranks(X), _ranks(Y)
    both = (RX >= 1) & (k >= RX) & (RY >= 1) & (k >= RY)
    return jnp.sum(both) / (n * k)


def knn_preservation(
    X: Float[Array, "n d"], Y: Float[Array, "n e"], *, n_neighbors: int = 5
) -> Float[Array, ""]:
    r"""Mean fraction of each point's $k$ nearest neighbours kept by the embedding.

    $$\frac1{nk}\sum_i \big|N_k^X(i)\cap N_k^Y(i)\big|.$$

    Random neighbourhoods score $k/(n-1)$; see `lcmc` for the version that
    subtracts that.

    Args:
        X: Original points ``(N, D)``.
        Y: Embedded points ``(N, E)``.
        n_neighbors: Neighbourhood size $k$.

    Returns:
        A scalar in ``[0, 1]``.

    Examples:
        >>> import jax
        >>> from manipy import metrics
        >>> X = jax.random.normal(jax.random.key(0), (30, 4))
        >>> float(metrics.knn_preservation(X, X))
        1.0
    """
    return _overlap(X, Y, n_neighbors)


def lcmc(
    X: Float[Array, "n d"], Y: Float[Array, "n e"], *, n_neighbors: int = 5
) -> Float[Array, ""]:
    r"""Local continuity meta-criterion (Chen & Buja, 2009).

    $$\mathrm{LCMC}(k) = \frac1{nk}\sum_i
    \big|N_k^X(i)\cap N_k^Y(i)\big| - \frac{k}{n-1},$$

    the neighbourhood overlap of `knn_preservation` above what a random
    embedding would keep.

    Args:
        X: Original points ``(N, D)``.
        Y: Embedded points ``(N, E)``.
        n_neighbors: Neighbourhood size $k$.

    Returns:
        A scalar, ``1 - k/(n-1)`` for a perfect embedding.

    Examples:
        >>> import jax
        >>> from manipy import metrics
        >>> X = jax.random.normal(jax.random.key(0), (21, 4))
        >>> round(float(metrics.lcmc(X, X, n_neighbors=5)), 6)  # 1 - 5/20
        0.75
    """
    n = X.shape[0]
    return _overlap(X, Y, n_neighbors) - n_neighbors / (n - 1)
