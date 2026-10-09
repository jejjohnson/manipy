r"""Isomap and landmark Isomap.

Isomap (Tenenbaum, de Silva & Langford, 2000) replaces Euclidean distances by
geodesic ones, the shortest-path lengths through a k-nearest-neighbour graph,
and embeds them by classical multidimensional scaling (MDS): with $D^{(2)}$
the squared geodesic distances and $J = I - \tfrac1n \mathbf 1\mathbf 1^\top$,

$$
B = -\tfrac12 J D^{(2)} J, \qquad Y = U_n \Lambda_n^{1/2},
$$

$(\Lambda_n, U_n)$ the top $n$ eigenpairs of $B$. Landmark Isomap (de Silva &
Tenenbaum, 2003) runs MDS on $m \ll N$ landmarks only and places every point
by distance-based triangulation, at $O(mN \log N)$ for the geodesics and
$O(m^3)$ for the eigenproblem instead of $O(N^2 \log N)$ and $O(N^3)$.

The shortest paths run on the CPU through `scipy.sparse.csgraph.dijkstra`:
they are not traced, so `Isomap.fit` cannot be ``jit``-ted or differentiated.
"""

from __future__ import annotations

from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import scipy.sparse as sp
from jaxtyping import Array, ArrayLike, Float, Int
from scipy.sparse.csgraph import dijkstra


__all__ = ["Isomap"]

Backend = Literal["exact", "pynndescent", "sklearn"]


def _geodesic_distances(
    X: Float[Array, "N D"],
    n_neighbors: int,
    sources: np.ndarray | None,
    backend: Backend,
    random_state: int | None,
) -> np.ndarray:
    """Shortest-path lengths ``(S, N)`` from ``sources`` (all points if
    ``None``) through the connected k-NN graph, edges weighted by length."""
    graph = kl.knn_graph(
        X,
        n_neighbors,
        weighting="connectivity",
        backend=backend,
        random_state=random_state,
        ensure_connected=True,
    )
    top = graph.topology
    senders, receivers = np.asarray(top.senders), np.asarray(top.receivers)
    diff = einx.subtract("e d, e d -> e d", X[senders], X[receivers])
    lengths = np.asarray(jnp.sqrt(einx.sum("e [d]", diff**2)), dtype=np.float64)
    # csgraph reads a stored zero as "no edge": keep duplicate points joined.
    lengths = np.maximum(lengths, np.finfo(np.float64).tiny)
    N = top.n_nodes
    G = sp.csr_matrix((lengths, (senders, receivers)), shape=(N, N))
    return dijkstra(G, directed=False, indices=sources)


def _double_centre(D2: Float[Array, "a b"]) -> Float[Array, "a b"]:
    """$-\\tfrac12 J D^{(2)} J$ for a square ``D2``."""
    rows = einx.mean("a [b] -> a", D2)
    cols = einx.mean("[a] b -> b", D2)
    centred = einx.subtract(
        "a b, b -> a b", einx.subtract("a b, a -> a b", D2, rows), cols
    )
    centred = centred + jnp.mean(D2)
    return -0.5 * centred


def _top_eigpairs(
    B: Float[Array, "m m"], n: int
) -> tuple[Float[Array, " n"], Float[Array, "m n"]]:
    """The ``n`` largest eigenpairs of the symmetric ``B``, descending, with
    deterministic signs (each vector's largest-magnitude entry positive)."""
    lam, U = jnp.linalg.eigh(0.5 * (B + einx.id("a b -> b a", B)))
    lam, U = lam[::-1][:n], U[:, ::-1][:, :n]
    peak = einx.argmax("[m] n -> n", jnp.abs(U))
    signs = jnp.sign(U[peak, jnp.arange(n)])
    return lam, einx.multiply("m n, n -> m n", U, jnp.where(signs == 0, 1.0, signs))


class Isomap(eqx.Module):
    r"""Isomap: classical MDS of geodesic distances on a k-NN graph.

    The graph is `kernellib.knn_graph` (symmetrised, and joined into one
    component with ``ensure_connected=True``, as scikit-learn's ``Isomap``
    does), each edge weighted by its Euclidean length. Geodesic distances
    $D_{ij}$ are its shortest-path lengths (Dijkstra, on the CPU), and

    $$
    B = -\tfrac12 J D^{(2)} J,\quad J = I - \tfrac1N \mathbf 1\mathbf 1^\top,
    \qquad Y = U_n \Lambda_n^{1/2},
    $$

    with $(\Lambda_n, U_n)$ the $n$ largest eigenpairs of $B$. Negative
    eigenvalues (geodesics are not exactly Euclidean) are clipped to zero.

    **Landmark Isomap** (``n_landmarks=m``): $m$ landmarks are drawn at
    random and only their geodesics $\Delta \in \mathbb R^{m \times N}$ are
    computed. MDS of the landmark block $\Delta_m$ gives
    $B_m = -\tfrac12 J \Delta_m^{(2)} J = V \Lambda V^\top$, and every point
    $a$ is placed by triangulation,

    $$
    y_a = -\tfrac12 \Lambda_n^{-1/2} V_n^\top (\delta_a - \bar\delta),
    $$

    $\delta_a$ its squared geodesic distances to the landmarks and
    $\bar\delta$ the mean column of $\Delta_m^{(2)}$. Landmarks land exactly
    on their MDS coordinates; with every point a landmark this is Isomap.

    The shortest paths are not traced: `fit` runs eagerly and cannot be
    ``jit``-ted or differentiated.

    Attributes:
        n_components: Embedding dimension $n$.
        n_neighbors: Neighbours per point in the k-NN graph.
        n_landmarks: Number of landmarks $m$, or ``None`` for exact Isomap.
            Needs $n \le m \le N$.
        neighbors_backend: ``"exact"``, ``"pynndescent"`` or ``"sklearn"``
            (see `kernellib.nearest_neighbors`).
        random_state: Seed for the landmarks and approximate neighbours.
        embedding: ``(N, n_components)``, ``None`` before `fit`.
        eigenvalues: The $n$ largest eigenvalues of $B$ (or $B_m$),
            descending; ``None`` before `fit`.
        landmarks: Indices of the landmarks, ``(m,)``; ``None`` for exact
            Isomap and before `fit`.

    Examples:
        Unroll a swiss roll; the embedding keeps neighbourhoods:

        >>> import jax
        >>> import manipy
        >>> X, t = manipy.datasets.swiss_roll(400, key=jax.random.key(0))
        >>> iso = manipy.Isomap(n_components=2, n_neighbors=10).fit(X)
        >>> iso.embedding.shape
        (400, 2)
        >>> score = manipy.metrics.trustworthiness(X, iso.embedding, n_neighbors=10)
        >>> bool(score > 0.9)
        True

        Landmark Isomap with 60 landmarks gives nearly the same embedding:

        >>> lm = manipy.Isomap(n_components=2, n_neighbors=10, n_landmarks=60)
        >>> lm = lm.fit(X)
        >>> lm.landmarks.shape
        (60,)
        >>> bool(
        ...     manipy.metrics.trustworthiness(X, lm.embedding, n_neighbors=10)
        ...     > 0.9
        ... )
        True
    """

    n_components: int = eqx.field(default=2, static=True)
    n_neighbors: int = eqx.field(default=10, static=True)
    n_landmarks: int | None = eqx.field(default=None, static=True)
    neighbors_backend: Backend = eqx.field(default="exact", static=True)
    random_state: int | None = eqx.field(default=None, static=True)
    embedding: Float[Array, "N n"] | None = None
    eigenvalues: Float[Array, " n"] | None = None
    landmarks: Int[Array, " m"] | None = None

    def __check_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {self.n_neighbors}.")
        if self.n_landmarks is not None and self.n_landmarks < self.n_components:
            raise ValueError(
                f"n_landmarks must be >= n_components = {self.n_components}, "
                f"got {self.n_landmarks}."
            )

    def fit(self, X: ArrayLike) -> Isomap:
        """Embed ``X``.

        Args:
            X: Points ``(N, D)``.

        Returns:
            The fitted module.

        Raises:
            ValueError: If ``X`` is not 2-D, ``n_neighbors >= N``,
                ``n_components > N`` or ``n_landmarks > N``.
        """
        X = jnp.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, D), got shape {X.shape}.")
        N = X.shape[0]
        if self.n_neighbors >= N:
            raise ValueError(f"n_neighbors = {self.n_neighbors} must be below N = {N}.")
        m = N if self.n_landmarks is None else self.n_landmarks
        if self.n_components > m or m > N:
            raise ValueError(
                f"Need n_components <= n_landmarks <= N; got {self.n_components}, "
                f"{m} and N = {N}."
            )

        landmarks = None
        if self.n_landmarks is None:
            D = jnp.asarray(
                _geodesic_distances(
                    X, self.n_neighbors, None, self.neighbors_backend, self.random_state
                ),
                dtype=X.dtype,
            )
            lam, U = _top_eigpairs(_double_centre(D**2), self.n_components)
            lam = jnp.maximum(lam, 0.0)
            Y = einx.multiply("N n, n -> N n", U, jnp.sqrt(lam))
        else:
            key = jax.random.key(0 if self.random_state is None else self.random_state)
            landmarks = jnp.sort(jax.random.choice(key, N, (m,), replace=False))
            Delta = jnp.asarray(
                _geodesic_distances(
                    X,
                    self.n_neighbors,
                    np.asarray(landmarks),
                    self.neighbors_backend,
                    self.random_state,
                ),
                dtype=X.dtype,
            )
            D2 = Delta**2
            D2_m = D2[:, landmarks]
            lam, V = _top_eigpairs(_double_centre(D2_m), self.n_components)
            lam = jnp.maximum(lam, 0.0)
            # Pseudo-inverse transpose of the landmark embedding: V / sqrt(λ),
            # with a zero column for a zero eigenvalue.
            inv_sqrt = jnp.where(
                lam > 0, 1.0 / jnp.sqrt(jnp.where(lam > 0, lam, 1.0)), 0
            )
            pinv = einx.multiply("m n, n -> m n", V, inv_sqrt)
            mean_col = einx.mean("m [l] -> m", D2_m)
            Y = -0.5 * einx.dot(
                "m n, m N -> N n", pinv, einx.subtract("m N, m -> m N", D2, mean_col)
            )

        return eqx.tree_at(
            lambda s: (s.embedding, s.eigenvalues, s.landmarks),
            self,
            (Y, lam, landmarks),
            is_leaf=lambda x: x is None,
        )
