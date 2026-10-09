r"""Locally linear embedding and its variants: modified LLE, Hessian LLE, LTSA.

All four methods describe each point by its $k$ nearest neighbours, sum one
positive semidefinite block per point into a sparse $N \times N$ matrix $M$,
and embed with the eigenvectors of the smallest eigenvalues of $M$ after the
constant one:

- **LLE** (Roweis & Saul, 2000): reconstruction weights
  $W = \arg\min \sum_i \|x_i - \sum_j W_{ij} x_j\|^2$, rows summing to one
  over the neighbours, and $M = (I - W)^\top (I - W)$.
- **Modified LLE** (Zhang & Wang, 2007): several linearly independent weight
  vectors per point, from the "almost null space" of the local Gram matrix,
  so the weights are stable when $k > D$.
- **Hessian LLE** (Donoho & Grimes, 2003): a local estimate of the Hessian
  quadratic form from each neighbourhood's tangent coordinates.
- **LTSA** (Zhang & Zha, 2004): the projection onto the complement of each
  neighbourhood's (centred) tangent space.

The per-point problems are ``vmap``-ped; $M$ is a `gaussx.SparseOperator`.
"""

from __future__ import annotations

from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import kernellib as kl
from jaxtyping import Array, ArrayLike, Float, Int

from manipy._embeddings._spectral import (
    EIGEN_SOLVERS,
    EigenSolver,
    alignment_operator,
    fix_signs,
    smallest_eigpairs,
)


__all__ = ["LocallyLinearEmbedding"]

Method = Literal["standard", "modified", "hessian", "ltsa"]
Backend = Literal["exact", "pynndescent", "sklearn"]
_METHODS = ("standard", "modified", "hessian", "ltsa")


def _local_offsets(X: Float[Array, "N D"], idx: Int[Array, "N k"]) -> Array:
    """Neighbours relative to their point, ``(N, k, D)``."""
    return einx.subtract("N k d, N d -> N k d", X[idx], X)


def barycenter_weights(Z: Float[Array, "k D"], reg: float) -> Float[Array, " k"]:
    r"""LLE reconstruction weights of one point from its neighbour offsets.

    Solves $(G + r I) w = \mathbf 1$ with the local Gram matrix
    $G = Z Z^\top$ and $r = \mathrm{reg} \cdot \operatorname{tr} G$ (``reg``
    if the trace is zero), then rescales $w$ to sum to one.

    Args:
        Z: The ``k`` neighbours minus the point, ``(k, D)``.
        reg: Relative regularisation.

    Returns:
        The weights ``(k,)``, summing to one.
    """
    G = einx.dot("a d, b d -> a b", Z, Z)
    trace = jnp.trace(G)
    r = jnp.where(trace > 0, reg * trace, reg)
    w = jnp.linalg.solve(
        G + r * jnp.eye(G.shape[0], dtype=G.dtype), jnp.ones_like(Z[:, 0])
    )
    return w / jnp.sum(w)


def _standard_blocks(
    X: Float[Array, "N D"], idx: Int[Array, "N k"], reg: float
) -> Float[Array, "N k1 k1"]:
    """Blocks $v_i v_i^\\top$, $v_i = (1, -w_i)$ on $(i, \\mathcal N_i)$."""
    W = jax.vmap(barycenter_weights, in_axes=(0, None))(_local_offsets(X, idx), reg)
    v = jnp.concatenate([jnp.ones_like(W[:, :1]), -W], axis=1)
    return einx.multiply("N a, N b -> N a b", v, v)


def _modified_blocks(
    X: Float[Array, "N D"], idx: Int[Array, "N k"], n_components: int, tol: float
) -> Float[Array, "N k1 k1"]:
    """Blocks $\\hat W_i \\hat W_i^\\top$ of modified LLE on $(i, \\mathcal N_i)$
    (Zhang & Wang, 2007, as in scikit-learn's ``method="modified"``)."""
    k = idx.shape[1]
    nev = min(X.shape[1], k)  # non-zero eigenvalues of the local Gram matrix
    Z = _local_offsets(X, idx)
    evals, V = jnp.linalg.eigh(einx.dot("N a d, N b d -> N a b", Z, Z))
    evals, V = evals[:, ::-1], V[:, :, ::-1]  # descending
    kept = jnp.arange(k) < nev
    evals = jnp.where(kept, jnp.maximum(evals, 0.0), 0.0)

    # Regularised LLE weights, from the eigendecomposition.
    reg = 1e-3 * einx.sum("N [a]", evals)
    ones_proj = einx.sum("N [a] b -> N b", V)
    tmp = ones_proj / einx.add("N b, N -> N b", evals, reg)
    w_reg = einx.dot("N a b, N b -> N a", V, tmp)
    w_reg = einx.divide("N a, N -> N a", w_reg, einx.sum("N [a]", w_reg))

    # eta: median ratio of small to large eigenvalue mass; s_i: size of each
    # point's almost-null space.
    big = einx.sum("N [a]", evals[:, :n_components])
    rho = einx.sum("N [a]", evals[:, n_components:]) / big
    eta = jnp.median(rho)
    s = jnp.full(idx.shape[0], k - nev)
    if nev > 1:
        # Cumulative sums over the leading eigenvalues, as a triangular product.
        upper = jnp.triu(jnp.ones((nev, nev), X.dtype))
        cums = einx.dot("N a, a b -> N b", evals[:, :nev], upper)
        safe = jnp.where(cums[:, :-1] > 0, cums[:, :-1], 1.0)
        eta_range = einx.divide("N, N a -> N a", cums[:, -1], safe) - 1.0
        search = jax.vmap(jnp.searchsorted, in_axes=(0, None))
        s = s + search(eta_range[:, ::-1], eta)

    # The bottom s_i eigenvectors, as a column mask on the full k columns.
    mask = (jnp.arange(k) >= k - einx.id("N -> N 1", s)).astype(X.dtype)
    Vi = einx.multiply("N a b, N b -> N a b", V, mask)
    col_sums = einx.sum("N [a] b -> N b", Vi)
    alpha = jnp.sqrt(einx.sum("N [b]", col_sums**2) / s)
    h = einx.multiply("N, N b -> N b", alpha, mask) - col_sums
    norm_h = jnp.sqrt(einx.sum("N [b]", h**2))
    h = jnp.where(
        einx.id("N -> N 1", norm_h) < tol,
        0.0,
        einx.divide("N b, N -> N b", h, jnp.where(norm_h > 0, norm_h, 1.0)),
    )
    # W_i = V_i (I - 2 h hᵀ) + (1 - α_i) w_reg 1ᵀ on the masked columns.
    Vh = einx.dot("N a b, N b -> N a", Vi, h)
    Wi = Vi - 2.0 * einx.multiply("N a, N b -> N a b", Vh, h)
    Wi = Wi + einx.multiply("N, N a, N b -> N a b", 1.0 - alpha, w_reg, mask)
    W_hat = jnp.concatenate([-einx.id("N b -> N 1 b", mask), Wi], axis=1)
    return einx.dot("N a c, N b c -> N a b", W_hat, W_hat)


def _tangent_basis(X: Float[Array, "N D"], idx: Int[Array, "N k"], n: int) -> Array:
    """Top ``n`` left singular vectors of each centred neighbourhood,
    ``(N, k, n)``: the local tangent coordinates."""
    P = X[idx]
    P = einx.subtract("N k d, N d -> N k d", P, einx.mean("N [k] d", P))
    _, V = jnp.linalg.eigh(einx.dot("N a d, N b d -> N a b", P, P))
    return V[:, :, ::-1][:, :, :n]


def _hessian_blocks(
    X: Float[Array, "N D"], idx: Int[Array, "N k"], n: int
) -> Float[Array, "N k k"]:
    r"""Blocks $H_i H_i^\top$ of Hessian LLE on $\mathcal N_i$ (Donoho &
    Grimes, 2003)."""
    U = _tangent_basis(X, idx, n)
    # Products u_a u_b, a <= b: the n (n + 1) / 2 quadratic monomials.
    a, b = jnp.triu_indices(n)
    quad = U[:, :, a] * U[:, :, b]
    Y = jnp.concatenate([jnp.ones_like(U[:, :, :1]), U, quad], axis=2)
    Q, _ = jnp.linalg.qr(Y)
    H = Q[:, :, n + 1 :]  # orthonormal basis of the Hessian part, (N, k, dp)
    return einx.dot("N a c, N b c -> N a b", H, H)


def _ltsa_blocks(
    X: Float[Array, "N D"], idx: Int[Array, "N k"], n: int
) -> Float[Array, "N k k"]:
    r"""Blocks $I - G_i G_i^\top$ of LTSA on $\mathcal N_i$, with
    $G_i = [\mathbf 1/\sqrt k, U_i]$ (Zhang & Zha, 2004)."""
    U = _tangent_basis(X, idx, n)
    k = idx.shape[1]
    G = jnp.concatenate([jnp.full_like(U[:, :, :1], 1.0 / jnp.sqrt(k)), U], axis=2)
    GGt = einx.dot("N a c, N b c -> N a b", G, G)
    return einx.subtract("a b, N a b -> N a b", jnp.eye(k, dtype=X.dtype), GGt)


class LocallyLinearEmbedding(eqx.Module):
    r"""Locally linear embedding (LLE), modified LLE, Hessian LLE and LTSA.

    For each point $x_i$ with neighbours $\mathcal N_i$ (its $k$ nearest,
    from `kernellib.nearest_neighbors`), a block $G_i$ on $(i, \mathcal N_i)$
    (LLE, MLLE) or on $\mathcal N_i$ (HLLE, LTSA) is summed into
    $M = \sum_i S_i^\top G_i S_i$, a `gaussx.SparseOperator`:

    - ``"standard"`` (Roweis & Saul, 2000): $G_i = v_i v_i^\top$ with
      $v_i = (1, -w_i)$ and $w_i$ the reconstruction weights,
      $(Z_i Z_i^\top + r_i I)\,w_i = \mathbf 1$, $\mathbf 1^\top w_i = 1$,
      $Z_i$ the neighbours minus $x_i$ and
      $r_i = \mathrm{reg}\cdot\operatorname{tr}(Z_i Z_i^\top)$. Then
      $M = (I - W)^\top (I - W)$.
    - ``"modified"`` (Zhang & Wang, 2007): $G_i = \hat W_i \hat W_i^\top$
      with $s_i$ weight vectors per point from the bottom eigenvectors of
      $Z_i Z_i^\top$ (formulas on the API page).
    - ``"hessian"`` (Donoho & Grimes, 2003): $G_i = H_i H_i^\top$, $H_i$ an
      orthonormal basis of the quadratic part of
      $\operatorname{span}[\mathbf 1, U_i, (u_a \odot u_b)_{a \le b}]$
      after its first $n+1$ columns, $U_i$ the top $n$ left singular vectors
      of the centred neighbourhood. Needs $k > n(n+3)/2$.
    - ``"ltsa"`` (Zhang & Zha, 2004): $G_i = I - [\mathbf 1/\sqrt k, U_i]
      [\mathbf 1/\sqrt k, U_i]^\top$, the projection off the local tangent
      space.

    The embedding is the eigenvectors of the $2$-nd to $(n+1)$-th smallest
    eigenvalues of $M$; the smallest, $\approx 0$, belongs to the constant
    vector. Columns have unit norm and a positive largest-magnitude entry.

    Attributes:
        n_components: Embedding dimension $n \le D$.
        n_neighbors: Neighbours per point $k$; ``"modified"`` needs
            $k \ge n$, ``"hessian"`` $k > n(n+3)/2$.
        method: ``"standard"``, ``"modified"``, ``"hessian"`` or ``"ltsa"``.
        reg: Relative regularisation of the local Gram matrices
            (``"standard"``).
        modified_tol: Below this norm the Householder vector of
            ``"modified"`` is taken as zero.
        eigen_solver: ``"dense"`` (JAX ``eigh``, exact, $O(N^3)$),
            or ``"arpack"`` (SciPy shift-invert on the sparse $M$, CPU, not
            traced; residual-checked).
        neighbors_backend: ``"exact"``, ``"pynndescent"`` or ``"sklearn"``.
        random_state: Seed for approximate neighbours and the iterative
            eigensolvers.
        embedding: ``(N, n_components)``, ``None`` before `fit`.
        eigenvalues: The $n$ eigenvalues of $M$ used, ascending.
        reconstruction_error: Their sum, scikit-learn's
            ``reconstruction_error_``.

    Examples:
        LLE of a swiss roll, scored by trustworthiness:

        >>> import jax
        >>> import manipy
        >>> X, t = manipy.datasets.swiss_roll(400, key=jax.random.key(0))
        >>> lle = manipy.LocallyLinearEmbedding(n_components=2, n_neighbors=12)
        >>> lle = lle.fit(X)
        >>> lle.embedding.shape
        (400, 2)
        >>> bool(
        ...     manipy.metrics.trustworthiness(X, lle.embedding, n_neighbors=10)
        ...     > 0.8
        ... )
        True

        Modified LLE:

        >>> mlle = manipy.LocallyLinearEmbedding(
        ...     n_components=2, n_neighbors=12, method="modified"
        ... ).fit(X)
        >>> bool(
        ...     manipy.metrics.trustworthiness(X, mlle.embedding, n_neighbors=10)
        ...     > 0.8
        ... )
        True

        Hessian LLE and LTSA:

        >>> for method in ("hessian", "ltsa"):
        ...     Y = manipy.LocallyLinearEmbedding(
        ...         n_components=2, n_neighbors=12, method=method
        ...     ).fit(X)
        ...     T = manipy.metrics.trustworthiness(X, Y.embedding, n_neighbors=10)
        ...     print(method, bool(T > 0.8))
        hessian True
        ltsa True
    """

    n_components: int = eqx.field(default=2, static=True)
    n_neighbors: int = eqx.field(default=10, static=True)
    method: Method = eqx.field(default="standard", static=True)
    reg: float = 1e-3
    modified_tol: float = 1e-12
    eigen_solver: EigenSolver = eqx.field(default="dense", static=True)
    neighbors_backend: Backend = eqx.field(default="exact", static=True)
    random_state: int | None = eqx.field(default=None, static=True)
    embedding: Float[Array, "N n"] | None = None
    eigenvalues: Float[Array, " n"] | None = None
    reconstruction_error: Float[Array, ""] | None = None

    def __check_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {self.method!r}.")
        if self.eigen_solver not in EIGEN_SOLVERS:
            raise ValueError(
                f"eigen_solver must be one of {EIGEN_SOLVERS}, "
                f"got {self.eigen_solver!r}."
            )
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {self.n_neighbors}.")
        if self.method == "modified" and self.n_neighbors < self.n_components:
            raise ValueError(
                'method="modified" needs n_neighbors >= n_components; got '
                f"{self.n_neighbors} < {self.n_components}."
            )
        n = self.n_components
        if self.method == "hessian" and self.n_neighbors <= n * (n + 3) // 2:
            raise ValueError(
                'method="hessian" needs n_neighbors > n_components (n_components '
                f"+ 3) / 2 = {n * (n + 3) // 2}; got {self.n_neighbors}."
            )

    def fit(self, X: ArrayLike) -> LocallyLinearEmbedding:
        """Embed ``X``.

        Args:
            X: Points ``(N, D)``.

        Returns:
            The fitted module.

        Raises:
            ValueError: If ``X`` is not 2-D, ``n_components > D``,
                ``n_neighbors >= N`` or ``n_components + 1 >= N``.
            RuntimeError: If an iterative ``eigen_solver`` fails its residual
                check.
        """
        X = jnp.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, D), got shape {X.shape}.")
        N, D = X.shape
        if self.n_components > D:
            raise ValueError(
                f"n_components = {self.n_components} exceeds the input dimension "
                f"D = {D}."
            )
        if self.n_neighbors >= N:
            raise ValueError(f"n_neighbors = {self.n_neighbors} must be below N = {N}.")
        idx = kl.nearest_neighbors(
            X,
            self.n_neighbors,
            backend=self.neighbors_backend,
            random_state=self.random_state,
        ).indices
        # LLE and MLLE blocks live on (i, N_i); HLLE and LTSA blocks on N_i.
        nodes = jnp.concatenate([einx.id("N -> N 1", jnp.arange(N)), idx], axis=1)
        if self.method == "standard":
            blocks = _standard_blocks(X, idx, self.reg)
        elif self.method == "modified":
            blocks = _modified_blocks(X, idx, self.n_components, self.modified_tol)
        elif self.method == "hessian":
            blocks, nodes = _hessian_blocks(X, idx, self.n_components), idx
        else:
            blocks, nodes = _ltsa_blocks(X, idx, self.n_components), idx
        M = alignment_operator(nodes, blocks, N)
        lam, U = smallest_eigpairs(
            M,
            self.n_components,
            solver=self.eigen_solver,
            seed=0 if self.random_state is None else self.random_state,
        )
        return eqx.tree_at(
            lambda s: (s.embedding, s.eigenvalues, s.reconstruction_error),
            self,
            (fix_signs(U), lam, jnp.sum(lam)),
            is_leaf=lambda x: x is None,
        )
