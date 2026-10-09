r"""Diffusion maps with anisotropic normalisation (Coifman & Lafon, 2006).

A kernel $K$ on the data (a dense Gram matrix, or the weights of a k-NN
graph) defines a random walk. With the degrees $q = K\mathbf 1$,

$$
K^{(\alpha)} = D_q^{-\alpha} K D_q^{-\alpha},\qquad
d = K^{(\alpha)}\mathbf 1,\qquad P = D_d^{-1} K^{(\alpha)},
$$

and the diffusion map at time $t$ sends $x_i$ to
$(\lambda_1^t \psi_1(i), \dots, \lambda_n^t \psi_n(i))$, with
$P\psi_k = \lambda_k \psi_k$, $1 = \lambda_0 > \lambda_1 \ge \dots$.
$\alpha = 0$ is the classical normalised graph Laplacian, $\alpha = \tfrac12$
the Fokker-Planck diffusion, and $\alpha = 1$ removes the sampling density,
so $P$ approximates the Laplace-Beltrami heat kernel of the manifold.
"""

from __future__ import annotations

from typing import Literal

import einx
import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import kernellib as kl
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from manipy._embeddings._spectral import fix_signs


__all__ = ["DiffusionMaps"]

DMSolver = Literal["dense", "lanczos"]
Backend = Literal["exact", "pynndescent", "sklearn"]
_SOLVERS = ("dense", "lanczos")


def _default_kernel(
    X: Float[Array, "N D"],
    n_neighbors: int | None,
    bandwidth: float | None,
    backend: Backend,
    random_state: int | None,
) -> kl.RBF:
    """The heat kernel $\\exp(-\\|x - y\\|^2 / 2\\sigma^2)$, with $\\sigma$
    ``bandwidth`` or the median distance: over all pairs for a dense Gram,
    over the k-NN edges for a graph (kernellib's ``knn_graph`` default)."""
    if bandwidth is not None:
        sigma = jnp.asarray(bandwidth, X.dtype)
    elif n_neighbors is None:
        sigma = kl.estimate_lengthscale(X, "median")
    else:
        knn = kl.nearest_neighbors(
            X, n_neighbors, backend=backend, random_state=random_state
        )
        sigma = jnp.median(knn.distances)
    return kl.RBF(lengthscale=jnp.where(sigma > 0, sigma, 1.0))


class DiffusionMaps(eqx.Module):
    r"""Diffusion maps (Coifman & Lafon, 2006).

    The affinity $K$ is the dense Gram matrix of ``kernel`` (self-loops
    included, ``n_neighbors=None``) or ``kernel`` on the edges of the
    symmetrised, connected `kernellib.knn_graph` (no self-loops). The kernel
    defaults to the heat kernel $\exp(-\|x-y\|^2/2\sigma^2)$ with $\sigma$
    ``bandwidth`` or the median distance (all pairs for the Gram, k-NN edges
    for the graph); Coifman & Lafon's $\varepsilon$ is $2\sigma^2$.

    With $q = K\mathbf 1$, $K^{(\alpha)} = D_q^{-\alpha} K D_q^{-\alpha}$
    and $d = K^{(\alpha)}\mathbf 1$, the Markov matrix
    $P = D_d^{-1} K^{(\alpha)}$ is similar to the symmetric conjugate
    $A = D_d^{-1/2} K^{(\alpha)} D_d^{-1/2}$. Its top eigenpairs
    $A\phi_k = \lambda_k\phi_k$ give the right eigenvectors of $P$,

    $$
    \psi_k = \sqrt{\textstyle\sum_j d_j}\; D_d^{-1/2}\phi_k,
    \qquad \textstyle\sum_i \pi_i \psi_k(i)^2 = 1,\quad
    \pi = d / \sum_j d_j,
    $$

    so $\psi_0 \equiv 1$ (dropped) and the embedding is
    $\Psi_t(i) = (\lambda_k^t \psi_k(i))_{k=1}^n$. With every component,
    $\|\Psi_t(i) - \Psi_t(j)\|$ is the diffusion distance
    $\|p_t(i, \cdot) - p_t(j, \cdot)\|_{L^2(1/\pi)}$.

    Attributes:
        n_components: Embedding dimension $n \le N - 1$.
        alpha: Anisotropy $\alpha \in [0, 1]$: $0$ the normalised graph
            Laplacian, $\tfrac12$ Fokker-Planck, $1$ Laplace-Beltrami.
        t: Diffusion time $t \ge 0$; the coordinates are scaled by
            $\lambda_k^t$.
        kernel: A kernellib kernel, or ``None`` for the heat kernel.
        n_neighbors: ``None`` for the dense Gram matrix, else the k-NN graph
            with this many neighbours.
        bandwidth: Heat-kernel width $\sigma$ (``kernel=None`` only);
            ``None`` for the median heuristic.
        eigen_solver: ``"dense"`` (``eigh``, exact) or ``"lanczos"``
            (`gaussx.eig` with ``rank`` well below $N$, residual-checked).
        neighbors_backend: ``"exact"``, ``"pynndescent"`` or ``"sklearn"``.
        random_state: Seed for approximate neighbours and Lanczos.
        embedding: $\Psi_t$, ``(N, n_components)``; ``None`` before `fit`.
        eigenvalues: $\lambda_1, \dots, \lambda_n$ of $P$, descending.
        eigenvectors: $\psi_1, \dots, \psi_n$, ``(N, n_components)``.
        degrees: $q = K\mathbf 1$, the kernel density estimate, ``(N,)``.
        fitted_kernel: The kernel used (with the resolved bandwidth).

    Examples:
        Diffusion maps of a swiss roll with the Laplace-Beltrami
        normalisation:

        >>> import jax
        >>> import manipy
        >>> X, t = manipy.datasets.swiss_roll(500, key=jax.random.key(0))
        >>> dm = manipy.DiffusionMaps(
        ...     n_components=2, alpha=1.0, t=4, n_neighbors=12
        ... )
        >>> dm = dm.fit(X)
        >>> dm.embedding.shape
        (500, 2)
        >>> bool(dm.eigenvalues[0] < 1.0)
        True
        >>> bool(
        ...     manipy.metrics.trustworthiness(X, dm.embedding, n_neighbors=10)
        ...     > 0.8
        ... )
        True
    """

    n_components: int = eqx.field(default=2, static=True)
    alpha: float = 1.0
    t: float = 1.0
    kernel: kl.AbstractKernel | None = None
    n_neighbors: int | None = eqx.field(default=None, static=True)
    bandwidth: float | None = None
    eigen_solver: DMSolver = eqx.field(default="dense", static=True)
    neighbors_backend: Backend = eqx.field(default="exact", static=True)
    random_state: int | None = eqx.field(default=None, static=True)
    embedding: Float[Array, "N n"] | None = None
    eigenvalues: Float[Array, " n"] | None = None
    eigenvectors: Float[Array, "N n"] | None = None
    degrees: Float[Array, " N"] | None = None
    fitted_kernel: kl.AbstractKernel | None = None

    def __check_init__(self) -> None:
        if self.eigen_solver not in _SOLVERS:
            raise ValueError(
                f"eigen_solver must be one of {_SOLVERS}, got {self.eigen_solver!r}."
            )
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {self.alpha}.")
        if self.t < 0:
            raise ValueError(f"t must be >= 0, got {self.t}.")
        if self.n_neighbors is not None and self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {self.n_neighbors}.")
        if self.kernel is not None and self.bandwidth is not None:
            raise ValueError("bandwidth applies to the default heat kernel only.")

    def fit(self, X: ArrayLike) -> DiffusionMaps:
        """Embed ``X``.

        Args:
            X: Points ``(N, D)``.

        Returns:
            The fitted module.

        Raises:
            ValueError: If ``X`` is not 2-D, ``n_components >= N`` or
                ``n_neighbors >= N``.
            RuntimeError: If ``"lanczos"`` fails its residual check.
        """
        X = jnp.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, D), got shape {X.shape}.")
        N = X.shape[0]
        if self.n_components >= N:
            raise ValueError(
                f"n_components = {self.n_components} must be below N = {N}."
            )
        if self.n_neighbors is not None and self.n_neighbors >= N:
            raise ValueError(f"n_neighbors = {self.n_neighbors} must be below N = {N}.")
        kernel = self.kernel
        if kernel is None:
            kernel = _default_kernel(
                X,
                self.n_neighbors,
                self.bandwidth,
                self.neighbors_backend,
                self.random_state,
            )

        if self.n_neighbors is None:
            K = kernel(X, X)
            q = einx.sum("i [j]", K)
            iq = q**-self.alpha
            Ka = einx.multiply("i, i j, j -> i j", iq, K, iq)
            d = einx.sum("i [j]", Ka)
            isd = 1.0 / jnp.sqrt(d)
            A = einx.multiply("i, i j, j -> i j", isd, Ka, isd)
            A = 0.5 * (A + einx.id("i j -> j i", A))
            operator: lx.AbstractLinearOperator = lx.MatrixLinearOperator(
                A, lx.symmetric_tag
            )
        else:
            graph = kl.knn_graph(
                X,
                self.n_neighbors,
                weighting=kernel,
                backend=self.neighbors_backend,
                random_state=self.random_state,
                ensure_connected=True,
            )
            s = np.asarray(graph.topology.senders)
            r = np.asarray(graph.topology.receivers)
            q = graph.degree()
            iq = q**-self.alpha
            wa = graph.weights * iq[s] * iq[r]
            d = jax.ops.segment_sum(wa, s, N) + jax.ops.segment_sum(wa, r, N)
            isd = 1.0 / jnp.sqrt(d)
            operator = gx.SparseOperator.from_coo(
                s, r, wa * isd[s] * isd[r], (N, N), symmetric=True
            )

        lam, Phi = self._top_eigpairs(operator, self.n_components + 1)
        lam, Phi = lam[1:], Phi[:, 1:]  # drop λ0 = 1, φ0 ∝ sqrt(d)
        Psi = einx.multiply("N n, N -> N n", fix_signs(Phi), jnp.sqrt(jnp.sum(d)) * isd)
        Y = einx.multiply("N n, n -> N n", Psi, lam**self.t)
        return eqx.tree_at(
            lambda m: (
                m.embedding,
                m.eigenvalues,
                m.eigenvectors,
                m.degrees,
                m.fitted_kernel,
            ),
            self,
            (Y, lam, Psi, q, kernel),
            is_leaf=lambda x: x is None,
        )

    def _top_eigpairs(
        self, A: lx.AbstractLinearOperator, k: int
    ) -> tuple[Float[Array, " k"], Float[Array, "N k"]]:
        """The ``k`` largest eigenpairs of the symmetric ``A``, descending."""
        N = A.in_size()
        if self.eigen_solver == "dense":
            lam, U = jnp.linalg.eigh(A.as_matrix())
            return lam[::-1][:k], U[:, ::-1][:, :k]
        # Krylov rank well below N (gaussx#649), doubled until the residuals
        # pass, up to N / 2. The spectrum of A lies in [-1, 1].
        seed = 0 if self.random_state is None else self.random_state
        tol = jnp.sqrt(jnp.finfo(A.in_structure().dtype).eps)
        rank = min(N // 2, max(20 * k, 100))
        while True:
            mu, U = gx.eig(A, rank=rank, key=jax.random.key(seed))
            order = jnp.argsort(-mu)[:k]
            lam, U = mu[order], U[:, order]
            AU = jax.vmap(A.mv, in_axes=1, out_axes=1)(U)
            R = AU - einx.multiply("N k, k -> N k", U, lam)
            if bool(jnp.all(jnp.sqrt(einx.sum("[N] k", R**2)) <= tol)):
                return lam, U
            if rank >= N // 2:
                raise RuntimeError(
                    f'eigen_solver="lanczos" did not converge at rank {rank} '
                    '(residual check failed); use eigen_solver="dense".'
                )
            rank = min(2 * rank, N // 2)
