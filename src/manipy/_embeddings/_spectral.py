r"""The shared eigen-step of the LLE family.

LLE, modified LLE, Hessian LLE and LTSA all end the same way: a sum of local
positive semidefinite blocks, one per point,

$$
M = \sum_i S_i^\top G_i S_i,
$$

with $S_i$ the selection of the point's neighbourhood $\mathcal N_i$ and
$G_i$ an $m \times m$ block, and the embedding is the eigenvectors of the
smallest eigenvalues of $M$ after the $\mathbf 1$ direction. `alignment_operator`
assembles $M$ as a `gaussx.SparseOperator`; `smallest_eigpairs` solves it.
"""

from __future__ import annotations

from typing import Literal

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jaxtyping import Array, Float, Int


EigenSolver = Literal["dense", "arpack"]
EIGEN_SOLVERS = ("dense", "arpack")


def alignment_operator(
    index: Int[Array, "N m"], blocks: Float[Array, "N m m"], n_nodes: int
) -> gx.SparseOperator:
    """$M = \\sum_i S_i^\\top G_i S_i$ as a sparse operator.

    Args:
        index: Row ``i`` lists the ``m`` nodes of block ``i``.
        blocks: The symmetric ``m x m`` blocks $G_i$.
        n_nodes: Size of $M$.

    Returns:
        The symmetric ``(n_nodes, n_nodes)`` operator; overlapping entries
        are summed.
    """
    idx = np.asarray(index)
    m = idx.shape[1]
    rows = einx.id("N a -> (N a b)", idx, b=m)
    cols = einx.id("N b -> (N a b)", idx, a=m)
    values = einx.id("N a b -> (N a b)", blocks)
    return gx.SparseOperator.from_coo(
        rows,
        cols,
        values,
        (n_nodes, n_nodes),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )


# `alignment_operator` stores both triangles (a non-symmetric pattern), so the
# pattern's rows / cols and the values are the full matrix.


def _to_scipy(M: gx.SparseOperator) -> sp.csr_matrix:
    values = np.asarray(M.values)
    return sp.csr_matrix((values, (M.pattern.rows, M.pattern.cols)), M.pattern.shape)


def _gershgorin(M: gx.SparseOperator) -> Float[Array, ""]:
    absrow = jax.ops.segment_sum(
        jnp.abs(M.values),
        jnp.asarray(M.pattern.rows),
        num_segments=M.pattern.shape[0],
    )
    return jnp.max(absrow)


def residuals(
    M: gx.SparseOperator, lam: Float[Array, " n"], U: Float[Array, "N n"]
) -> Float[Array, " n"]:
    """$\\|M u_k - \\lambda_k u_k\\|$ for each column of ``U``."""
    MU = jax.vmap(M.mv, in_axes=1, out_axes=1)(U)
    R = MU - einx.multiply("N n, n -> N n", U, lam)
    return jnp.sqrt(einx.sum("[N] n", R**2))


def _drop_constant(
    M: gx.SparseOperator, U: Float[Array, "N k"]
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    """Rayleigh-Ritz on the part of ``span(U)`` orthogonal to $\\mathbf 1$.

    $\\mathbf 1$ is an exact null vector of every alignment matrix here, so
    it lies in the span of the ``k`` smallest eigenvectors. When the null
    space is degenerate (a flat sheet under HLLE or LTSA) ``eigh`` may return
    any basis of it, and dropping the first column would not drop
    $\\mathbf 1$. Projecting it out and re-solving on the remaining
    ``k - 1`` directions does, and changes nothing otherwise.
    """
    P = einx.subtract("N k, k -> N k", U, einx.mean("[N] k", U))
    s, Q = jnp.linalg.eigh(einx.dot("N a, N b -> a b", P, P))
    # Drop the smallest direction (the one along 1), orthonormalise the rest.
    V = einx.dot(
        "N a, a b -> N b", P, einx.divide("a b, b -> a b", Q, jnp.sqrt(s))[:, 1:]
    )
    MV = jax.vmap(M.mv, in_axes=1, out_axes=1)(V)
    T = einx.dot("N a, N b -> a b", V, MV)
    lam, S = jnp.linalg.eigh(0.5 * (T + einx.id("a b -> b a", T)))
    return lam, einx.dot("N a, a b -> N b", V, S)


def smallest_eigpairs(
    M: gx.SparseOperator,
    n_components: int,
    *,
    solver: EigenSolver = "dense",
    seed: int = 0,
) -> tuple[Float[Array, " n"], Float[Array, "N n"]]:
    """The ``n_components`` smallest eigenpairs (ascending) of the symmetric
    positive semidefinite ``M`` orthogonal to its null vector $\\mathbf 1$.

    The ``n_components + 1`` smallest eigenvectors are computed by
    ``solver``, then $\\mathbf 1$ is projected out (`_drop_constant`).

    - ``"dense"``: ``jnp.linalg.eigh`` of the materialised matrix, exact,
      $O(N^3)$; differentiable in the values.
    - ``"arpack"``: SciPy's ``eigsh`` in shift-invert mode around a small
      negative shift $\\sigma = -10^{-10} c$, so the factorised
      $M - \\sigma I$ is positive definite (CPU, not traced).

    No Lanczos option: the bottom of these spectra is clustered (gaps of
    $\\sim 10^{-7} c$ on a swiss roll), so Lanczos on the flipped operator
    $cI - M$ (`gaussx.eig` with ``rank=``) only converges with a Krylov
    space close to $N$, where it is unreliable (gaussx#649).

    $c$ is the Gershgorin bound on $\\lambda_{\\max}(M)$. ``"arpack"``
    checks every residual $\\|Mu - \\lambda u\\|$ against
    $\\sqrt{\\varepsilon}\\, c$ and raises instead of returning wrong pairs.

    Raises:
        ValueError: For an unknown ``solver``, or ``n_components + 1`` not
            below ``N``.
        RuntimeError: If ``"arpack"`` fails the residual check.
    """
    if solver not in EIGEN_SOLVERS:
        raise ValueError(
            f"eigen_solver must be one of {EIGEN_SOLVERS}, got {solver!r}."
        )
    N = M.pattern.shape[0]
    k = n_components + 1
    if k >= N:
        raise ValueError(
            f"Need n_components + 1 < N; got n_components = {n_components} and N = {N}."
        )
    if solver == "dense":
        _, U = jnp.linalg.eigh(M.as_matrix())
        return _drop_constant(M, U[:, :k])

    c = _gershgorin(M)
    v0 = np.random.default_rng(seed).uniform(size=N)
    sigma = -1e-10 * float(c)
    lam_np, U_np = spla.eigsh(_to_scipy(M), k=k, sigma=sigma, which="LM", v0=v0)
    lam = jnp.asarray(lam_np, dtype=M.values.dtype)
    U = jnp.asarray(U_np, dtype=M.values.dtype)
    tol = jnp.sqrt(jnp.finfo(M.values.dtype).eps) * c
    if not bool(jnp.all(residuals(M, lam, U) <= tol)):
        raise RuntimeError(
            'eigen_solver="arpack" did not converge on the smallest eigenpairs '
            '(residual check failed); use eigen_solver="dense".'
        )
    return _drop_constant(M, U)


def fix_signs(U: Float[Array, "N n"]) -> Float[Array, "N n"]:
    """Flip each column so its largest-magnitude entry is positive."""
    peak = einx.argmax("[N] n -> n", jnp.abs(U))
    signs = jnp.sign(U[peak, jnp.arange(U.shape[1])])
    return einx.multiply("N n, n -> N n", U, jnp.where(signs == 0, 1.0, signs))
