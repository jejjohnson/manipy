r"""Per-domain data and the joint block assembly of manifold alignment.

Every joint matrix of the alignment is block diagonal over the domains except
the class terms (`manipy._alignment._class_graphs`), so nothing here is ever
formed at ``ΣN × ΣN`` or ``ΣN × ΣD`` size. With $Z = \operatorname{blkdiag}
(\bar X_1, \dots, \bar X_m)$ and a block-diagonal graph $W_g =
\operatorname{blkdiag}(W_1, \dots, W_m)$,

$$
Z^\top L_g Z = \operatorname{blkdiag}_i(\bar X_i^\top L_i \bar X_i), \qquad
Z^\top D_g Z = \operatorname{blkdiag}_i(\bar X_i^\top D_i \bar X_i),
$$

each computed with the domain's sparse `laplacian_operator()` or its degree
vector, in $O(|E_i| D_i + N_i D_i^2)$.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import kernellib as kl
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int


__all__ = [
    "Domain",
    "Weighting",
    "block_diagonal",
    "geometric_terms",
    "labelled_rows",
    "make_domains",
    "potential_term",
]

# The edge weightings `kernellib.knn_graph` accepts (kernellib does not export
# its own alias).
Weighting = Literal["heat", "connectivity", "cosine"] | kl.AbstractKernel


class Domain(eqx.Module):
    """One domain of an alignment problem: centred data, labels and its graph.

    Attributes:
        X: The domain's samples centred at ``mean``, ``(N_i, D_i)``.
        mean: The per-feature mean that was removed, ``(D_i,)``.
        labels: Integer labels, ``-1`` for unlabelled, ``(N_i,)``.
        graph: The k-nearest-neighbour graph of the samples (no edges to any
            other domain).
    """

    X: Float[Array, "N D"]
    mean: Float[Array, " D"]
    labels: Int[Array, " N"]
    graph: kl.AbstractGraph

    @property
    def labelled(self) -> np.ndarray:
        """Boolean mask of the labelled samples (concrete, so not under jit)."""
        return np.asarray(self.labels) >= 0


def make_domains(
    X: Sequence[ArrayLike],
    y: Sequence[ArrayLike],
    *,
    n_neighbors: int,
    weighting: Weighting,
    bandwidth: float | None,
) -> tuple[Domain, ...]:
    """Validate and centre each domain and build its k-NN graph.

    Args:
        X: One ``(N_i, D_i)`` array per domain.
        y: One ``(N_i,)`` integer label array per domain, ``-1`` for
            unlabelled.
        n_neighbors: Neighbours per sample in each domain's graph.
        weighting: Edge weighting, see `kernellib.knn_graph`.
        bandwidth: Heat-kernel width, ``None`` for the median distance.

    Returns:
        One `Domain` per input domain.

    Raises:
        ValueError: For mismatched lengths or shapes, or a domain with too few
            samples for its neighbours.
    """
    if len(X) != len(y):
        raise ValueError(f"Got {len(X)} data arrays but {len(y)} label arrays.")
    if len(X) == 0:
        raise ValueError("At least one domain is needed.")
    domains = []
    for i, (Xi, yi) in enumerate(zip(X, y, strict=True)):
        Xi = jnp.asarray(Xi)
        yi = jnp.asarray(yi)
        if Xi.ndim != 2:
            raise ValueError(f"X[{i}] must be 2-D (N_i, D_i), got {Xi.shape}.")
        if yi.shape != Xi.shape[:1]:
            raise ValueError(
                f"y[{i}] must have shape ({Xi.shape[0]},), got {yi.shape}."
            )
        if not jnp.issubdtype(yi.dtype, jnp.integer):
            raise ValueError(f"y[{i}] must hold integer labels, got {yi.dtype}.")
        if Xi.shape[0] <= n_neighbors:
            raise ValueError(
                f"Domain {i} has {Xi.shape[0]} samples, which is not more than "
                f"n_neighbors = {n_neighbors}."
            )
        mean = einx.mean("[n] d -> d", Xi)
        graph = kl.knn_graph(Xi, n_neighbors, weighting=weighting, bandwidth=bandwidth)
        domains.append(
            Domain(einx.subtract("n d, d -> n d", Xi, mean), mean, yi, graph)
        )
    return tuple(domains)


def block_diagonal(blocks: Sequence[Float[Array, "a b"]]) -> Float[Array, "A B"]:
    """The block-diagonal matrix of ``blocks``."""
    return jsl.block_diag(*blocks)


def _quadratic(
    M: lx.AbstractLinearOperator, F: Float[Array, "N p"]
) -> Float[Array, "p p"]:
    """$F^\\top M F$ for a (sparse) operator $M$."""
    MF = jax.vmap(M.mv, in_axes=1, out_axes=1)(F)
    return einx.dot("n a, n b -> a b", F, MF)


@eqx.filter_jit
def _domain_terms(
    X: Float[Array, "N D"], graph: kl.AbstractGraph
) -> tuple[
    Float[Array, "D D"], Float[Array, "D D"], Float[Array, "D D"], Float[Array, ""]
]:
    """$\bar X^\top L \bar X$, $\bar X^\top D \bar X$, $\bar X^\top \bar X$
    and the total weight of one domain's graph."""
    degree = graph.degree()
    lap = _quadratic(graph.laplacian_operator(), X)
    deg = einx.dot("n a, n b -> a b", einx.multiply("n a, n -> n a", X, degree), X)
    return lap, deg, einx.dot("n a, n b -> a b", X, X), jnp.sum(degree)


def geometric_terms(
    domains: Sequence[Domain],
) -> tuple[
    Float[Array, "S S"], Float[Array, "S S"], Float[Array, "S S"], Float[Array, ""]
]:
    r"""The block-diagonal geometric terms of the joint problem.

    Args:
        domains: The domains, in order.

    Returns:
        ``(ZᵀL_gZ, ZᵀD_gZ, ZᵀZ, total)``, the first three ``(ΣD, ΣD)``;
        ``total`` is the total weight $\mathbf 1^\top W_g \mathbf 1 =
        \operatorname{tr}(L_g)$ of the joint graph.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from manipy._alignment._domains import geometric_terms, make_domains
        >>> X = [
        ...     jax.random.normal(jax.random.key(i), (20, d))
        ...     for i, d in [(0, 2), (1, 3)]
        ... ]
        >>> y = [jnp.full(20, -1), jnp.full(20, -1)]
        >>> doms = make_domains(
        ...     X, y, n_neighbors=4, weighting="heat", bandwidth=None
        ... )
        >>> ZLZ, ZDZ, ZZ, total = geometric_terms(doms)
        >>> ZLZ.shape, bool(jnp.all(ZLZ[:2, 2:] == 0))  # no cross-domain block
        ((5, 5), True)
    """
    lap, deg, gram = [], [], []
    total = jnp.zeros(())
    for d in domains:
        terms = _domain_terms(d.X, d.graph)
        lap.append(terms[0])
        deg.append(terms[1])
        gram.append(terms[2])
        total = total + terms[3]
    return block_diagonal(lap), block_diagonal(deg), block_diagonal(gram), total


def potential_term(
    domains: Sequence[Domain], spatial_graphs: Sequence[kl.AbstractGraph]
) -> tuple[Float[Array, "S S"], Float[Array, ""]]:
    r"""SEMA's spatial-spectral potential $Z^\top P Z$ and its trace.

    $P_i$ is the Laplacian of `kernellib.spatial_spectral_graph` on domain
    $i$'s spatial graph $S_i$: spatially adjacent, spectrally alike samples
    are pulled together.

    Args:
        domains: The domains, in order.
        spatial_graphs: One graph per domain, on that domain's samples.

    Returns:
        ``(ZᵀPZ, tr(P))``.

    Raises:
        ValueError: For the wrong number of graphs or a node-count mismatch.
    """
    if len(spatial_graphs) != len(domains):
        raise ValueError(
            f"Got {len(spatial_graphs)} spatial graphs for {len(domains)} domains."
        )
    blocks = []
    trace = jnp.zeros(())
    for i, (d, S) in enumerate(zip(domains, spatial_graphs, strict=True)):
        if S.n_nodes != d.X.shape[0]:
            raise ValueError(
                f"spatial_graphs[{i}] has {S.n_nodes} nodes but domain {i} has "
                f"{d.X.shape[0]} samples."
            )
        # Spectral distances are translation invariant: the centred data give
        # the same graph as the raw data.
        g = kl.spatial_spectral_graph(d.X, S)
        blocks.append(_quadratic(g.laplacian_operator(), d.X))
        trace = trace + jnp.sum(g.degree())
    return block_diagonal(blocks), trace


def labelled_rows(
    domains: Sequence[Domain],
) -> tuple[Float[Array, "l S"], Int[Array, " l"]]:
    """The labelled rows of $Z$ and their labels, domain after domain.

    Row $a$ of $Z$ for a sample of domain $i$ is $\\bar x_a$ placed at the
    columns of domain $i$ (cumulative offsets), zeros elsewhere.

    Args:
        domains: The domains, in order.

    Returns:
        ``(Z_l, y_l)``, of shapes ``(ℓ, ΣD)`` and ``(ℓ,)``.
    """
    dims = [d.X.shape[1] for d in domains]
    total = sum(dims)
    rows, labels = [], []
    offset = 0
    for d, dim in zip(domains, dims, strict=True):
        mask = d.labelled
        rows.append(jnp.pad(d.X[mask], ((0, 0), (offset, total - offset - dim))))
        labels.append(d.labels[mask])
        offset += dim
    return jnp.concatenate(rows), jnp.concatenate(labels)
