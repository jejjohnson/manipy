r"""Linear manifold alignment: Wang, SSMA and SEMA.

Domains $X_i \in \mathbb R^{N_i \times D_i}$, $i = 1, \dots, m$, with partial
labels, are mapped by per-domain projections $F_i \in \mathbb R^{D_i \times n}$
into one shared $n$-dimensional space. Stacking the projections,
$F = [F_1; \dots; F_m]$, and the centred data,
$Z = \operatorname{blkdiag}(\bar X_1, \dots, \bar X_m)$, every method solves
one generalised eigenproblem of size $\Sigma D = \sum_i D_i$,

$$
(A + \lambda_r Z^\top Z) f = \lambda (B + \lambda_r Z^\top Z) f,
$$

for the $n$ smallest $\lambda$, with $A$ and $B$ given in
`ManifoldAlignment`.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import einx
import equinox as eqx
import gaussx as gx
import jax.numpy as jnp
import kernellib as kl
import lineax as lx
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from manipy._alignment._class_graphs import class_quadratic_forms, ssma_scales
from manipy._alignment._domains import (
    Weighting,
    geometric_terms,
    labelled_rows,
    make_domains,
    potential_term,
)


__all__ = ["ManifoldAlignment"]

_METHODS = ("wang", "ssma", "sema")


class ManifoldAlignment(eqx.Module):
    r"""Semi-supervised linear manifold alignment of several domains.

    Finds one linear projection per domain into a shared space in which each
    domain keeps its own neighbourhood geometry, same-class samples meet
    across domains, and (SSMA) different classes are pushed apart. Domains
    may have different numbers of features (e.g. two hyperspectral sensors
    with different bands); only a few samples need labels.

    With $L_g, D_g$ the Laplacian and degree matrix of the block-diagonal
    k-NN graph (no cross-domain edges), $L_s, L_d$ the Laplacians of the
    same- and different-label graphs over all labelled samples, and $P$ the
    block-diagonal spatial-spectral potential:

    | ``method`` | $A$ | $B$ |
    |---|---|---|
    | ``"wang"`` | $Z^\top(L_g + \mu L_s)Z$ | $Z^\top D_g Z$ |
    | ``"ssma"`` | $Z^\top((1-\mu) L_g + \mu L_s)Z$ | $Z^\top L_d Z$ |
    | ``"sema"`` | $Z^\top((1-\mu)(L_g + \tilde\alpha P) + \mu L_s)Z$ | $Z^\top D_g Z$ |

    SSMA rescales $L_s$ and $L_d$ to the total weight of $W_g$ (see
    `ssma_scales`). SEMA's $\tilde\alpha = \alpha \operatorname{tr}(L_g) /
    \operatorname{tr}(P)$ with ``normalize_potential=True``, else $\alpha$.
    The ridge is $\lambda_r = r \cdot \operatorname{tr}(B) /
    \operatorname{tr}(Z^\top Z)$ for ``ridge`` $= r$: $Z^\top Z$ normalised
    by its mean eigenvalue $\operatorname{tr}(Z^\top Z)/\Sigma D$, times $r$
    times the mean eigenvalue of $B$, so ``ridge`` is scale free.

    Projections are the rows of the stacked eigenvectors at the *cumulative*
    feature offsets of the domains, so any number of domains works.

    Attributes:
        method: ``"wang"`` (Wang & Mahadevan, 2011), ``"ssma"`` (Tuia et al.,
            2014) or ``"sema"`` (Schrödinger alignment, with a
            spatial-spectral potential; needs ``spatial_graphs`` in `fit`).
        n_components: Dimension $n$ of the shared space, at most $\Sigma D$.
        mu: Weight $\mu$ of the same-label term.
        alpha: Weight of the potential (``"sema"`` only).
        normalize_potential: Scale $\alpha$ by $\operatorname{tr}(L_g) /
            \operatorname{tr}(P)$ (``"sema"`` only). ``False`` reproduces the
            original MATLAB code.
        ridge: Relative ridge $r \ge 0$ (see above). ``0`` solves the pencil
            exactly through gaussx's singular-$B$ path ($B$ need not be
            positive definite).
        n_neighbors: Neighbours per sample in each domain's k-NN graph.
        weighting: Edge weighting of the k-NN graphs, see
            `kernellib.knn_graph`.
        bandwidth: Heat-kernel width, ``None`` for the median distance.
        standardize: z-score each domain's embedding with the mean and
            standard deviation of its *labelled* embeddings (all of its
            samples if it has fewer than two labels), as SSMA's reference code
            did before classification. ``transform`` reapplies it.
        projections: Per-domain projections $F_i$, ``(D_i, n)``; ``None``
            before `fit`.
        means: Per-domain feature means removed before projecting.
        embedding_means: Per-domain embedding means (``standardize`` only).
        embedding_scales: Per-domain embedding standard deviations
            (``standardize`` only).
        eigenvalues: The $n$ generalised eigenvalues, ascending.

    Examples:
        Two "sensors" see the same three classes through different bands (a
        random linear map of a common 2-D signal into 4 and 3 bands). Half of
        each class is labelled; the rest is to be classified.

        >>> import einx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import manipy
        >>> k_lat, k_a, k_b, k_na, k_nb = jax.random.split(jax.random.key(0), 5)
        >>> centres = jnp.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
        >>> y = jnp.repeat(jnp.arange(3), 30)
        >>> latent = centres[y] + jax.random.normal(k_lat, (90, 2))
        >>> X_a = latent @ jax.random.normal(k_a, (2, 4))  # sensor A: 4 bands
        >>> X_b = latent @ jax.random.normal(k_b, (2, 3))  # sensor B: 3 bands
        >>> X_a = X_a + 0.1 * jax.random.normal(k_na, X_a.shape)  # sensor noise
        >>> X_b = X_b + 0.1 * jax.random.normal(k_nb, X_b.shape)
        >>> partial = jnp.where(jnp.arange(90) % 2 == 0, y, -1)  # -1: unlabelled
        >>> ma = manipy.ManifoldAlignment(method="ssma", n_components=2, mu=0.5)
        >>> ma = ma.fit([X_a, X_b], [partial, partial])
        >>> [P.shape for P in ma.projections]  # rows 0:4 and 4:7 of F
        [(4, 2), (3, 2)]
        >>> Z_a = ma.transform(X_a, domain=0)
        >>> Z_b = ma.transform(X_b, domain=1)
        >>> Z_a.shape, Z_b.shape
        ((90, 2), (90, 2))

        Train on sensor A, classify sensor B in the shared space (nearest
        class mean):

        >>> means = jnp.stack(
        ...     [einx.mean("[n] k -> k", Z_a[y == c]) for c in range(3)]
        ... )
        >>> dist = einx.sum(
        ...     "m c [k]", einx.subtract("m k, c k -> m c k", Z_b, means) ** 2
        ... )
        >>> float(jnp.mean(einx.argmin("m [c] -> m", dist) == y)) > 0.85
        True
    """

    method: Literal["wang", "ssma", "sema"] = eqx.field(default="ssma", static=True)
    n_components: int = eqx.field(default=10, static=True)
    mu: float = 0.5
    alpha: float = 1.0
    normalize_potential: bool = eqx.field(default=True, static=True)
    ridge: float = 1e-6
    n_neighbors: int = eqx.field(default=10, static=True)
    weighting: Weighting = "heat"
    bandwidth: float | None = None
    standardize: bool = eqx.field(default=False, static=True)
    projections: tuple[Float[Array, "D_i n"], ...] | None = None
    means: tuple[Float[Array, " D_i"], ...] | None = None
    embedding_means: tuple[Float[Array, " n"], ...] | None = None
    embedding_scales: tuple[Float[Array, " n"], ...] | None = None
    eigenvalues: Float[Array, " n"] | None = None

    def __check_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {self.method!r}.")
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        if self.ridge < 0:
            raise ValueError(f"ridge must be >= 0, got {self.ridge}.")
        if self.method != "wang" and not 0.0 <= self.mu <= 1.0:
            raise ValueError(
                f"mu must lie in [0, 1] for {self.method!r}, got {self.mu}."
            )

    def fit(
        self,
        X: Sequence[ArrayLike],
        y: Sequence[ArrayLike],
        spatial_graphs: Sequence[kl.AbstractGraph] | None = None,
    ) -> ManifoldAlignment:
        """Fit the per-domain projections.

        Args:
            X: One ``(N_i, D_i)`` array per domain.
            y: One ``(N_i,)`` integer label array per domain, ``-1`` for
                unlabelled samples. Labels are shared across domains: class
                ``c`` in one domain is class ``c`` in every other.
            spatial_graphs: One graph per domain on its samples (e.g.
                `kernellib.grid_graph` of the image), for ``"sema"`` only.

        Returns:
            The fitted module.

        Raises:
            ValueError: For inconsistent inputs, ``n_components > ΣD``, a
                method whose class terms have too few labels, or
                ``"sema"`` without ``spatial_graphs``.
        """
        domains = make_domains(
            X,
            y,
            n_neighbors=self.n_neighbors,
            weighting=self.weighting,
            bandwidth=self.bandwidth,
        )
        dims = [d.X.shape[1] for d in domains]
        if self.n_components > sum(dims):
            raise ValueError(
                f"n_components = {self.n_components} exceeds the total number of "
                f"features ΣD = {sum(dims)}."
            )
        if self.method == "sema" and spatial_graphs is None:
            raise ValueError('method="sema" needs spatial_graphs, one per domain.')

        ZLZ, ZDZ, gram, total_weight = geometric_terms(domains)
        Z_l, y_l = labelled_rows(domains)
        if Z_l.shape[0] == 0:
            raise ValueError("No labelled samples: at least one label is needed.")
        Ls, Ld, counts = class_quadratic_forms(Z_l, y_l)
        if self.method == "ssma" and counts.shape[0] < 2:
            raise ValueError('method="ssma" needs labelled samples of two classes.')

        mu = self.mu
        if self.method == "wang":
            A = ZLZ + mu * Ls
            B = ZDZ
        elif self.method == "ssma":
            n_samples = sum(d.X.shape[0] for d in domains)
            scale_s, scale_d = ssma_scales(counts, n_samples, total_weight)
            A = (1 - mu) * ZLZ + mu * scale_s * Ls
            B = scale_d * Ld
        else:
            assert spatial_graphs is not None
            ZPZ, trace_p = potential_term(domains, spatial_graphs)
            alpha = jnp.asarray(self.alpha, ZLZ.dtype)
            if self.normalize_potential:
                # tr(L_g) is the total weight of W_g.
                alpha = alpha * total_weight / jnp.maximum(trace_p, 1e-30)
            A = (1 - mu) * (ZLZ + alpha * ZPZ) + mu * Ls
            B = ZDZ

        tagged = lx.symmetric_tag
        if self.ridge > 0:
            lam = self.ridge * jnp.trace(B) / jnp.trace(gram)
            A = A + lam * gram
            B = B + lam * gram
            tagged = lx.positive_semidefinite_tag  # positive definite: Cholesky
        A = 0.5 * (A + einx.id("a b -> b a", A))
        B = 0.5 * (B + einx.id("a b -> b a", B))
        eigenvalues, F = gx.eigh_generalized(
            lx.MatrixLinearOperator(A, lx.symmetric_tag),
            lx.MatrixLinearOperator(B, tagged),
            rank=self.n_components,
        )
        if not bool(jnp.all(jnp.isfinite(eigenvalues))):
            raise ValueError(
                "The alignment eigenproblem has no finite solution: B + λ_r ZᵀZ "
                "is singular. A domain's centred data are probably rank "
                "deficient (fewer samples than features, or collinear "
                "features); reduce its features (e.g. PCA) first."
            )

        # Cumulative offsets: domain i owns rows offsets[i]:offsets[i + 1].
        offsets = np.cumsum([0, *dims])
        projections = tuple(F[start:stop] for start, stop in pairwise(offsets))
        means = tuple(d.mean for d in domains)

        embedding_means = embedding_scales = None
        if self.standardize:
            stats = [
                _labelled_stats(einx.dot("n d, d k -> n k", d.X, P), d.labelled)
                for d, P in zip(domains, projections, strict=True)
            ]
            embedding_means = tuple(m for m, _ in stats)
            embedding_scales = tuple(s for _, s in stats)

        return eqx.tree_at(
            lambda m: (
                m.projections,
                m.means,
                m.embedding_means,
                m.embedding_scales,
                m.eigenvalues,
            ),
            self,
            (projections, means, embedding_means, embedding_scales, eigenvalues),
            is_leaf=lambda x: x is None,
        )

    def transform(self, X: ArrayLike, *, domain: int) -> Float[Array, "M n"]:
        """Map samples of one domain into the shared space.

        Args:
            X: Samples of domain ``domain``, ``(M, D_domain)``.
            domain: Index of the domain the samples come from, in the order
                given to `fit`.

        Returns:
            The embedding, ``(M, n_components)``.

        Raises:
            ValueError: If the module is not fitted, ``domain`` is out of
                range, or ``X`` has the wrong number of features.
        """
        if self.projections is None or self.means is None:
            raise ValueError("ManifoldAlignment is not fitted; call fit first.")
        if not 0 <= domain < len(self.projections):
            raise ValueError(
                f"domain must be in [0, {len(self.projections)}), got {domain}."
            )
        X = jnp.asarray(X)
        P = self.projections[domain]
        if X.ndim != 2 or X.shape[1] != P.shape[0]:
            raise ValueError(
                f"Domain {domain} has {P.shape[0]} features; got X of shape {X.shape}."
            )
        Xc = einx.subtract("m d, d -> m d", X, self.means[domain])
        E = einx.dot("m d, d k -> m k", Xc, P)
        if self.embedding_means is not None and self.embedding_scales is not None:
            E = einx.subtract("m k, k -> m k", E, self.embedding_means[domain])
            E = einx.divide("m k, k -> m k", E, self.embedding_scales[domain])
        return E


def _labelled_stats(
    E: Float[Array, "N n"], labelled: np.ndarray
) -> tuple[Float[Array, " n"], Float[Array, " n"]]:
    """Mean and standard deviation (``ddof=1``, as MATLAB's ``zscore``) of the
    labelled rows of ``E``, or of all rows with fewer than two labels."""
    rows = E[labelled] if int(labelled.sum()) >= 2 else E
    mean = einx.mean("[l] k -> k", rows)
    centred = einx.subtract("l k, k -> l k", rows, mean)
    var = einx.sum("[l] k -> k", centred**2) / (rows.shape[0] - 1)
    std = jnp.sqrt(var)
    return mean, jnp.where(std > 0, std, 1.0)
