r"""Same- and different-label quadratic forms, in closed form.

The class graphs join labelled samples across *all* domains: $W_s[a, b] = 1$
if $y_a = y_b$ and $W_d[a, b] = 1$ if $y_a \ne y_b$. They are never formed.
For the complete graph $K_c$ on the $n_c$ labelled rows $Z_c$ of class $c$,

$$
Z^\top L_{K_c} Z = n_c Z_c^\top Z_c - (Z_c^\top \mathbf 1)(Z_c^\top \mathbf 1)^\top,
$$

so, with $n_{y_a}$ the size of sample $a$'s class and $S = Z_\ell^\top Y$ the
per-class column sums ($Y$ one-hot),

$$
Z^\top L_s Z = \sum_c Z^\top L_{K_c} Z
= Z_\ell^\top \operatorname{diag}(n_{y}) Z_\ell - S S^\top,
\qquad
Z^\top L_d Z = Z^\top L_{K_\ell} Z - Z^\top L_s Z,
$$

because $W_d = W_{K_\ell} - W_s$ off the diagonal (self-loops cancel in
$L = D - W$). Cost $O(\ell (\Sigma D)^2)$ time and $O((\Sigma D)^2)$ memory,
against $O(\ell^2)$ for the dense $\ell \times \ell$ graphs.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float, Int


__all__ = ["class_quadratic_forms", "ssma_scales"]


def class_quadratic_forms(
    Z_l: Float[Array, "l S"], y_l: Int[ArrayLike, " l"]
) -> tuple[Float[Array, "S S"], Float[Array, "S S"], Int[Array, " C"]]:
    r"""$Z^\top L_s Z$ and $Z^\top L_d Z$ from the labelled rows of $Z$.

    Args:
        Z_l: The labelled rows of the joint block matrix $Z$, ``(ℓ, ΣD)``.
        y_l: Their labels, ``(ℓ,)`` (any integers; ``-1`` is not expected
            here, the unlabelled rows are already dropped).

    Returns:
        ``(ZᵀL_sZ, ZᵀL_dZ, counts)``: the two ``(ΣD, ΣD)`` quadratic forms and
        the number of labelled samples per class (classes in sorted order).

    Examples:
        Two classes of two points in one 1-D domain: the same-label graph has
        the edges ``(0, 1)`` and ``(2, 3)``, so $Z^\top L_s Z = (z_0 - z_1)^2
        + (z_2 - z_3)^2$.

        >>> import jax.numpy as jnp
        >>> from manipy._alignment._class_graphs import class_quadratic_forms
        >>> Z = jnp.array([[0.0], [1.0], [3.0], [5.0]])
        >>> Ls, Ld, counts = class_quadratic_forms(Z, jnp.array([0, 0, 1, 1]))
        >>> Ls.tolist(), counts.tolist()
        ([[5.0]], [2, 2])
    """
    # Labels are concrete (they decide shapes), so the classes are found on
    # the host.
    _, inverse, counts = np.unique(
        np.asarray(y_l), return_inverse=True, return_counts=True
    )
    counts = jnp.asarray(counts)
    same, different = _quadratic_forms(Z_l, jnp.asarray(inverse), counts)
    return same, different, counts


@jax.jit
def _quadratic_forms(
    Z_l: Float[Array, "l S"], inverse: Int[Array, " l"], counts: Int[Array, " C"]
) -> tuple[Float[Array, "S S"], Float[Array, "S S"]]:
    n_labelled = Z_l.shape[0]
    onehot = jax.nn.one_hot(inverse, counts.shape[0], dtype=Z_l.dtype)
    class_size = counts[inverse].astype(Z_l.dtype)  # n_{y_a}, per labelled row
    sums = einx.dot("l a, l c -> a c", Z_l, onehot)  # Z_cᵀ1 for each class
    weighted = einx.dot(
        "l a, l b -> a b", einx.multiply("l a, l -> l a", Z_l, class_size), Z_l
    )
    same = weighted - einx.dot("a c, b c -> a b", sums, sums)
    total_sum = einx.sum("[l] a -> a", Z_l)
    complete = n_labelled * einx.dot("l a, l b -> a b", Z_l, Z_l) - einx.multiply(
        "a, b -> a b", total_sum, total_sum
    )
    return same, complete - same


def ssma_scales(
    counts: Int[ArrayLike, " C"],
    n_samples: int,
    total_weight: Float[ArrayLike, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""SSMA's rescaling of the same- and different-label graphs.

    Tuia et al.'s reference code adds the identity (over all $\Sigma N$
    samples) to each class graph and rescales it so its total weight equals
    that of the geometric graph $W_g$:
    $W_s \leftarrow \frac{\mathbf 1^\top W_g \mathbf 1}
    {\mathbf 1^\top (W_s + I)\mathbf 1}(W_s + I)$, likewise $W_d$. The
    self-loops of $I$ (and $W_s$'s diagonal, $y_a = y_a$) cancel in
    $L = D - W$, so the only effect is a scalar on each Laplacian, which is
    all this returns. With $\ell = \sum_c n_c$ labelled samples,

    $$
    \mathbf 1^\top (W_s + I)\mathbf 1 = \sum_c n_c^2 + \Sigma N, \qquad
    \mathbf 1^\top (W_d + I)\mathbf 1 = \ell^2 - \sum_c n_c^2 + \Sigma N.
    $$

    Args:
        counts: Labelled samples per class, ``(C,)``.
        n_samples: Total number of samples $\Sigma N$ over all domains.
        total_weight: $\mathbf 1^\top W_g \mathbf 1$ (both triangles).

    Returns:
        ``(scale_s, scale_d)``, multiplying $L_s$ and $L_d$.

    Examples:
        Classes of 2 and 1 labelled samples among 5 samples, and a geometric
        graph of total weight 12: $12 / (4 + 1 + 5)$ and $12 / (9 - 5 + 5)$.

        >>> import jax.numpy as jnp
        >>> from manipy._alignment._class_graphs import ssma_scales
        >>> s, d = ssma_scales(jnp.array([2, 1]), 5, 12.0)
        >>> round(float(s), 6), round(float(d), 6)
        (1.2, 1.333333)
    """
    counts = jnp.asarray(counts)
    n_labelled = jnp.sum(counts)
    same_pairs = jnp.sum(counts**2)  # ordered pairs with y_a = y_b, incl. a = b
    total_weight = jnp.asarray(total_weight)
    scale_s = total_weight / (same_pairs + n_samples)
    scale_d = total_weight / (n_labelled**2 - same_pairs + n_samples)
    return scale_s, scale_d
