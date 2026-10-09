r"""Nyström out-of-sample extension of spectral embeddings (Bengio et al., 2004).

A spectral embedding of $N$ training points is a set of eigenvectors of a
random-walk matrix $P = D^{-1}W$ built from an affinity $w$: $P y_k = \mu_k
y_k$. Writing the eigen-equation at a new point $x$, with its affinities to
the training points, extends every eigenvector to it,

$$
y_k(x) = \frac{1}{\mu_k} \sum_j \frac{w(x, x_j)}{d(x)}\, y_{jk},
\qquad d(x) = \sum_j w(x, x_j).
$$

For Laplacian / Schrödinger eigenmaps ($L y = \lambda D y$) $\mu_k = 1 -
\lambda_k$, which is Bengio et al.'s formula; for diffusion maps $\mu_k$ is
the Markov eigenvalue and $w$ the anisotropic kernel. At a training point
the affinities are the training row of $W$, so the extension reproduces the
embedding there.
"""

from __future__ import annotations

import dataclasses

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import kernellib as kl
from jaxtyping import Array, ArrayLike, Float

from manipy._embeddings import DiffusionMaps


__all__ = ["NystromExtension"]

Eigenmap = kl.LaplacianEigenmaps | kl.SchrodingerEigenmaps


def _sq_distances(A: Float[Array, "M D"], B: Float[Array, "N D"]) -> Array:
    """Squared distances by expansion, with pairs equal up to its rounding
    (100 machine epsilons relative to the squared norms) set to exactly 0."""
    sa = einx.sum("m [d]", A**2)
    sb = einx.sum("n [d]", B**2)
    norms = einx.add("m, n -> m n", sa, sb)
    d2 = norms - 2.0 * einx.dot("m d, n d -> m n", A, B)
    tol = 100 * jnp.finfo(d2.dtype).eps
    return jnp.where(d2 > tol * norms, d2, 0.0)


class NystromExtension(eqx.Module):
    r"""Nyström extension of a fitted spectral embedding to new points.

    Works with `kernellib.LaplacianEigenmaps` and
    `kernellib.SchrodingerEigenmaps` fitted on their k-NN graph with
    ``constraint="degree"``, and with `manipy.DiffusionMaps`. With
    $w(x, x_j)$ the affinity the model's graph would give a new point,
    $q(x) = \sum_j w(x, x_j)$ and the model's anisotropy $\alpha$ ($0$ for
    eigenmaps),

    $$
    w^{(\alpha)}(x, x_j) = \frac{w(x, x_j)}{q(x)^\alpha q_j^\alpha},\qquad
    y(x) = \frac{1}{\mu} \odot \sum_j
    \frac{w^{(\alpha)}(x, x_j)}{\sum_l w^{(\alpha)}(x, x_l)}\, y_j,
    $$

    with $y_j$ the training embedding and $\mu$ the eigenvalues of the
    random walk: $1 - \lambda$ for eigenmaps, $\lambda$ for diffusion maps.

    **The affinity of a new point** mirrors the training graph:

    - a dense kernel (diffusion maps with ``n_neighbors=None``):
      $w(x, x_j) = k(x, x_j)$ for every $j$;
    - a symmetrised k-NN graph: an edge to $x_j$ if $x_j$ is among the $k$
      nearest training points of $x$, *or* $x$ is closer to $x_j$ than
      $x_j$'s own $k$-th neighbour $r_j$ (so $x_j$ would list $x$), weighted
      by the graph's kernel (heat or connectivity). A query at distance
      zero from a training point is treated as that point: no self-edge.

    So a training point gets exactly its training row of $W$ and is mapped to
    its embedding (unless ``ensure_connected`` added bridge edges for it, in
    diffusion maps). For Schrödinger eigenmaps new points carry no
    potential, so the formula is exact at training points whose potential row
    is zero (unlabelled points of a label or barrier potential).

    Cost: $O(MN)$ time and memory for $M$ queries.

    Attributes:
        X: Training points ``(N, D)``; ``None`` before `fit`.
        coefficients: $y_j / \mu$, ``(N, n)``.
        kernel: Edge / affinity kernel; ``None`` for connectivity weights.
        n_neighbors: $k$ of the training graph, ``None`` for a dense kernel.
        radii: Each training point's distance to its $k$-th neighbour.
        alpha: Anisotropy $\alpha$ of the training normalisation.
        degrees: Training degrees $q_j$ (for $\alpha > 0$).

    Examples:
        Fit Laplacian eigenmaps on half of a swiss roll and extend them to
        the other half; training points come back exactly:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import kernellib as kl
        >>> import manipy
        >>> X, t = manipy.datasets.swiss_roll(600, key=jax.random.key(0))
        >>> train, test = X[:300], X[300:]
        >>> le = kl.LaplacianEigenmaps(n_components=2, n_neighbors=10).fit(train)
        >>> ext = manipy.NystromExtension().fit(le, train)
        >>> bool(jnp.allclose(ext.transform(train), le.embedding, atol=1e-5))
        True
        >>> ext.transform(test).shape
        (300, 2)

        Diffusion maps on a dense kernel:

        >>> dm = manipy.DiffusionMaps(n_components=2, alpha=1.0, t=2).fit(train)
        >>> ext = manipy.NystromExtension().fit(dm, train)
        >>> bool(jnp.allclose(ext.transform(train), dm.embedding, atol=1e-5))
        True
    """

    X: Float[Array, "N D"] | None = None
    coefficients: Float[Array, "N n"] | None = None
    kernel: kl.AbstractKernel | None = None
    n_neighbors: int | None = eqx.field(default=None, static=True)
    radii: Float[Array, " N"] | None = None
    alpha: float = 0.0
    degrees: Float[Array, " N"] | None = None

    def fit(self, model: Eigenmap | DiffusionMaps, X: ArrayLike) -> NystromExtension:
        """Set up the extension of a fitted ``model`` trained on ``X``.

        Args:
            model: A fitted `kernellib.LaplacianEigenmaps`,
                `kernellib.SchrodingerEigenmaps` or `manipy.DiffusionMaps`.
            X: The points the model was fitted on, ``(N, D)``.

        Returns:
            The fitted extension.

        Raises:
            TypeError: For another kind of model.
            ValueError: If the model is not fitted, ``X`` does not match it,
                an eigenmap was fitted on a precomputed graph or with
                ``constraint="identity"``, or an eigenvalue makes $\\mu = 0$.
        """
        X = jnp.asarray(X, dtype=float)
        if isinstance(model, DiffusionMaps):
            if model.embedding is None or model.eigenvalues is None:
                raise ValueError("DiffusionMaps is not fitted; call fit first.")
            Y, mu = model.embedding, model.eigenvalues
            kernel, k = model.fitted_kernel, model.n_neighbors
            alpha, degrees = model.alpha, model.degrees
            radii = None
            if k is not None:
                knn = kl.nearest_neighbors(
                    X,
                    k,
                    backend=model.neighbors_backend,
                    random_state=model.random_state,
                )
                radii = knn.distances[:, -1]
        elif isinstance(model, Eigenmap):
            if model.embedding is None or model.eigenvalues is None:
                raise ValueError(
                    f"{type(model).__name__} is not fitted; call fit first."
                )
            if not isinstance(model.graph, kl.KNNGraph):
                raise ValueError(
                    "The Nyström extension needs the eigenmap's own k-NN graph; "
                    "it was fitted on a precomputed graph."
                )
            if model.constraint != "degree":
                raise ValueError(
                    'The Nyström extension needs constraint="degree" (a random '
                    "walk normalisation)."
                )
            Y, mu = model.embedding, 1.0 - model.eigenvalues
            knn = model.graph
            k, radii = knn.n_neighbors, knn.distances[:, -1]
            if model.weighting == "heat":
                sigma = model.bandwidth
                if sigma is None:  # kernellib's default: the median k-NN distance
                    sigma = jnp.median(knn.distances)
                kernel = kl.RBF(lengthscale=sigma)
            else:
                kernel = None
            alpha, degrees = 0.0, None
        else:
            raise TypeError(
                "NystromExtension extends kernellib.LaplacianEigenmaps, "
                "kernellib.SchrodingerEigenmaps or manipy.DiffusionMaps; got "
                f"{type(model).__name__}."
            )
        if X.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X must be the {Y.shape[0]} training points of the model; got "
                f"shape {X.shape}."
            )
        if bool(jnp.any(mu == 0)):
            raise ValueError("An eigenvalue of the random walk is 0; cannot extend.")
        coefficients = einx.divide("N n, n -> N n", Y, mu)
        return dataclasses.replace(
            self,
            X=X,
            coefficients=coefficients,
            kernel=kernel,
            n_neighbors=k,
            radii=radii,
            alpha=alpha,
            degrees=degrees,
        )

    def affinities(self, X_new: ArrayLike) -> Float[Array, "M N"]:
        """$w(x, x_j)$ of new points to the training points (see above).

        Args:
            X_new: Query points ``(M, D)``.

        Returns:
            ``(M, N)`` non-negative affinities.

        Raises:
            ValueError: If the extension is not fitted or ``X_new`` has the
                wrong number of features.
        """
        if self.X is None:
            raise ValueError("NystromExtension is not fitted; call fit first.")
        X_new = jnp.asarray(X_new, dtype=self.X.dtype)
        if X_new.ndim != 2 or X_new.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"X_new must have {self.X.shape[1]} features; got shape {X_new.shape}."
            )
        if self.n_neighbors is None:
            assert self.kernel is not None
            return self.kernel(X_new, self.X)
        assert self.radii is not None
        D2 = _sq_distances(X_new, self.X)
        D2 = jnp.where(D2 > 0, D2, jnp.inf)  # a coincident point is "self"
        kth = -jax.lax.top_k(-D2, self.n_neighbors)[0][:, -1]
        edge = einx.less_equal("m n, m -> m n", D2, kth)
        # A training point is its neighbour's k-th neighbour at exactly r_j;
        # the slack absorbs the rounding of the two distance computations.
        reach = self.radii**2 * (1.0 + 100 * jnp.finfo(D2.dtype).eps)
        edge = edge | einx.less_equal("m n, n -> m n", D2, reach)
        weights = (
            jnp.ones_like(D2) if self.kernel is None else self.kernel(X_new, self.X)
        )
        return jnp.where(edge, weights, 0.0)

    def transform(self, X_new: ArrayLike) -> Float[Array, "M n"]:
        """Embed new points.

        Args:
            X_new: Query points ``(M, D)``.

        Returns:
            ``(M, n_components)``; at a training point, its embedding.
        """
        W = self.affinities(X_new)
        assert self.coefficients is not None
        if self.alpha > 0:
            assert self.degrees is not None
            q = einx.sum("m [n]", W)
            W = einx.divide("m n, m -> m n", W, q**self.alpha)
            W = einx.divide("m n, n -> m n", W, self.degrees**self.alpha)
        P = einx.divide("m n, m -> m n", W, einx.sum("m [n]", W))
        return einx.dot("m n, n k -> m k", P, self.coefficients)
