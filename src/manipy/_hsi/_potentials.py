"""Image-level Schrödinger potentials on top of kernellib."""

from __future__ import annotations

import einx
import jax.numpy as jnp
import kernellib as kl
import lineax as lx
from jaxtyping import Array, Float, Int

from manipy._hsi._image import image_to_array, pixel_graph


def spatial_spectral_potential_image(
    image: Float[Array, "h w d"], *, bandwidth: float | None = None
) -> lx.AbstractLinearOperator:
    """The spatial-spectral Schrödinger potential of a hyperspectral cube.

    The grid version of Cahill, Czaja and Messinger's potential: the
    Laplacian of the pixel grid (`pixel_graph`) with every edge reweighted by
    the heat kernel of the spectral distance of its two pixels,

    $$V = L(w), \\quad w_{ij} = e^{-\\|x_i - x_j\\|^2 / 2\\sigma^2}
    \\text{ for adjacent pixels } i \\sim j.$$

    Adjacent *and* alike pixels are pulled together; there is no coordinate
    search. Equivalent to
    ``kl.spatial_spectral_graph(X, pixel_graph(shape)).laplacian_operator()``.

    Args:
        image: Cube ``(H, W, D)``.
        bandwidth: Heat-kernel width $\\sigma$ on spectral distances; ``None``
            for the median spectral distance over the grid edges.

    Returns:
        A sparse symmetric positive semi-definite operator on ``H * W``
        nodes, to pass as the potential of `kernellib.SchrodingerEigenmaps`.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from manipy import hsi
        >>> cube = jax.random.normal(jax.random.key(0), (6, 5, 3))
        >>> V = hsi.spatial_spectral_potential_image(cube)
        >>> bool(jnp.allclose(V.mv(jnp.ones(30)), 0.0, atol=1e-6))
        True
    """
    X, shape = image_to_array(image)
    graph = kl.spatial_spectral_graph(X, pixel_graph(shape), bandwidth=bandwidth)
    return graph.laplacian_operator()


def partial_label_potential(
    labels: Int[Array, ...], *, unlabeled: int = 0
) -> Float[Array, "n n"]:
    """The potential that links pixels sharing a known label.

    Forwards to `kernellib.label_potential`: the Laplacian of the graph
    joining every pair of *labelled* pixels with the same label. The legacy
    MATLAB ``PartialLabelsPotential.m`` joined the first $n_i$ samples of
    each class instead of the labelled ones; this one does not.

    Note:
        The potential is dense, ``(N, N)``. Use it on subsampled images until
        kernellib provides a sparse variant.

    Args:
        labels: Labels ``(N,)`` or an ``(H, W)`` map (flattened row-major).
            Pass only the *training* labels; hide the rest with ``unlabeled``.
        unlabeled: The value marking pixels with no known label. ``0`` (the
            usual HSI background) by default, unlike `kernellib.label_potential`.

    Returns:
        The dense potential ``(N, N)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from manipy import hsi
        >>> V = hsi.partial_label_potential(jnp.array([0, 2, 0, 2]))
        >>> V[1].tolist()  # pixels 1 and 3 share label 2
        [0.0, 1.0, 0.0, -1.0]
    """
    labels = jnp.asarray(labels)
    if labels.ndim == 2:
        labels = jnp.asarray(einx.id("h w -> (h w)", labels))
    return kl.label_potential(labels, unlabeled=unlabeled)
