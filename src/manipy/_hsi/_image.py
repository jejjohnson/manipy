"""Image <-> pixel-array conversion in kernellib's row-major node order."""

from __future__ import annotations

from typing import Literal

import einx
import jax.numpy as jnp
import kernellib as kl
from jaxtyping import Array, Float, Int


def image_to_array(image: Float[Array, "h w ..."]) -> tuple[Array, tuple[int, int]]:
    """Flatten an image to one row per pixel, in row-major order.

    Pixel ``(r, c)`` of an ``(H, W)`` image becomes row ``r * W + c``. That is
    the node order of `kernellib.GridGraph`, so graph quantities computed
    from ``X`` line up with the image without any permutation.

    Warning:
        The 2016-2018 code (``manifold_learning`` and ``manilearn``) flattened
        images **column-major** (MATLAB's ``reshape``). Pixel indices, train /
        test splits and embeddings saved by those experiments do not carry
        over; recompute them with this function.

    Args:
        image: A cube ``(H, W, D)`` (``D`` spectral bands) or a label map
            ``(H, W)``.

    Returns:
        ``(X, shape)``: ``X`` of shape ``(H * W, D)`` (or ``(H * W,)`` for a
        2-D input) and the spatial ``shape = (H, W)`` that `array_to_image`
        needs to invert it.

    Raises:
        ValueError: If ``image`` is not 2-D or 3-D.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> from manipy import hsi
        >>> img = einx.id("(h w d) -> h w d", jnp.arange(12.0), h=2, w=3)
        >>> X, shape = hsi.image_to_array(img)
        >>> X.shape, shape
        ((6, 2), (2, 3))
        >>> X[1].tolist()  # pixel (0, 1), not (1, 0)
        [2.0, 3.0]
    """
    image = jnp.asarray(image)
    if image.ndim == 3:
        h, w, _ = image.shape
        return jnp.asarray(einx.id("h w d -> (h w) d", image)), (h, w)
    if image.ndim == 2:
        h, w = image.shape
        return jnp.asarray(einx.id("h w -> (h w)", image)), (h, w)
    raise ValueError(f"image must be (H, W) or (H, W, D); got shape {image.shape}")


def array_to_image(X: Float[Array, "n ..."], shape: tuple[int, int]) -> Array:
    """Invert `image_to_array`: fold pixel rows back into an image.

    Args:
        X: Pixel values ``(H * W, D)`` (an embedding, say) or ``(H * W,)``
            (predicted labels).
        shape: The spatial shape ``(H, W)``.

    Returns:
        An ``(H, W, D)`` cube, or an ``(H, W)`` map for 1-D ``X``.

    Raises:
        ValueError: If ``X`` does not have ``H * W`` rows or is not 1-D / 2-D.

    Examples:
        >>> import einx
        >>> import jax.numpy as jnp
        >>> from manipy import hsi
        >>> img = einx.id("(h w d) -> h w d", jnp.arange(24.0), h=4, w=3)
        >>> X, shape = hsi.image_to_array(img)
        >>> bool(jnp.array_equal(hsi.array_to_image(X, shape), img))
        True
    """
    X = jnp.asarray(X)
    h, w = shape
    if X.ndim not in (1, 2) or X.shape[0] != h * w:
        raise ValueError(f"X of shape {X.shape} does not fold into an {h}x{w} image")
    if X.ndim == 2:
        return jnp.asarray(einx.id("(h w) d -> h w d", X, h=h, w=w))
    return jnp.asarray(einx.id("(h w) -> h w", X, h=h, w=w))


def pixel_coordinates(shape: tuple[int, int]) -> Int[Array, "n 2"]:
    """The ``(row, column)`` of every pixel, in row-major order.

    Args:
        shape: The spatial shape ``(H, W)``.

    Returns:
        Integer array ``(H * W, 2)``; row ``i`` is the location of pixel ``i``
        of `image_to_array`.

    Examples:
        >>> from manipy import hsi
        >>> hsi.pixel_coordinates((2, 3)).tolist()
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    """
    h, w = shape
    rows, cols = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    return jnp.asarray(einx.id("c h w -> (h w) c", jnp.stack([rows, cols])))


def pixel_graph(
    shape: tuple[int, int], *, connectivity: Literal["face", "full"] = "face"
) -> kl.GridGraph:
    """The pixel-adjacency graph of an image (wraps `kernellib.grid_graph`).

    Nodes follow the row-major order of `image_to_array`. The edge list is
    never stored (see `kernellib.GridGraph`).

    Args:
        shape: The spatial shape ``(H, W)``.
        connectivity: ``"face"`` (4 neighbours) or ``"full"`` (8 neighbours).

    Returns:
        A `kernellib.GridGraph` on ``H * W`` nodes.

    Examples:
        >>> from manipy import hsi
        >>> hsi.pixel_graph((3, 4)).n_nodes
        12
    """
    return kl.grid_graph(shape, connectivity=connectivity)
