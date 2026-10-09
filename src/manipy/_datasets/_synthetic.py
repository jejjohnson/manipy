"""Synthetic manifolds, sampled with JAX PRNG keys."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def swiss_roll(
    n_samples: int = 1000, *, key: Array, noise: float = 0.0
) -> tuple[Float[Array, "n 3"], Float[Array, " n"]]:
    r"""The Swiss roll: a 2-D sheet rolled up in 3-D.

    $x = t\cos t,\ y = 21 v,\ z = t\sin t$ with
    $t = \tfrac{3\pi}{2}(1 + 2u)$ and $u, v \sim U(0, 1)$, the construction of
    `sklearn.datasets.make_swiss_roll`.

    Args:
        n_samples: Number of points.
        key: JAX PRNG key.
        noise: Standard deviation of isotropic Gaussian noise added to ``X``.

    Returns:
        ``(X, t)``: points ``(n_samples, 3)`` and the position along the roll
        ``t``, the intrinsic coordinate to colour by.

    Examples:
        >>> import jax
        >>> from manipy import datasets
        >>> X, t = datasets.swiss_roll(100, key=jax.random.key(0))
        >>> X.shape, t.shape
        ((100, 3), (100,))
    """
    ku, kv, kn = jax.random.split(key, 3)
    t = 1.5 * jnp.pi * (1.0 + 2.0 * jax.random.uniform(ku, (n_samples,)))
    height = 21.0 * jax.random.uniform(kv, (n_samples,))
    X = jnp.stack([t * jnp.cos(t), height, t * jnp.sin(t)], axis=-1)
    return X + noise * jax.random.normal(kn, X.shape), t


def s_curve(
    n_samples: int = 1000, *, key: Array, noise: float = 0.0
) -> tuple[Float[Array, "n 3"], Float[Array, " n"]]:
    r"""The S-curve: a 2-D sheet bent into an S in 3-D.

    $x = \sin t,\ y = 2v,\ z = \operatorname{sign}(t)(\cos t - 1)$ with
    $t = 3\pi(u - \tfrac12)$, as in `sklearn.datasets.make_s_curve`.

    Args:
        n_samples: Number of points.
        key: JAX PRNG key.
        noise: Standard deviation of isotropic Gaussian noise added to ``X``.

    Returns:
        ``(X, t)``: points ``(n_samples, 3)`` and the position along the
        curve ``t``.

    Examples:
        >>> import jax
        >>> from manipy import datasets
        >>> X, t = datasets.s_curve(100, key=jax.random.key(0))
        >>> X.shape, t.shape
        ((100, 3), (100,))
    """
    ku, kv, kn = jax.random.split(key, 3)
    t = 3.0 * jnp.pi * (jax.random.uniform(ku, (n_samples,)) - 0.5)
    y = 2.0 * jax.random.uniform(kv, (n_samples,))
    X = jnp.stack([jnp.sin(t), y, jnp.sign(t) * (jnp.cos(t) - 1.0)], axis=-1)
    return X + noise * jax.random.normal(kn, X.shape), t


def severed_sphere(
    n_samples: int = 1000, *, key: Array, noise: float = 0.0, cap: float = jnp.pi / 8
) -> tuple[Float[Array, "n 3"], Float[Array, " n"]]:
    r"""A unit sphere with both polar caps severed and a meridian strip cut.

    Points are uniform in area on the band of polar angles
    $\theta \in [\text{cap}, \pi - \text{cap}]$ and azimuths
    $\varphi \in [0, 2\pi - 0.55]$, so the result is a simply connected
    patch that a 2-D embedding can unroll (a port of ``get_severed_sphere``
    from the 2016 ``manifold_learning`` code, which sampled ``theta``
    non-uniformly and returned a variable number of points).

    Args:
        n_samples: Number of points.
        key: JAX PRNG key.
        noise: Standard deviation of isotropic Gaussian noise added to ``X``.
        cap: Polar angle, in radians, severed at each pole.

    Returns:
        ``(X, phi)``: points ``(n_samples, 3)`` and the azimuth ``phi``.

    Examples:
        >>> import jax
        >>> from manipy import datasets
        >>> X, phi = datasets.severed_sphere(100, key=jax.random.key(0))
        >>> X.shape, phi.shape
        ((100, 3), (100,))
    """
    ku, kp, kn = jax.random.split(key, 3)
    lo, hi = jnp.cos(jnp.pi - cap), jnp.cos(cap)
    theta = jnp.arccos(lo + (hi - lo) * jax.random.uniform(ku, (n_samples,)))
    phi = (2.0 * jnp.pi - 0.55) * jax.random.uniform(kp, (n_samples,))
    X = jnp.stack(
        [jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)],
        axis=-1,
    )
    return X + noise * jax.random.normal(kn, X.shape), phi
