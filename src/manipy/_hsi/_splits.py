"""Stratified train / test splits of a labelled image."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int


def stratified_split(
    labels: Int[Array, ...],
    *,
    fraction: float | None = None,
    count: int | None = None,
    key: Array,
    background: int = 0,
) -> tuple[Int[Array, " m"], Int[Array, " k"]]:
    """Split the labelled pixels into train and test, per class.

    Each class contributes ``round(fraction * n_c)`` (at least one) or
    ``count`` pixels to the training set; every other labelled pixel is test.
    The ``background`` label belongs to neither. The output sizes depend on
    the data, so this runs eagerly and cannot be jitted.

    Args:
        labels: Ground truth ``(N,)`` or an ``(H, W)`` map (flattened
            row-major, like `image_to_array`).
        fraction: Share of each class to train on, in ``(0, 1)``.
        count: Training pixels per class (all of a class smaller than that).
            Exactly one of ``fraction`` and ``count`` is required.
        key: JAX PRNG key.
        background: The unlabelled / background value, ignored.

    Returns:
        ``(train, test)``: sorted integer indices into the flattened labels.

    Raises:
        ValueError: Unless exactly one of ``fraction`` and ``count`` is given
            and valid.

    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from manipy import hsi
        >>> y = jnp.array([0, 1, 1, 1, 1, 2, 2, 0, 2, 2])
        >>> train, test = hsi.stratified_split(
        ...     y, fraction=0.5, key=jax.random.key(0)
        ... )
        >>> y[train].tolist(), y[test].tolist()
        ([1, 1, 2, 2], [1, 1, 2, 2])
    """
    if (fraction is None) == (count is None):
        raise ValueError("pass exactly one of `fraction` and `count`")
    if fraction is not None and not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1); got {fraction}")
    if count is not None and count < 1:
        raise ValueError(f"count must be at least 1; got {count}")

    labels = jnp.asarray(labels)
    if labels.ndim == 2:
        labels = jnp.asarray(einx.id("h w -> (h w)", labels))
    y = np.asarray(labels)
    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for c in np.unique(y):
        if c == background:
            continue
        idx = np.flatnonzero(y == c)
        n = idx.shape[0]
        n_train = (
            max(1, round(fraction * n)) if fraction is not None else int(count or 0)
        )
        n_train = min(n_train, n)
        perm = np.asarray(jax.random.permutation(jax.random.fold_in(key, int(c)), n))
        chosen = idx[perm]
        train.append(chosen[:n_train])
        test.append(chosen[n_train:])
    empty = np.zeros(0, dtype=np.int32)
    tr = np.sort(np.concatenate(train)) if train else empty
    te = np.sort(np.concatenate(test)) if test else empty
    return jnp.asarray(tr, dtype=jnp.int32), jnp.asarray(te, dtype=jnp.int32)
