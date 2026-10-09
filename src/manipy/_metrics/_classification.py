"""Classification metrics from the confusion matrix (HSI conventions).

Labels are non-negative integers. ``n_classes`` defaults to
``max(label) + 1`` read from the data, which needs concrete arrays; pass it
to use these functions under ``jit``.
"""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def confusion_matrix(
    y_true: Int[Array, " n"],
    y_pred: Int[Array, " n"],
    *,
    n_classes: int | None = None,
) -> Float[Array, "k k"]:
    """The confusion matrix ``C``, with ``C[i, j]`` the pixels of true class ``i``
    predicted as class ``j``.

    Args:
        y_true: True labels ``(N,)``, in ``[0, n_classes)``.
        y_pred: Predicted labels ``(N,)``.
        n_classes: Number of classes; default ``max(y_true, y_pred) + 1``.

    Returns:
        Counts ``(K, K)`` as floats.

    Examples:
        >>> import jax.numpy as jnp
        >>> from manipy import metrics
        >>> metrics.confusion_matrix(
        ...     jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 1, 1])
        ... ).tolist()
        [[1.0, 1.0], [0.0, 2.0]]
    """
    y_true, y_pred = jnp.asarray(y_true), jnp.asarray(y_pred)
    if n_classes is None:
        n_classes = int(jnp.maximum(y_true.max(), y_pred.max())) + 1
    t = jax.nn.one_hot(y_true, n_classes)
    p = jax.nn.one_hot(y_pred, n_classes)
    return einx.dot("n i, n j -> i j", t, p)


def overall_accuracy(
    y_true: Int[Array, " n"], y_pred: Int[Array, " n"]
) -> Float[Array, ""]:
    """Overall accuracy (OA): the fraction of correctly classified pixels.

    Examples:
        >>> import jax.numpy as jnp
        >>> from manipy import metrics
        >>> float(
        ...     metrics.overall_accuracy(
        ...         jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 1, 1])
        ...     )
        ... )
        0.75
    """
    return jnp.mean(jnp.asarray(y_true) == jnp.asarray(y_pred))


def per_class_accuracy(
    y_true: Int[Array, " n"],
    y_pred: Int[Array, " n"],
    *,
    n_classes: int | None = None,
) -> Float[Array, " k"]:
    """Producer's accuracy of each class: its recall, ``C[i, i] / sum_j C[i, j]``.

    A class with no true pixels gets ``nan`` (and is skipped by
    `average_accuracy`).

    Args:
        y_true: True labels ``(N,)``.
        y_pred: Predicted labels ``(N,)``.
        n_classes: Number of classes; default read from the data.

    Returns:
        Accuracies ``(K,)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from manipy import metrics
        >>> metrics.per_class_accuracy(
        ...     jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 1, 1])
        ... ).tolist()
        [0.5, 1.0]
    """
    C = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    support = einx.sum("i [j] -> i", C)
    return jnp.where(support > 0, jnp.diag(C) / jnp.maximum(support, 1.0), jnp.nan)


def average_accuracy(
    y_true: Int[Array, " n"],
    y_pred: Int[Array, " n"],
    *,
    n_classes: int | None = None,
) -> Float[Array, ""]:
    """Average accuracy (AA): the mean of `per_class_accuracy` over the classes
    present in ``y_true``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from manipy import metrics
        >>> float(
        ...     metrics.average_accuracy(
        ...         jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 1, 1])
        ...     )
        ... )
        0.75
    """
    return jnp.nanmean(per_class_accuracy(y_true, y_pred, n_classes=n_classes))


def _kappa_terms(C: Float[Array, "k k"]):
    n = jnp.sum(C)
    rows = einx.sum("i [j] -> i", C)
    cols = einx.sum("[i] j -> j", C)
    return n, rows, cols


def cohen_kappa(
    y_true: Int[Array, " n"],
    y_pred: Int[Array, " n"],
    *,
    n_classes: int | None = None,
) -> Float[Array, ""]:
    r"""Cohen's kappa: agreement beyond chance.

    $$\kappa = \frac{p_o - p_e}{1 - p_e},\quad
    p_o = \frac{\operatorname{tr} C}{n},\quad
    p_e = \frac{\sum_i C_{i+} C_{+i}}{n^2}.$$

    ``1`` is perfect agreement and ``0`` is what chance gives. Undefined
    (``nan``) when ``p_e = 1``, i.e. both labelings are one constant class.

    Examples:
        >>> import jax.numpy as jnp
        >>> from manipy import metrics
        >>> float(
        ...     metrics.cohen_kappa(
        ...         jnp.array([0, 0, 1, 1]), jnp.array([0, 1, 1, 1])
        ...     )
        ... )
        0.5
    """
    C = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    n, rows, cols = _kappa_terms(C)
    po = jnp.trace(C) / n
    pe = jnp.dot(rows, cols) / n**2
    return (po - pe) / (1.0 - pe)


def cohen_kappa_variance(
    y_true: Int[Array, " n"],
    y_pred: Int[Array, " n"],
    *,
    n_classes: int | None = None,
) -> Float[Array, ""]:
    r"""Large-sample variance of `cohen_kappa` (Congalton & Green, 2009).

    With $\theta_1 = \sum_i C_{ii}/n$, $\theta_2 = \sum_i C_{i+}C_{+i}/n^2$,
    $\theta_3 = \sum_i C_{ii}(C_{i+} + C_{+i})/n^2$ and
    $\theta_4 = \sum_{ij} C_{ij}(C_{j+} + C_{+i})^2/n^3$,

    $$\operatorname{Var}(\hat\kappa) = \frac1n\Big[
    \frac{\theta_1(1-\theta_1)}{(1-\theta_2)^2}
    + \frac{2(1-\theta_1)(2\theta_1\theta_2-\theta_3)}{(1-\theta_2)^3}
    + \frac{(1-\theta_1)^2(\theta_4-4\theta_2^2)}{(1-\theta_2)^4}\Big].$$

    Two kappas differ significantly when
    $|\kappa_1-\kappa_2|/\sqrt{\operatorname{Var}_1+\operatorname{Var}_2} > 1.96$.

    Examples:
        >>> import jax.numpy as jnp
        >>> from manipy import metrics
        >>> y = jnp.array([0, 0, 1, 1])
        >>> float(metrics.cohen_kappa_variance(y, y))  # perfect agreement
        0.0
    """
    C = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    n, rows, cols = _kappa_terms(C)
    t1 = jnp.trace(C) / n
    t2 = jnp.dot(rows, cols) / n**2
    t3 = jnp.dot(jnp.diag(C), rows + cols) / n**2
    pair = einx.add("j, i -> i j", rows, cols)  # C_{j+} + C_{+i}
    t4 = jnp.sum(C * pair**2) / n**3
    a = t1 * (1 - t1) / (1 - t2) ** 2
    b = 2 * (1 - t1) * (2 * t1 * t2 - t3) / (1 - t2) ** 3
    c = (1 - t1) ** 2 * (t4 - 4 * t2**2) / (1 - t2) ** 4
    return (a + b + c) / n
