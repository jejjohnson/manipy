"""Classification and embedding-quality metrics.

The public view of ``manipy._metrics``.
"""

from __future__ import annotations

from manipy._metrics import (
    average_accuracy,
    cohen_kappa,
    cohen_kappa_variance,
    confusion_matrix,
    continuity,
    knn_preservation,
    lcmc,
    overall_accuracy,
    per_class_accuracy,
    trustworthiness,
)


__all__ = [
    "average_accuracy",
    "cohen_kappa",
    "cohen_kappa_variance",
    "confusion_matrix",
    "continuity",
    "knn_preservation",
    "lcmc",
    "overall_accuracy",
    "per_class_accuracy",
    "trustworthiness",
]
