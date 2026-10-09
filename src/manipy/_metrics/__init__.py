"""Classification and embedding-quality metrics."""

from __future__ import annotations

from manipy._metrics._classification import (
    average_accuracy,
    cohen_kappa,
    cohen_kappa_variance,
    confusion_matrix,
    overall_accuracy,
    per_class_accuracy,
)
from manipy._metrics._quality import (
    continuity,
    knn_preservation,
    lcmc,
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
