"""Synthetic manifolds and downloadable hyperspectral benchmark scenes."""

from __future__ import annotations

from manipy._datasets._hsi import indian_pines, pavia_university, salinas
from manipy._datasets._synthetic import s_curve, severed_sphere, swiss_roll


__all__ = [
    "indian_pines",
    "pavia_university",
    "s_curve",
    "salinas",
    "severed_sphere",
    "swiss_roll",
]
