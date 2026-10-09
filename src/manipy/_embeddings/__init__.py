"""Dimensionality-reduction methods that are not spectral graph embeddings."""

from manipy._embeddings._diffusion_maps import DiffusionMaps
from manipy._embeddings._isomap import Isomap
from manipy._embeddings._lle import LocallyLinearEmbedding


__all__ = ["DiffusionMaps", "Isomap", "LocallyLinearEmbedding"]
