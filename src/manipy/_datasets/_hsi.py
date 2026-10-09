"""Downloadable hyperspectral benchmark scenes (extra ``manipy-jax[data]``).

``pooch`` and ``scipy.io`` are imported inside the loaders, so nothing here
loads on ``import manipy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


_BASE_URL = "https://www.ehu.eus/ccwintco/uploads"


@dataclass(frozen=True)
class _Scene:
    """Where one scene lives: a (path, sha256, .mat variable) per file."""

    image: tuple[str, str | None, str]
    ground_truth: tuple[str, str | None, str]


# The SHA-256 slots are ``None`` until the digests are pinned: the EHU server
# refused scripted downloads while this was written, so the files could not
# be hashed. ``pooch`` downloads without verification while a digest is
# ``None``; fill them in with ``pooch.file_hash`` on a verified download.
_SCENES: dict[str, _Scene] = {
    "indian_pines": _Scene(
        ("6/67/Indian_pines_corrected.mat", None, "indian_pines_corrected"),
        ("c/c4/Indian_pines_gt.mat", None, "indian_pines_gt"),
    ),
    "pavia_university": _Scene(
        ("9/91/PaviaU.mat", None, "paviaU"),
        ("5/50/PaviaU_gt.mat", None, "paviaU_gt"),
    ),
    "salinas": _Scene(
        ("f/f1/Salinas_corrected.mat", None, "salinas_corrected"),
        ("f/fa/Salinas_gt.mat", None, "salinas_gt"),
    ),
}


def _load_mat(path: str | Path, variable: str) -> Array:
    from scipy.io import loadmat  # extra `data`

    return jnp.asarray(loadmat(str(path))[variable])


def _load(
    name: str, data_dir: str | Path | None
) -> tuple[Float[Array, "h w b"], Int[Array, "h w"]]:
    try:
        import pooch  # extra `data`
    except ImportError as err:
        raise ImportError(
            "downloading datasets needs the `data` extra: "
            "pip install 'manipy-jax[data]'"
        ) from err

    scene = _SCENES[name]
    root = pooch.os_cache("manipy") if data_dir is None else Path(data_dir)
    arrays = []
    for rel, digest, variable in (scene.image, scene.ground_truth):
        local = pooch.retrieve(
            url=f"{_BASE_URL}/{rel}",
            known_hash=None if digest is None else f"sha256:{digest}",
            fname=rel.rsplit("/", 1)[-1],
            path=root,
        )
        arrays.append(_load_mat(local, variable))
    image, gt = arrays
    return image.astype(jnp.float32), gt.astype(jnp.int32)


def indian_pines(
    data_dir: str | Path | None = None,
) -> tuple[Float[Array, "145 145 200"], Int[Array, "145 145"]]:
    """AVIRIS Indian Pines: 145 x 145 pixels, 200 bands, 16 crop classes.

    The water-absorption-corrected cube (200 of the 220 bands). Downloaded
    once with ``pooch`` from the Grupo de Inteligencia Computacional (EHU)
    mirror and cached.

    Args:
        data_dir: Cache directory; default the user cache (``pooch.os_cache``).

    Returns:
        ``(image, ground_truth)``: a float32 cube ``(145, 145, 200)`` and an
        int32 label map ``(145, 145)`` where ``0`` is unlabelled.

    Raises:
        ImportError: Without the ``data`` extra (``pip install 'manipy-jax[data]'``).

    Examples:
        The Indian Pines workflow of the spec: spatial-spectral
        Schrödinger eigenmaps, an SVM on 10 % of the labels, then OA / AA /
        kappa. It downloads the scene, so it is shown, not run.

        >>> import einx, jax, kernellib as kl, manipy  # doctest: +SKIP
        >>> from sklearn.svm import SVC  # doctest: +SKIP
        >>> cube, gt = manipy.datasets.indian_pines()  # doctest: +SKIP
        >>> X, shape = manipy.hsi.image_to_array(cube)  # doctest: +SKIP
        >>> V = manipy.hsi.spatial_spectral_potential_image(cube)  # doctest: +SKIP
        >>> Y = (
        ...     kl.SchrodingerEigenmaps(n_components=30, n_neighbors=20, alpha=17.8)
        ...     .fit(X, V)
        ...     .embedding
        ... )  # doctest: +SKIP
        >>> y = einx.rearrange("h w -> (h w)", gt)  # doctest: +SKIP
        >>> train, test = manipy.hsi.stratified_split(
        ...     y, fraction=0.1, key=jax.random.key(0)
        ... )  # doctest: +SKIP
        >>> pred = SVC().fit(Y[train], y[train]).predict(Y[test])  # doctest: +SKIP
        >>> oa = manipy.metrics.overall_accuracy(y[test], pred)  # doctest: +SKIP
        >>> aa = manipy.metrics.average_accuracy(y[test], pred)  # doctest: +SKIP
        >>> kappa = manipy.metrics.cohen_kappa(y[test], pred)  # doctest: +SKIP
    """
    return _load("indian_pines", data_dir)


def pavia_university(
    data_dir: str | Path | None = None,
) -> tuple[Float[Array, "610 340 103"], Int[Array, "610 340"]]:
    """ROSIS Pavia University: 610 x 340 pixels, 103 bands, 9 urban classes.

    Downloaded once with ``pooch`` from the EHU mirror and cached.

    Args:
        data_dir: Cache directory; default the user cache (``pooch.os_cache``).

    Returns:
        ``(image, ground_truth)``: a float32 cube ``(610, 340, 103)`` and an
        int32 label map ``(610, 340)`` where ``0`` is unlabelled.

    Raises:
        ImportError: Without the ``data`` extra.

    Examples:
        >>> from manipy import datasets  # doctest: +SKIP
        >>> image, gt = datasets.pavia_university()  # doctest: +SKIP
        >>> image.shape  # doctest: +SKIP
        (610, 340, 103)
    """
    return _load("pavia_university", data_dir)


def salinas(
    data_dir: str | Path | None = None,
) -> tuple[Float[Array, "512 217 204"], Int[Array, "512 217"]]:
    """AVIRIS Salinas: 512 x 217 pixels, 204 bands, 16 crop classes.

    The water-absorption-corrected cube. Downloaded once with ``pooch`` from
    the EHU mirror and cached.

    Args:
        data_dir: Cache directory; default the user cache (``pooch.os_cache``).

    Returns:
        ``(image, ground_truth)``: a float32 cube ``(512, 217, 204)`` and an
        int32 label map ``(512, 217)`` where ``0`` is unlabelled.

    Raises:
        ImportError: Without the ``data`` extra.

    Examples:
        >>> from manipy import datasets  # doctest: +SKIP
        >>> image, gt = datasets.salinas()  # doctest: +SKIP
        >>> gt.shape  # doctest: +SKIP
        (512, 217)
    """
    return _load("salinas", data_dir)
