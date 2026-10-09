"""Synthetic manifolds (fast) and the HSI downloaders (integration)."""

from __future__ import annotations

import socket

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from manipy import datasets


@pytest.mark.parametrize("fn", [datasets.swiss_roll, datasets.s_curve])
def test_sheet_manifolds(fn) -> None:
    X, t = fn(50, key=jax.random.key(0))
    assert X.shape == (50, 3) and t.shape == (50,)
    X2, _ = fn(50, key=jax.random.key(0))
    np.testing.assert_array_equal(X, X2)
    Xn, _ = fn(50, key=jax.random.key(0), noise=0.5)
    assert float(jnp.abs(Xn - X).max()) > 0.0


def test_swiss_roll_geometry() -> None:
    X, t = datasets.swiss_roll(200, key=jax.random.key(0))
    np.testing.assert_allclose(jnp.hypot(X[:, 0], X[:, 2]), t, rtol=1e-6)
    assert float(t.min()) >= 1.5 * np.pi and float(t.max()) <= 4.5 * np.pi


def test_s_curve_geometry() -> None:
    X, t = datasets.s_curve(200, key=jax.random.key(0))
    np.testing.assert_allclose(X[:, 0], jnp.sin(t), atol=1e-6)
    assert float(jnp.abs(t).max()) <= 1.5 * np.pi


def test_severed_sphere_is_a_banded_unit_sphere() -> None:
    X, phi = datasets.severed_sphere(500, key=jax.random.key(0))
    np.testing.assert_allclose(jnp.sqrt(einx.sum("n [d] -> n", X**2)), 1.0, atol=1e-6)
    assert float(jnp.abs(X[:, 2]).max()) <= np.cos(np.pi / 8) + 1e-6  # poles severed
    assert float(phi.max()) <= 2 * np.pi - 0.55
    assert X.shape == (500, 3)  # exactly n_samples, unlike the 2016 code


def _seed_cache(tmp_path, name: str, variable: str, array) -> None:
    from scipy.io import savemat

    savemat(tmp_path / name, {variable: np.asarray(array)})


def test_hsi_loaders_read_a_seeded_cache(tmp_path) -> None:
    pytest.importorskip("pooch")
    pytest.importorskip("scipy")
    cube = np.random.default_rng(0).uniform(size=(4, 3, 5))
    gt = np.arange(12).reshape(4, 3)
    for fname, var, arr in [
        ("Indian_pines_corrected.mat", "indian_pines_corrected", cube),
        ("Indian_pines_gt.mat", "indian_pines_gt", gt),
        ("PaviaU.mat", "paviaU", cube),
        ("PaviaU_gt.mat", "paviaU_gt", gt),
        ("Salinas_corrected.mat", "salinas_corrected", cube),
        ("Salinas_gt.mat", "salinas_gt", gt),
    ]:
        _seed_cache(tmp_path, fname, var, arr)
    for loader in (datasets.indian_pines, datasets.pavia_university, datasets.salinas):
        image, labels = loader(tmp_path)
        assert image.shape == (4, 3, 5) and image.dtype == jnp.float32
        assert labels.shape == (4, 3) and labels.dtype == jnp.int32
        np.testing.assert_array_equal(labels, gt)


def _online() -> bool:
    try:
        socket.create_connection(("www.ehu.eus", 443), timeout=3).close()
    except OSError:
        return False
    return True


@pytest.mark.integration
@pytest.mark.parametrize(
    ("loader", "bands", "size"),
    [
        (datasets.indian_pines, 200, (145, 145)),
        (datasets.pavia_university, 103, (610, 340)),
        (datasets.salinas, 204, (512, 217)),
    ],
)
def test_hsi_download(loader, bands: int, size: tuple[int, int], tmp_path) -> None:
    pytest.importorskip("pooch")
    pytest.importorskip("scipy")
    if not _online():
        pytest.skip("offline")
    import requests

    try:
        image, gt = loader(tmp_path)
    except requests.exceptions.RequestException as err:  # mirror down / blocking
        pytest.skip(f"EHU mirror unavailable: {err}")
    assert image.shape == (*size, bands)
    assert gt.shape == size and int(gt.min()) == 0
