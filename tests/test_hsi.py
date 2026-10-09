"""HSI helpers: row-major conversion, pixel grids, potentials, splits."""

from __future__ import annotations

import einx
import jax
import jax.numpy as jnp
import kernellib as kl
import numpy as np
import pytest

from manipy import hsi


def test_image_roundtrip_and_row_major_order() -> None:
    cube = jax.random.normal(jax.random.key(0), (4, 3, 5))
    X, shape = hsi.image_to_array(cube)
    assert X.shape == (12, 5) and shape == (4, 3)
    np.testing.assert_array_equal(X[1 * 3 + 2], cube[1, 2])  # row r, col c -> r*W+c
    np.testing.assert_array_equal(hsi.array_to_image(X, shape), cube)


def test_label_map_roundtrip() -> None:
    gt = einx.id("(h w) -> h w", jnp.arange(6), h=2, w=3)
    y, shape = hsi.image_to_array(gt)
    assert y.shape == (6,)
    np.testing.assert_array_equal(hsi.array_to_image(y, shape), gt)


def test_conversion_errors() -> None:
    with pytest.raises(ValueError, match="must be"):
        hsi.image_to_array(jnp.zeros(4))
    with pytest.raises(ValueError, match="does not fold"):
        hsi.array_to_image(jnp.zeros(5), (2, 3))


def test_pixel_coordinates_match_image_order() -> None:
    cube = jax.random.normal(jax.random.key(1), (3, 4, 2))
    X, shape = hsi.image_to_array(cube)
    rc = hsi.pixel_coordinates(shape)
    np.testing.assert_array_equal(X, cube[rc[:, 0], rc[:, 1]])


@pytest.mark.parametrize(("connectivity", "edges"), [("face", 17), ("full", 29)])
def test_pixel_graph_is_the_grid_graph(connectivity: str, edges: int) -> None:
    g = hsi.pixel_graph((3, 4), connectivity=connectivity)
    assert isinstance(g, kl.GridGraph)
    assert g.n_nodes == 12
    assert g.topology.n_edges == edges


def test_spatial_spectral_potential_matches_kernellib() -> None:
    cube = jax.random.normal(jax.random.key(2), (5, 4, 3))
    V = hsi.spatial_spectral_potential_image(cube, bandwidth=1.5)
    X, shape = hsi.image_to_array(cube)
    ref = kl.spatial_spectral_graph(X, kl.grid_graph(shape), bandwidth=1.5)
    z = jax.random.normal(jax.random.key(3), (20,))
    np.testing.assert_allclose(V.mv(z), ref.laplacian_operator().mv(z))
    np.testing.assert_allclose(V.mv(jnp.ones(20)), 0.0, atol=1e-10)


def test_partial_label_potential_links_the_labelled_pixels() -> None:
    # The MATLAB bug linked the first n_i samples; pixels 3 and 5 here.
    y = jnp.array([0, 0, 0, 2, 0, 2])
    V = hsi.partial_label_potential(y)
    expected = np.zeros((6, 6))
    expected[[3, 5], [3, 5]] = 1.0
    expected[3, 5] = expected[5, 3] = -1.0
    np.testing.assert_array_equal(V, expected)
    np.testing.assert_array_equal(
        hsi.partial_label_potential(einx.id("(h w) -> h w", y, h=2, w=3)), V
    )


def test_stratified_split_fraction() -> None:
    y = jnp.array([0] * 10 + [1] * 20 + [2] * 40 + [3] * 3)
    train, test = hsi.stratified_split(y, fraction=0.25, key=jax.random.key(0))
    assert set(train.tolist()).isdisjoint(test.tolist())
    assert jnp.all(y[train] != 0) and jnp.all(y[test] != 0)  # background ignored
    assert len(train) + len(test) == 63
    counts = {c: int(jnp.sum(y[train] == c)) for c in (1, 2, 3)}
    assert counts == {1: 5, 2: 10, 3: 1}  # at least one per class
    assert jnp.all(jnp.diff(train) > 0) and jnp.all(jnp.diff(test) > 0)


def test_stratified_split_count_and_reproducible() -> None:
    y = jnp.array([1] * 5 + [2] * 2 + [0] * 3)
    a = hsi.stratified_split(y, count=3, key=jax.random.key(0))
    b = hsi.stratified_split(y, count=3, key=jax.random.key(0))
    np.testing.assert_array_equal(a[0], b[0])
    assert int(jnp.sum(y[a[0]] == 1)) == 3
    assert int(jnp.sum(y[a[0]] == 2)) == 2  # the whole small class
    assert int(jnp.sum(y[a[1]] == 2)) == 0


def test_stratified_split_custom_background_and_2d() -> None:
    y = jnp.array([[-1, 1, 1], [1, 2, 2]])
    train, test = hsi.stratified_split(
        y, fraction=0.5, key=jax.random.key(0), background=-1
    )
    assert 0 not in train.tolist() + test.tolist()
    assert len(train) + len(test) == 5


def test_stratified_split_argument_errors() -> None:
    y = jnp.array([1, 1, 2, 2])
    key = jax.random.key(0)
    with pytest.raises(ValueError, match="exactly one"):
        hsi.stratified_split(y, key=key)
    with pytest.raises(ValueError, match="exactly one"):
        hsi.stratified_split(y, fraction=0.5, count=1, key=key)
    with pytest.raises(ValueError, match="fraction"):
        hsi.stratified_split(y, fraction=1.5, key=key)
    with pytest.raises(ValueError, match="count"):
        hsi.stratified_split(y, count=0, key=key)
