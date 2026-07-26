import numpy as np

from mimesis_earth.attributes import population_density, round_preserving_sum
from mimesis_earth.mesh import build_mesh


def test_density_positive_and_deterministic():
    mesh = build_mesh(2000, np.random.default_rng(50))
    land_idx = np.arange(600)
    a = population_density(mesh, land_idx, np.random.default_rng(51))
    b = population_density(mesh, land_idx, np.random.default_rng(51))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (600,)
    assert (a > 0).all()


def test_density_spatially_varied():
    mesh = build_mesh(2000, np.random.default_rng(52))
    land_idx = np.arange(800)
    d = population_density(mesh, land_idx, np.random.default_rng(53))
    # cities exist: the densest atom is much denser than the median
    assert d.max() / np.median(d) > 3.0


def test_round_preserving_sum_exact():
    rng = np.random.default_rng(54)
    x = rng.uniform(0.1, 10.0, size=1000)
    out = round_preserving_sum(x, 1_000_000)
    assert out.sum() == 1_000_000
    assert (out >= 0).all()
    # proportions roughly preserved
    big, small = x.argmax(), x.argmin()
    assert out[big] > out[small]


def test_round_preserving_sum_rejects_zero_weights():
    import pytest

    with pytest.raises(ValueError, match="positive sum"):
        round_preserving_sum(np.zeros(10), 100)


def test_round_preserving_sum_total_smaller_than_n():
    # small worlds: fewer population units than leaves -> zeros are fine,
    # sum stays exact
    rng = np.random.default_rng(55)
    x = rng.uniform(0.1, 10.0, size=500)
    out = round_preserving_sum(x, 100)
    assert out.sum() == 100
    assert (out >= 0).all()


def test_density_noncontiguous_land_idx():
    # Task 10 passes arbitrary land atom subsets, not ranges
    mesh = build_mesh(2000, np.random.default_rng(56))
    rng = np.random.default_rng(57)
    land_idx = np.sort(rng.choice(2000, size=700, replace=False))
    d = population_density(mesh, land_idx, np.random.default_rng(58))
    assert d.shape == (700,)
    assert (d > 0).all()
