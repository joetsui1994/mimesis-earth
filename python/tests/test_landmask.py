import numpy as np
import pytest

from mimesis_earth.landmask import build_landmask
from mimesis_earth.mesh import build_mesh
from mimesis_earth.spec import WorldSpec


@pytest.fixture(scope="module")
def mesh():
    return build_mesh(4000, np.random.default_rng(10))


def test_land_fraction_respected(mesh):
    spec = WorldSpec(levels=[3, 3], n_landmasses=3, land_fraction=0.3, resolution=4000)
    mask = build_landmask(mesh, spec, np.random.default_rng(11))
    frac = mask.land.mean()
    assert 0.25 < frac < 0.35


def test_every_landmass_group_nonempty(mesh):
    spec = WorldSpec(levels=[4, 3], n_landmasses=4, resolution=4000)
    mask = build_landmask(mesh, spec, np.random.default_rng(12))
    groups = set(mask.group[mask.land].tolist())
    assert groups == set(range(4))
    # sea atoms have group -1
    assert (mask.group[~mask.land] == -1).all()


def test_bridges_connect_within_groups(mesh):
    spec = WorldSpec(
        levels=[3, 3], n_landmasses=3, coast_ruggedness=1.0, resolution=4000
    )
    mask = build_landmask(mesh, spec, np.random.default_rng(13))
    # every bridge joins two land atoms of the same group
    for a, b in mask.bridges:
        assert mask.land[a] and mask.land[b]
        assert mask.group[a] == mask.group[b]


def test_many_landmasses_low_spread(mesh):
    # regression: kappa must scale with n_landmasses or this fails structurally
    spec = WorldSpec(
        levels=[32], n_landmasses=32, spread=0.0, land_fraction=0.3,
        resolution=4000,
    )
    for seed in range(100, 105):
        mask = build_landmask(mesh, spec, np.random.default_rng(seed))
        assert set(mask.group[mask.land].tolist()) == set(range(32))


def test_many_landmasses_low_land_fraction():
    # regression: global quantile cut must never erase whole landmass groups
    mesh8k = build_mesh(8000, np.random.default_rng(15))
    for spread in (0.0, 0.5, 1.0):
        spec = WorldSpec(
            levels=[63], n_landmasses=63, spread=spread, land_fraction=0.1,
            resolution=8000,
        )
        for seed in (200, 201, 202):
            mask = build_landmask(mesh8k, spec, np.random.default_rng(seed))
            assert set(mask.group[mask.land].tolist()) == set(range(63))
            assert abs(mask.land.mean() - 0.1) < 0.01


def test_deterministic(mesh):
    spec = WorldSpec(levels=[3, 3], n_landmasses=2, resolution=4000)
    m1 = build_landmask(mesh, spec, np.random.default_rng(14))
    m2 = build_landmask(mesh, spec, np.random.default_rng(14))
    np.testing.assert_array_equal(m1.land, m2.land)
    np.testing.assert_array_equal(m1.group, m2.group)
    np.testing.assert_array_equal(m1.bridges, m2.bridges)


def test_land_is_high_ground(mesh):
    spec = WorldSpec(levels=[3, 3], n_landmasses=3, resolution=4000)
    mask = build_landmask(mesh, spec, np.random.default_rng(60))
    assert mask.elevation.shape == (len(mesh.points),)
    # land sits above sea: mean land elevation > mean sea elevation
    assert mask.elevation[mask.land].mean() > mask.elevation[~mask.land].mean() + 0.5
