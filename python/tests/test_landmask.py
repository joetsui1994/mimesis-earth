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


def test_deterministic(mesh):
    spec = WorldSpec(levels=[3, 3], n_landmasses=2, resolution=4000)
    m1 = build_landmask(mesh, spec, np.random.default_rng(14))
    m2 = build_landmask(mesh, spec, np.random.default_rng(14))
    np.testing.assert_array_equal(m1.land, m2.land)
    np.testing.assert_array_equal(m1.group, m2.group)
    np.testing.assert_array_equal(m1.bridges, m2.bridges)
