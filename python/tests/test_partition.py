import numpy as np
import pytest

from mimesis_earth.mesh import build_mesh
from mimesis_earth.partition import (
    allocate_counts,
    child_counts,
    partition_atoms,
    pick_seeds,
)


@pytest.fixture(scope="module")
def mesh():
    return build_mesh(2000, np.random.default_rng(20))


def test_pick_seeds_distinct(mesh):
    idx = np.arange(500)
    seeds = pick_seeds(mesh.points[idx], 8, np.random.default_rng(21))
    assert len(seeds) == 8
    assert len(set(seeds.tolist())) == 8


def test_partition_covers_exactly_once(mesh):
    atom_idx = np.arange(len(mesh.points))  # whole sphere as one "unit"
    parts = partition_atoms(
        mesh, atom_idx, 6, extra_edges=None, roughness=0.4,
        rng=np.random.default_rng(22),
    )
    assert len(parts) == 6
    combined = np.sort(np.concatenate(parts))
    np.testing.assert_array_equal(combined, atom_idx)
    assert all(len(p) > 0 for p in parts)


def test_partition_parts_are_contiguous(mesh):
    from scipy.sparse.csgraph import connected_components

    atom_idx = np.arange(len(mesh.points))
    parts = partition_atoms(
        mesh, atom_idx, 5, extra_edges=None, roughness=0.0,
        rng=np.random.default_rng(23),
    )
    for p in parts:
        sub = mesh.adjacency[p][:, p]
        n_comp, _ = connected_components(sub, directed=False)
        assert n_comp == 1


def test_partition_deterministic(mesh):
    atom_idx = np.arange(1000)
    a = partition_atoms(mesh, atom_idx, 4, None, 0.5, np.random.default_rng(24))
    b = partition_atoms(mesh, atom_idx, 4, None, 0.5, np.random.default_rng(24))
    for pa, pb in zip(a, b):
        np.testing.assert_array_equal(pa, pb)


def test_child_counts_exact_when_variance_zero():
    counts = child_counts(6, 10, 0.0, np.random.default_rng(25))
    assert (counts == 6).all()


def test_child_counts_varies_and_positive():
    counts = child_counts(6, 200, 0.5, np.random.default_rng(26))
    assert counts.min() >= 1
    assert counts.std() > 0


def test_allocate_counts():
    out = allocate_counts(8, np.array([100.0, 50.0, 10.0]))
    assert out.sum() == 8
    assert (out >= 1).all()
    assert out[0] >= out[1] >= out[2]
    # every group gets one even when tiny
    out = allocate_counts(3, np.array([1000.0, 1.0, 1.0]))
    assert out.tolist() == [1, 1, 1]
