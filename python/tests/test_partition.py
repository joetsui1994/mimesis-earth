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


def test_partition_handles_disconnected_atoms(mesh):
    # three disjoint clusters, no bridges: chord-distance fallback must
    # assign every atom and keep parts non-empty
    z = mesh.points[:, 2]
    atom_idx = np.concatenate([
        np.flatnonzero(z > 0.9),
        np.flatnonzero(z < -0.9),
        np.flatnonzero(np.abs(z) < 0.05),
    ])
    parts = partition_atoms(mesh, atom_idx, 2, None, 0.3, np.random.default_rng(30))
    combined = np.sort(np.concatenate(parts))
    np.testing.assert_array_equal(combined, np.sort(atom_idx))
    assert all(len(p) > 0 for p in parts)


def test_partition_contiguous_across_bridges(mesh):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    north = np.flatnonzero(mesh.points[:, 2] > 0.85)
    south = np.flatnonzero(mesh.points[:, 2] < -0.85)
    atom_idx = np.concatenate([north, south])
    bridges = np.array([[north[0], south[0]]])
    parts = partition_atoms(mesh, atom_idx, 3, bridges, 0.5, np.random.default_rng(31))
    assert sum(len(p) for p in parts) == len(atom_idx)
    for part in parts:
        pos = {int(a): i for i, a in enumerate(part)}
        rows, cols = [], []
        for a, b in np.vstack([mesh.edges, bridges]):
            if int(a) in pos and int(b) in pos:
                rows.append(pos[int(a)])
                cols.append(pos[int(b)])
        adj = coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(part), len(part))
        )
        n_comp, _ = connected_components(adj, directed=False)
        assert n_comp == 1


def test_allocate_counts_rejects_zero_weights():
    with pytest.raises(ValueError):
        allocate_counts(5, np.zeros(5))


def test_partition_balance():
    # Lloyd refinement must keep part sizes reasonably even
    big = build_mesh(6000, np.random.default_rng(40))
    atom_idx = np.arange(6000)
    for seed in (50, 51, 52):
        parts = partition_atoms(
            big, atom_idx, 8, None, 0.4, np.random.default_rng(seed)
        )
        sizes = np.array([len(p) for p in parts])
        cv = sizes.std() / sizes.mean()
        assert cv < 0.3, f"seed {seed}: sizes {sizes.tolist()} cv {cv:.2f}"


def test_redistribute_counts():
    from mimesis_earth.partition import redistribute_counts

    out = redistribute_counts(np.array([5, 5, 5]), np.array([3, 10, 10]))
    assert out.sum() == 15
    assert out[0] == 3
    assert (out <= np.array([3, 10, 10])).all()
    # no-op when everything fits
    np.testing.assert_array_equal(
        redistribute_counts(np.array([2, 2]), np.array([5, 5])), [2, 2]
    )
