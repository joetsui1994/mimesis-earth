# python/tests/test_agglomerate.py
import numpy as np
from mimesis_earth.agglomerate import region_grow


def path_graph(n, w=1.0):
    """0-1-2-...-(n-1) line graph; unit sizes."""
    nbr = {i: [] for i in range(n)}
    for i in range(n - 1):
        nbr[i].append((i + 1, w))
        nbr[i + 1].append((i, w))
    return nbr, np.ones(n)


def test_region_grow_splits_path_in_half():
    nbr, sizes = path_graph(6)
    assign = region_grow(nbr, sizes, np.array([3.0, 3.0]), [0, 5],
                         np.random.default_rng(0))
    assert set(assign.tolist()) == {0, 1}
    assert (assign == 0).sum() == 3 and (assign == 1).sum() == 3
    # contiguous: group 0 is a prefix, group 1 a suffix
    assert assign.tolist() == [0, 0, 0, 1, 1, 1]


def test_region_grow_balances_interior_seeds():
    # interior seeds with room to grow; feed-most-behind should keep the two
    # groups near-equal on a symmetric path. (Endpoint/adjacent seeds are a
    # degenerate 1-D case no region-grower can balance and are not a target.)
    nbr, sizes = path_graph(10)
    assign = region_grow(nbr, sizes, np.array([5.0, 5.0]), [2, 7],
                         np.random.default_rng(1))
    a, b = (assign == 0).sum(), (assign == 1).sum()
    assert abs(a - b) <= 2


def test_region_grow_field_bias_puts_border_on_ridge():
    # high field at item 5 (middle); border should form there (both groups avoid it)
    nbr, sizes = path_graph(11)
    field = np.zeros(11)
    field[5] = 10.0
    assign = region_grow(nbr, sizes, np.array([5.5, 5.5]), [0, 10],
                         np.random.default_rng(2), field=field, lam=3.0)
    # item 5 is a boundary item: it has a neighbor in the other group
    left = assign[4]
    right = assign[6]
    assert left != right  # the ridge splits the two groups


def test_region_grow_deterministic():
    nbr, sizes = path_graph(20)
    a = region_grow(nbr, sizes, np.array([10.0, 10.0]), [0, 19], np.random.default_rng(3))
    b = region_grow(nbr, sizes, np.array([10.0, 10.0]), [0, 19], np.random.default_rng(3))
    assert a.tolist() == b.tolist()


def test_region_grow_straggler_guard_assigns_all():
    # a "T": item 3 hangs off item 1; both seeds far. Every item must be assigned.
    nbr = {0: [(1, 1.0)], 1: [(0, 1.0), (2, 1.0), (3, 1.0)],
           2: [(1, 1.0)], 3: [(1, 1.0)]}
    sizes = np.ones(4)
    assign = region_grow(nbr, sizes, np.array([2.0, 2.0]), [0, 2],
                         np.random.default_rng(4))
    assert (assign >= 0).all()
    assert set(assign.tolist()) == {0, 1}


from mimesis_earth.mesh import build_mesh
from mimesis_earth.agglomerate import build_item_graph, BRIDGE_EPS


def test_build_item_graph_adjacency_and_bridges():
    mesh = build_mesh(2000, np.random.default_rng(10))
    # two parts: northern cap and everything else
    z = mesh.points[:, 2]
    north = np.flatnonzero(z > 0.5)
    rest = np.flatnonzero(z <= 0.5)
    neighbors, sizes = build_item_graph(mesh, [north, rest])
    assert len(sizes) == 2 and sizes[0] == len(north)
    # the two parts touch, so they are neighbors with weight > BRIDGE_EPS
    assert any(j == 1 and w > BRIDGE_EPS for j, w in neighbors[0])
    # a bridge adds a low-weight link between two otherwise-disjoint parts
    a, b = int(north[0]), int(rest[0])
    nb2, _ = build_item_graph(mesh, [np.array([a]), np.array([b])],
                              bridges=np.array([[a, b]]))
    assert nb2[0] == [(1, BRIDGE_EPS)]


from scipy.sparse.csgraph import connected_components
from mimesis_earth.agglomerate import leaf_partition
from mimesis_earth.spec import MIN_ATOMS_PER_LEAF


def test_leaf_partition_covers_and_meets_min():
    mesh = build_mesh(6000, np.random.default_rng(11))
    z = mesh.points[:, 2]
    group = np.flatnonzero(z > 0.2)          # one big cap, connected
    parts = leaf_partition(mesh, group, 20, roughness=0.5, size_variance=0.4,
                           atom_cost=None, rng=np.random.default_rng(12))
    assert len(parts) == 20
    covered = np.sort(np.concatenate(parts))
    np.testing.assert_array_equal(covered, np.sort(group))
    assert all(len(p) >= MIN_ATOMS_PER_LEAF for p in parts)
    # each part is a single connected blob (single island, contiguous)
    for p in parts:
        sub = mesh.adjacency[p][:, p]
        assert connected_components(sub, directed=False)[0] == 1


from mimesis_earth.agglomerate import allocate_group_counts


def test_allocate_group_counts_exact_and_feasible():
    # 3 groups, sizes 1000/500/300; levels [6,5,6]; MIN 8
    group_sizes = np.array([1000.0, 500.0, 300.0])
    C, D = allocate_group_counts(group_sizes, [6, 5, 6])
    assert C.sum() == 6 and (C >= 1).all()
    # D_g = C_g * 5 * 6
    assert (D == C * 30).all()


def test_allocate_group_counts_infeasible_raises():
    # tiny third group cannot host 1 country * 5 * 6 * 8 = 240 atoms
    group_sizes = np.array([5000.0, 5000.0, 100.0])
    import pytest
    with pytest.raises(ValueError, match="too small"):
        allocate_group_counts(group_sizes, [6, 5, 6])
