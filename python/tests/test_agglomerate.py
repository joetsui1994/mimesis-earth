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


def test_region_grow_field_bias_avoids_ridge_items():
    # Diamond: seeds 0 and 3; two middle items 1 (low field) and 2 (high field),
    # each touching both seeds. On a symmetric graph balance alone can't decide
    # which middle item goes where -- the field must. With the bias, the growing
    # group eats the LOW-field item and the HIGH-field item lands on the border
    # (the other group), and this must hold for ANY rng (field dominates the
    # tiny tie-break). A no-op field would make the split rng-dependent.
    nbr = {0: [(1, 1.0), (2, 1.0)], 1: [(0, 1.0), (3, 1.0)],
           2: [(0, 1.0), (3, 1.0)], 3: [(1, 1.0), (2, 1.0)]}
    sizes = np.ones(4)
    field = np.array([0.0, 0.0, 10.0, 0.0])  # item 2 is the ridge
    for trial in range(5):
        assign = region_grow(nbr, sizes, np.array([2.0, 2.0]), [0, 3],
                             np.random.default_rng(trial), field=field, lam=3.0)
        assert assign[1] == assign[0]   # low-field item absorbed by seed 0's group
        assert assign[2] == assign[3]   # high-field item pushed onto the border


def test_region_grow_deterministic():
    nbr, sizes = path_graph(20)
    a = region_grow(nbr, sizes, np.array([10.0, 10.0]), [0, 19], np.random.default_rng(3))
    b = region_grow(nbr, sizes, np.array([10.0, 10.0]), [0, 19], np.random.default_rng(3))
    assert a.tolist() == b.tolist()


def test_region_grow_raises_on_disconnected_graph():
    # The straggler guard's real job: an item with no path to any seed cannot be
    # assigned, so region_grow must RAISE rather than silently drop it (which
    # would violate the contiguity/coverage invariant). Here {0,1} holds the only
    # seed and {2,3} is a separate component.
    import pytest
    nbr = {0: [(1, 1.0)], 1: [(0, 1.0)], 2: [(3, 1.0)], 3: [(2, 1.0)]}
    sizes = np.ones(4)
    with pytest.raises(RuntimeError):
        region_grow(nbr, sizes, np.array([2.0]), [0], np.random.default_rng(4))


def test_region_grow_covers_connected_graph():
    # sanity: on a connected graph every item is assigned to a valid group
    nbr, sizes = path_graph(8)
    assign = region_grow(nbr, sizes, np.array([4.0, 4.0]), [0, 7],
                         np.random.default_rng(4))
    assert (assign >= 0).all() and set(assign.tolist()) == {0, 1}


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


from mimesis_earth.spec import WorldSpec
from mimesis_earth.mesh import build_mesh as _bm
from mimesis_earth.landmask import build_landmask
from mimesis_earth.agglomerate import partition_world


def _small_world_inputs(seed=0):
    spec = WorldSpec(n_landmasses=2, levels=[2, 3, 3], resolution=8000,
                     land_fraction=0.4, seed=seed)
    rng = np.random.default_rng(seed)
    mesh = _bm(spec.resolution, rng)
    mask = build_landmask(mesh, spec, rng)
    grow = np.zeros(len(mesh.points))
    return spec, mesh, mask, grow, rng


def test_partition_world_shape_and_nesting():
    spec, mesh, mask, grow, rng = _small_world_inputs()
    level_nodes = partition_world(mesh, mask, spec, atom_cost=None,
                                  grow_field=grow, rng=rng)
    assert len(level_nodes) == 3
    assert len(level_nodes[0]) == 2          # levels[0] countries total
    assert len(level_nodes[1]) == 2 * 3      # provinces total
    assert len(level_nodes[2]) == 2 * 3 * 3  # districts total
    # every district's atoms are non-empty and disjoint; union = all land
    all_atoms = np.sort(np.concatenate([n["atoms"] for n in level_nodes[2]]))
    land = np.sort(np.flatnonzero(mask.land))
    np.testing.assert_array_equal(all_atoms, land)
    # parent indices are valid and children tile parents
    for lvl in (1, 2):
        for node in level_nodes[lvl]:
            assert 0 <= node["parent"] < len(level_nodes[lvl - 1])
    # level-0 nodes carry landmass id
    assert all(n["landmass"] is not None for n in level_nodes[0])
