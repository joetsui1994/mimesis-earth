import numpy as np
import pytest

from mimesis_earth.mesh import build_mesh
from mimesis_earth.partition import (
    allocate_counts,
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


def test_coupled_counts_exact_total_and_min_one():
    from mimesis_earth.partition import coupled_counts

    sizes = np.array([1000.0, 100.0, 10.0])
    for variance in (0.0, 0.5, 1.0, 2.0):
        out = coupled_counts(30, sizes, 0.7, variance, np.random.default_rng(70))
        assert out.sum() == 30
        assert (out >= 1).all()


def test_coupled_counts_coupling_behavior():
    from mimesis_earth.partition import coupled_counts

    sizes = np.array([900.0, 90.0, 10.0])
    flat = coupled_counts(30, sizes, 0.0, 0.0, np.random.default_rng(71))
    prop = coupled_counts(30, sizes, 1.0, 0.0, np.random.default_rng(71))
    assert flat.tolist() == [10, 10, 10]
    assert prop[0] > 20 and prop[2] == 1


def test_coupled_counts_deterministic():
    from mimesis_earth.partition import coupled_counts

    sizes = np.array([500.0, 300.0, 200.0])
    a = coupled_counts(24, sizes, 0.7, 0.8, np.random.default_rng(72))
    b = coupled_counts(24, sizes, 0.7, 0.8, np.random.default_rng(72))
    np.testing.assert_array_equal(a, b)


def test_honor_minimums():
    from mimesis_earth.partition import honor_minimums

    counts = np.array([1, 8, 1])
    minimums = np.array([3, 1, 1])
    out = honor_minimums(counts, minimums)
    assert out.sum() == 10
    assert out[0] == 3
    # shortfall tolerated when donors run dry
    out2 = honor_minimums(np.array([1, 1]), np.array([5, 5]))
    assert out2.sum() == 2


def test_partition_weighted_sizes():
    from scipy.sparse.csgraph import connected_components

    big = build_mesh(4000, np.random.default_rng(60))
    atom_idx = np.arange(4000)
    cvs = {}
    for sv in (0.0, 0.8):
        vals = []
        for seed in (61, 62, 63):
            parts = partition_atoms(
                big, atom_idx, 8, None, 0.4, np.random.default_rng(seed),
                size_variance=sv,
            )
            sizes = np.array([len(p) for p in parts])
            assert sizes.sum() == 4000
            assert sizes.min() > 0
            for p in parts:
                sub = big.adjacency[p][:, p]
                n_comp, _ = connected_components(sub, directed=False)
                assert n_comp == 1
            vals.append(sizes.std() / sizes.mean())
        cvs[sv] = float(np.mean(vals))
    assert cvs[0.0] < 0.15
    assert cvs[0.8] > 0.3


def test_partition_weighted_deterministic(mesh):
    from scipy.sparse.csgraph import connected_components

    atom_idx = np.arange(1500)
    a = partition_atoms(mesh, atom_idx, 5, None, 0.5,
                        np.random.default_rng(64), size_variance=0.7)
    b = partition_atoms(mesh, atom_idx, 5, None, 0.5,
                        np.random.default_rng(64), size_variance=0.7)
    for pa, pb in zip(a, b):
        np.testing.assert_array_equal(pa, pb)
    for p in a:
        sub = mesh.adjacency[p][:, p]
        n_comp, _ = connected_components(sub, directed=False)
        assert n_comp == 1


def test_partition_weighted_contiguity_fuzz():
    # NOTE: deviates from the reviewer's literal atom_idx sampling (uniform
    # random choice over all 2000 atoms), which produced genuinely
    # disconnected inputs -- a random subset of a sparse mesh graph fractures
    # into isolated vertices, so no partitioner could keep every part
    # contiguous. Per the reviewer's own fallback guidance, atom_idx is
    # instead a BFS-order prefix from a random start atom: any prefix of a
    # BFS visiting order is a connected induced subgraph (each newly visited
    # node is discovered via an edge from an already-visited node), so this
    # guarantees a connected input while keeping randomized size/shape.
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    big = build_mesh(2000, np.random.default_rng(65))
    for sv in (0.4, 1.0):
        for seed in range(66, 74):
            rng = np.random.default_rng(seed)
            n = int(rng.integers(60, 2000))
            k = int(rng.integers(2, max(3, n // 30)))
            start = int(np.random.default_rng(seed + 100).integers(2000))
            order = breadth_first_order(
                big.adjacency, start, directed=False, return_predecessors=False
            )
            atom_idx = np.sort(order[:n])
            parts = partition_atoms(
                big, atom_idx, k, None, 0.3, np.random.default_rng(seed + 200),
                size_variance=sv,
            )
            assert sum(len(p) for p in parts) == len(atom_idx)
            for p in parts:
                if len(p) < 2:
                    continue
                sub = big.adjacency[p][:, p]
                n_comp, _ = connected_components(sub, directed=False)
                assert n_comp == 1, (sv, seed, len(p))


def test_plan_islands_single_component(mesh):
    from mimesis_earth.partition import plan_islands

    plans = plan_islands(mesh, np.arange(2000), 5, 0.7, np.random.default_rng(80))
    assert len(plans) == 1
    atoms, k = plans[0]
    assert k == 5 and len(atoms) == 2000


def test_plan_islands_allocates_per_island(mesh):
    from mimesis_earth.partition import plan_islands

    z = mesh.points[:, 2]
    north = np.flatnonzero(z > 0.88)
    south = np.flatnonzero(z < -0.88)
    atom_idx = np.concatenate([north, south])
    plans = plan_islands(mesh, atom_idx, 6, 0.7, np.random.default_rng(81))
    assert len(plans) == 2
    ks = sorted(k for _, k in plans)
    assert sum(ks) == 6 and ks[0] >= 1
    covered = np.sort(np.concatenate([a for a, _ in plans]))
    np.testing.assert_array_equal(covered, np.sort(atom_idx))


def test_plan_islands_clusters_when_quota_short(mesh):
    from mimesis_earth.partition import plan_islands

    z = mesh.points[:, 2]
    bands = [
        np.flatnonzero(z > 0.9),
        np.flatnonzero((z > 0.4) & (z < 0.6)),
        np.flatnonzero((z > -0.6) & (z < -0.4)),
        np.flatnonzero(z < -0.9),
    ]
    atom_idx = np.concatenate(bands)
    plans = plan_islands(mesh, atom_idx, 2, 0.7, np.random.default_rng(82))
    assert len(plans) == 2
    assert all(k == 1 for _, k in plans)
    covered = np.sort(np.concatenate([a for a, _ in plans]))
    np.testing.assert_array_equal(covered, np.sort(atom_idx))


def test_plan_islands_mainland_not_starved_by_islands():
    from mimesis_earth.partition import _island_analysis, plan_islands

    # A dominant mainland flanked by several small islands with a tight
    # province budget (k == island count) and max coupling must NOT collapse
    # the mainland into a single giant province: it keeps its size-weighted
    # share while the smaller islands cluster into shared island-provinces.
    big = build_mesh(5000, np.random.default_rng(20))
    pts = big.points
    lon = np.arctan2(pts[:, 1], pts[:, 0])
    z = pts[:, 2]
    mainland = np.flatnonzero(z > 0.45)
    islands = [
        np.flatnonzero((z < -0.55) & (np.abs(lon - lon0) < 0.25))
        for lon0 in np.linspace(-2.6, 2.6, 4)
    ]
    atom_idx = np.unique(np.concatenate([mainland] + islands))
    _, _, sizes, sizeable = _island_analysis(big, atom_idx, np.random.default_rng(0))
    assert len(sizeable) == 5  # guards the fixture: 1 mainland + 4 sizeable islands
    k = len(sizeable)
    plans = plan_islands(big, atom_idx, k, 1.0, np.random.default_rng(1))
    covered = np.sort(np.concatenate([a for a, _ in plans]))
    np.testing.assert_array_equal(covered, np.sort(atom_idx))
    assert sum(kk for _, kk in plans) == k
    mainland_plan = max(plans, key=lambda g: len(g[0]))
    assert mainland_plan[1] >= 2  # mainland no longer starved to one province


def test_plan_islands_gives_islands_own_provinces_when_budget_allows():
    from mimesis_earth.partition import _island_analysis, plan_islands

    # With ample budget the small islands still each get their own province;
    # mainland-share-first must not swallow them when there is room.
    big = build_mesh(5000, np.random.default_rng(20))
    pts = big.points
    lon = np.arctan2(pts[:, 1], pts[:, 0])
    z = pts[:, 2]
    mainland = np.flatnonzero(z > 0.45)
    islands = [
        np.flatnonzero((z < -0.55) & (np.abs(lon - lon0) < 0.25))
        for lon0 in np.linspace(-2.6, 2.6, 4)
    ]
    atom_idx = np.unique(np.concatenate([mainland] + islands))
    _, _, _, sizeable = _island_analysis(big, atom_idx, np.random.default_rng(0))
    k = len(sizeable) + 8  # generous budget
    plans = plan_islands(big, atom_idx, k, 0.85, np.random.default_rng(2))
    assert sum(kk for _, kk in plans) == k
    covered = np.sort(np.concatenate([a for a, _ in plans]))
    np.testing.assert_array_equal(covered, np.sort(atom_idx))
    # every sizeable island is its own group and hosts at least one province
    assert len(plans) == len(sizeable)
    assert all(kk >= 1 for _, kk in plans)


def test_partition_weighted_extreme_variance(mesh):
    from scipy.sparse.csgraph import connected_components

    parts = partition_atoms(
        mesh, np.arange(2000), 8, None, 0.4, np.random.default_rng(90),
        size_variance=2.0,
    )
    assert sum(len(p) for p in parts) == 2000
    assert all(len(p) > 0 for p in parts)
    for p in parts:
        if len(p) < 2:
            continue
        n_comp, _ = connected_components(mesh.adjacency[p][:, p], directed=False)
        assert n_comp == 1


def test_partition_cost_field_locks_to_crests(mesh):
    # expensive band around the equator: borders forced to cross it must
    # settle on its crest (watershed behavior), concentrating border atoms there
    field = np.where(np.abs(mesh.points[:, 2]) < 0.15, 3.0, 0.0)
    atom_cost = np.exp(field)
    atom_idx = np.arange(len(mesh.points))

    def border_mean_cost(parts):
        label = np.empty(len(mesh.points), dtype=int)
        for i, p in enumerate(parts):
            label[p] = i
        e = mesh.edges
        border = label[e[:, 0]] != label[e[:, 1]]
        atoms = np.unique(np.concatenate([e[border, 0], e[border, 1]]))
        return field[atoms].mean()

    parts_flat = partition_atoms(
        mesh, atom_idx, 6, None, 0.3, np.random.default_rng(95)
    )
    parts_cost = partition_atoms(
        mesh, atom_idx, 6, None, 0.3, np.random.default_rng(95),
        atom_cost=atom_cost,
    )
    assert sum(len(p) for p in parts_cost) == len(atom_idx)
    # with the cost field, border atoms concentrate ON the band's crest
    assert border_mean_cost(parts_cost) > 2.0 * border_mean_cost(parts_flat)


def test_partition_cost_field_contiguity_without_repair(mesh):
    from scipy.sparse.csgraph import connected_components

    # size_variance=0 -> repair pass off; symmetric edge re-weighting must
    # keep parts contiguous by the shortest-path-tree property
    rng_field = np.random.default_rng(96)
    atom_cost = np.exp(1.5 * rng_field.normal(size=len(mesh.points)))
    for seed in (97, 98, 99):
        parts = partition_atoms(
            mesh, np.arange(len(mesh.points)), 7, None, 0.5,
            np.random.default_rng(seed), atom_cost=atom_cost,
        )
        for p in parts:
            n_comp, _ = connected_components(
                mesh.adjacency[p][:, p], directed=False
            )
            assert n_comp == 1


def test_partition_cost_field_bounded_stays_balanced(mesh):
    # A BOUNDED cost field must still split a blob reasonably evenly: the field
    # governs reachability in the weighted Voronoi, so an UNBOUNDED one lets a
    # low-cost basin dominate and starves the rest into slivers. generate.py
    # clips the exponent to keep this true (else a mainland collapses into one
    # giant province). size_variance=0 isolates the field's own effect.
    from mimesis_earth.noise import sphere_noise

    field = sphere_noise(mesh.points, np.random.default_rng(200))
    unbounded = np.exp(2.0 * field)
    bounded = np.exp(np.clip(2.0 * field, -1.0, 1.0))  # matches generate.py
    floor = len(mesh.points) // (4 * 6)
    bounded_min, unbounded_min = [], []
    for seed in (201, 202, 203, 204, 205):
        for cost, out in ((bounded, bounded_min), (unbounded, unbounded_min)):
            parts = partition_atoms(
                mesh, np.arange(len(mesh.points)), 6, None, 1.0,
                np.random.default_rng(seed), size_variance=0.0, atom_cost=cost,
            )
            assert sum(len(p) for p in parts) == len(mesh.points)
            out.append(min(len(p) for p in parts))
    # bounded field never collapses a part to a sliver...
    assert min(bounded_min) >= floor, bounded_min
    # ...whereas the unbounded field does (guards that the clip is load-bearing)
    assert min(unbounded_min) < floor, unbounded_min


def test_partition_cost_field_deterministic(mesh):
    atom_cost = np.exp(np.linspace(-1, 1, len(mesh.points)))
    a = partition_atoms(mesh, np.arange(1000), 4, None, 0.4,
                        np.random.default_rng(101), atom_cost=atom_cost)
    b = partition_atoms(mesh, np.arange(1000), 4, None, 0.4,
                        np.random.default_rng(101), atom_cost=atom_cost)
    for pa, pb in zip(a, b):
        np.testing.assert_array_equal(pa, pb)
