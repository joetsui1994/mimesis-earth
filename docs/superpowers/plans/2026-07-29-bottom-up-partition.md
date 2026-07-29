# Bottom-up partition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the top-down partitioner with a bottom-up agglomeration (leaf districts → provinces → countries) using field-biased region-growing, so higher-level borders inherit district-scale meander.

**Architecture:** Keep the mesh/landmask front end and the geometry/population/attribute back end untouched. Between them, a new `agglomerate.py` partitions each landmass into many jagged leaf districts, then agglomerates them upward with a seeded, size-balanced, ridge-field-biased region grower. Output is the same `level_nodes` structure the back end already consumes.

**Tech Stack:** Python 3.12, numpy, scipy.sparse (csgraph), shapely (unchanged), pydantic (spec), Vite/TypeScript (web UI).

**Spec:** `docs/superpowers/specs/2026-07-29-bottom-up-partition-design.md`

**Working directory:** repo root `/Users/user/Documents/work/mimesis-earth`. Run Python from `python/` with `PYTHONPATH=src` using the venv at `../.venv/bin/python`. Branch: `feat/bottom-up-partition` (already checked out).

---

## File structure

- **Create** `python/src/mimesis_earth/agglomerate.py` — region grower, item-graph builder, per-island leaf partition, per-group count allocation + feasibility, and the `partition_world` driver.
- **Create** `python/tests/test_agglomerate.py` — unit tests for the above.
- **Create** `python/scripts/measure_borders.py` — pooled tortuosity + balance acceptance measurement.
- **Modify** `python/src/mimesis_earth/spec.py` — remove `count_coupling`/`count_variance`; make `border_roughness` a scalar; bump `GENERATOR_VERSION`.
- **Modify** `python/src/mimesis_earth/generate.py` — compute `atom_cost` (leaf) + `grow_field` (macro), call `partition_world`, run feasibility check; delete the top-down loop.
- **Modify** `python/src/mimesis_earth/partition.py` — delete `coupled_counts`, `honor_minimums`, `plan_islands`, `_cluster_islands`, `count_sizeable_islands`, `_island_analysis`, `ISLET_MAX_ATOMS`.
- **Modify** `python/tests/test_partition.py`, `python/tests/test_world.py`, `python/tests/test_spec.py` — drop tests for deleted functions/fields; keep property tests.
- **Modify** `web/index.html`, `web/src/api.ts`, `web/src/panel.ts` — remove `coupling`/`counts` sliders; scalar `border_roughness`; strip unknown fields before POST.

---

## Task 1: Region-grow primitive + item-graph builder (plain)

**Files:**
- Create: `python/src/mimesis_earth/agglomerate.py`
- Test: `python/tests/test_agglomerate.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mimesis_earth.agglomerate'`.

- [ ] **Step 3: Write minimal implementation**

```python
# python/src/mimesis_earth/agglomerate.py
"""Bottom-up agglomeration: leaf districts -> provinces -> countries."""

from collections import defaultdict

import numpy as np

BRIDGE_EPS = 1e-6   # link weight for cross-water bridge edges
GROW_BIAS = 3.0     # region-grow field-bias strength (multiplies border_roughness)


def region_grow(neighbors, sizes, targets, seeds, rng, field=None, lam=0.0):
    """Grow K contiguous groups over an item graph, contiguous by construction.

    neighbors: dict item -> list[(neighbor, link_weight)].
    sizes:     array[float] item mass.
    targets:   array[float] length K, desired group mass.
    seeds:     list[int] length K, one starting item per group.
    field/lam: if given, prefer eating LOW-field frontier items (borders settle
               on high-field ridges). lam=0 -> plain strongest-link growth.

    Returns assign: array[int] length n, each in 0..K-1 (straggler guard fills
    any item the frontier never reached via an adjacent assigned group).
    """
    n = len(sizes)
    K = len(seeds)
    assign = np.full(n, -1)
    filled = np.zeros(K)
    frontier = [set() for _ in range(K)]
    link = [defaultdict(float) for _ in range(K)]
    # Two-phase seeding: assign ALL seeds first, then build frontiers. This
    # prevents a seed that is adjacent to another seed from being added to the
    # earlier group's frontier and later "stolen" (reassigned).
    for g, s in enumerate(seeds):
        assign[s] = g
        filled[g] = sizes[s]
    for g, s in enumerate(seeds):
        for nb, w in neighbors[s]:
            if assign[nb] == -1:
                frontier[g].add(nb)
                link[g][nb] += w
    remaining = n - K
    while remaining > 0:
        cand = [g for g in range(K) if frontier[g]]
        if not cand:
            break
        g = min(cand, key=lambda g: filled[g] / targets[g])
        items = sorted(frontier[g])  # canonical order -> determinism
        if field is not None:
            scores = [link[g][it] - lam * field[it] + 1e-9 * rng.random() for it in items]
        else:
            scores = [link[g][it] + 1e-9 * rng.random() for it in items]
        best = items[int(np.argmax(scores))]
        assign[best] = g
        filled[g] += sizes[best]
        remaining -= 1
        for gg in range(K):
            frontier[gg].discard(best)
        for nb, w in neighbors[best]:
            if assign[nb] == -1:
                frontier[g].add(nb)
                link[g][nb] += w
    _attach_stragglers(neighbors, assign)
    return assign


def _attach_stragglers(neighbors, assign):
    """Attach any unassigned item to its strongest-link assigned neighbor's group
    (never by chord distance -> preserves contiguity). Iterates so a straggler
    that only touches other stragglers is resolved once a neighbor is placed."""
    while True:
        stragglers = np.flatnonzero(assign == -1)
        if len(stragglers) == 0:
            return
        progressed = False
        for it in sorted(stragglers.tolist()):
            best_g, best_w = -1, -1.0
            for nb, w in neighbors[it]:
                if assign[nb] >= 0 and w > best_w:
                    best_g, best_w = int(assign[nb]), w
            if best_g >= 0:
                assign[it] = best_g
                progressed = True
        if not progressed:
            raise RuntimeError(
                "region_grow: items isolated from all seeds "
                f"({len(stragglers)} left) -- disconnected item graph"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/src/mimesis_earth/agglomerate.py python/tests/test_agglomerate.py
git commit -m "feat: region_grow primitive (plain, contiguous by construction)"
```

---

## Task 2: Field bias, balance, determinism, straggler guard

**Files:**
- Modify: `python/tests/test_agglomerate.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to python/tests/test_agglomerate.py

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
```

- [ ] **Step 2: Run to verify they fail or pass**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py -q`
Expected: these four pass already if Task 1 was written correctly (they exercise behavior already implemented). If `test_region_grow_field_bias_puts_border_on_ridge` fails, the bias sign is wrong — verify `scores` subtracts `lam * field[it]`.

- [ ] **Step 3: No new implementation needed** (Task 1 covered bias, balance, determinism, straggler). If a test fails, fix `region_grow` accordingly.

- [ ] **Step 4: Run full agglomerate tests**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_agglomerate.py
git commit -m "test: region_grow balance, field-bias, determinism, straggler guard"
```

---

## Task 3: Item-graph builder (mesh edges + ε bridges)

**Files:**
- Modify: `python/src/mimesis_earth/agglomerate.py`
- Modify: `python/tests/test_agglomerate.py`

- [ ] **Step 1: Write the failing test**

```python
# append to python/tests/test_agglomerate.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py::test_build_item_graph_adjacency_and_bridges -q`
Expected: FAIL — `cannot import name 'build_item_graph'`.

- [ ] **Step 3: Write implementation** (append to `agglomerate.py`)

```python
def build_item_graph(mesh, parts, bridges=None):
    """Adjacency over 'parts' (lists of atom indices). Edge weight = summed
    shared-border arc length. Bridge atom-pairs add BRIDGE_EPS links so
    across-water neighbors are reachable but eaten last."""
    lab = np.full(len(mesh.points), -1)
    for i, p in enumerate(parts):
        lab[p] = i
    e = mesh.edges
    a_all, b_all = lab[e[:, 0]], lab[e[:, 1]]
    m = (a_all >= 0) & (b_all >= 0) & (a_all != b_all)
    a, b = a_all[m], b_all[m]
    w = np.arccos(
        np.clip(np.sum(mesh.points[e[m, 0]] * mesh.points[e[m, 1]], axis=1), -1, 1)
    )
    nbr = defaultdict(lambda: defaultdict(float))
    for i, j, ww in zip(a.tolist(), b.tolist(), w.tolist()):
        nbr[i][j] += ww
        nbr[j][i] += ww
    if bridges is not None and len(bridges):
        ba, bb = lab[bridges[:, 0]], lab[bridges[:, 1]]
        bm = (ba >= 0) & (bb >= 0) & (ba != bb)
        for i, j in zip(ba[bm].tolist(), bb[bm].tolist()):
            nbr[i][j] += BRIDGE_EPS
            nbr[j][i] += BRIDGE_EPS
    neighbors = {i: [(j, ww) for j, ww in nbr[i].items()] for i in range(len(parts))}
    sizes = np.array([len(p) for p in parts], dtype=float)
    return neighbors, sizes
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/src/mimesis_earth/agglomerate.py python/tests/test_agglomerate.py
git commit -m "feat: build_item_graph with epsilon bridge links"
```

---

## Task 4: Per-island leaf partition (single-island, clustering, MIN clamp)

**Files:**
- Modify: `python/src/mimesis_earth/agglomerate.py`
- Modify: `python/tests/test_agglomerate.py`

- [ ] **Step 1: Write the failing test**

```python
# append to python/tests/test_agglomerate.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py::test_leaf_partition_covers_and_meets_min -q`
Expected: FAIL — `cannot import name 'leaf_partition'`.

- [ ] **Step 3: Write implementation** (append to `agglomerate.py`; add imports at top)

```python
# add to the imports block at the top of agglomerate.py:
from scipy.sparse.csgraph import connected_components

from mimesis_earth.partition import allocate_counts, partition_atoms, redistribute_counts
from mimesis_earth.spec import MIN_ATOMS_PER_LEAF
```

```python
def leaf_partition(mesh, group_atoms, n_districts, roughness, size_variance,
                   atom_cost, rng):
    """Partition one landmass group's atoms into n_districts single-island
    districts. Physical islands (mesh components, no bridges) are the units;
    the smallest are clustered onto their nearest until there are <= n_districts
    units, each >= MIN_ATOMS_PER_LEAF. Districts are then allocated across units
    (clamped so none exceeds unit_atoms // MIN_ATOMS_PER_LEAF) and each unit is
    cut with partition_atoms."""
    group_atoms = np.asarray(group_atoms)
    sub = mesh.adjacency[group_atoms][:, group_atoms]
    ncomp, comp = connected_components(sub, directed=False)
    units = [group_atoms[comp == c] for c in range(ncomp)]
    cents = [mesh.points[u].mean(0) for u in units]

    def merge_smallest():
        i = int(np.argmin([len(u) for u in units]))
        ci = cents[i]
        j = min((k for k in range(len(units)) if k != i),
                key=lambda k: float(np.linalg.norm(cents[k] - ci)))
        units[j] = np.concatenate([units[j], units[i]])
        cents[j] = mesh.points[units[j]].mean(0)
        del units[i]
        del cents[i]

    while len(units) > 1 and (
        len(units) > n_districts or min(len(u) for u in units) < MIN_ATOMS_PER_LEAF
    ):
        merge_smallest()

    unit_sizes = np.array([len(u) for u in units], dtype=float)
    caps = np.maximum(1, (unit_sizes // MIN_ATOMS_PER_LEAF).astype(int))
    alloc = redistribute_counts(allocate_counts(n_districts, unit_sizes), caps)
    districts = []
    for u, k in zip(units, alloc.tolist()):
        if k <= 1:
            districts.append(u)
        else:
            districts.extend(
                partition_atoms(mesh, u, k, None, roughness, rng,
                                size_variance=size_variance, atom_cost=atom_cost)
            )
    return districts
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py -q`
Expected: PASS. (If `redistribute_counts` raises "not enough total capacity", the group is infeasible — that is the feasibility case handled in Task 5/7, not here.)

- [ ] **Step 5: Commit**

```bash
git add python/src/mimesis_earth/agglomerate.py python/tests/test_agglomerate.py
git commit -m "feat: per-island leaf_partition with island clustering + MIN clamp"
```

---

## Task 5: Count allocation + per-group feasibility

**Files:**
- Modify: `python/src/mimesis_earth/agglomerate.py`
- Modify: `python/tests/test_agglomerate.py`

- [ ] **Step 1: Write the failing test**

```python
# append to python/tests/test_agglomerate.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py::test_allocate_group_counts_exact_and_feasible -q`
Expected: FAIL — `cannot import name 'allocate_group_counts'`.

- [ ] **Step 3: Write implementation** (append to `agglomerate.py`; add `import math` at top)

```python
def allocate_group_counts(group_sizes, levels):
    """Countries per landmass group (proportional to size, each >= 1) and the
    derived leaf-district count per group (C_g * prod(levels[1:])). Raises if a
    group is too small to host D_g * MIN_ATOMS_PER_LEAF atoms (review L)."""
    group_sizes = np.asarray(group_sizes, dtype=float)
    C = redistribute_counts(
        allocate_counts(levels[0], group_sizes), group_sizes.astype(int)
    )
    leaves_per_country = math.prod(levels[1:]) if len(levels) > 1 else 1
    D = C * leaves_per_country
    need = D * MIN_ATOMS_PER_LEAF
    if (group_sizes < need).any():
        g = int(np.flatnonzero(group_sizes < need)[0])
        raise ValueError(
            f"landmass group {g} is too small: has {int(group_sizes[g])} atoms, "
            f"needs >= {int(need[g])} for {int(D[g])} districts at "
            f"MIN_ATOMS_PER_LEAF={MIN_ATOMS_PER_LEAF}. Lower n_landmasses, raise "
            f"resolution or land_fraction, or lower spread."
        )
    return C, D
```

Add `import math` to the top imports of `agglomerate.py`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/src/mimesis_earth/agglomerate.py python/tests/test_agglomerate.py
git commit -m "feat: per-group count allocation with feasibility check"
```

---

## Task 6: The `partition_world` driver

**Files:**
- Modify: `python/src/mimesis_earth/agglomerate.py`
- Modify: `python/tests/test_agglomerate.py`

- [ ] **Step 1: Write the failing test**

```python
# append to python/tests/test_agglomerate.py
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py::test_partition_world_shape_and_nesting -q`
Expected: FAIL — `cannot import name 'partition_world'`.

- [ ] **Step 3: Write implementation** (append to `agglomerate.py`)

```python
def _item_field(grow_field, parts):
    """Mean grow_field over each part's atoms."""
    return np.array([float(grow_field[p].mean()) for p in parts])


def _grow_targets(total_mass, k, size_variance, rng):
    if size_variance <= 0:
        return np.full(k, total_mass / k)
    w = rng.lognormal(0.0, size_variance, size=k)
    return total_mass * w / w.sum()


def partition_world(mesh, mask, spec, atom_cost, grow_field, rng):
    """Bottom-up partition. Returns level_nodes: list per level of
    dicts {atoms, parent, landmass}. Leaves are districts; parents set by
    field-biased region-grow. See design spec Components 1-4."""
    levels = spec.levels
    n_levels = len(levels)
    roughness = float(spec.border_roughness)
    lam = GROW_BIAS * roughness
    group_sizes = np.array(
        [(mask.group == g).sum() for g in range(spec.n_landmasses)], dtype=float
    )
    C, D = allocate_group_counts(group_sizes, levels)

    level_nodes = [[] for _ in range(n_levels)]

    for g in range(spec.n_landmasses):
        group_atoms = np.flatnonzero(mask.group == g)
        # per-level counts for this group (index 0 = countries ... last = leaves)
        cnt = [int(C[g])]
        for lvl in range(1, n_levels):
            cnt.append(cnt[-1] * levels[lvl])

        # --- leaves (finest level) ---
        leaves = leaf_partition(mesh, group_atoms, cnt[-1], roughness,
                                spec.size_variance, atom_cost, rng)

        # --- agglomerate upward: parts[level] = list of atom arrays;
        #     parent_of[level][i] = index into parts[level-1] within this group.
        parts = [None] * n_levels
        parent_of = [None] * n_levels
        parts[n_levels - 1] = leaves
        for lvl in range(n_levels - 2, -1, -1):
            child_parts = parts[lvl + 1]
            neighbors, sizes = build_item_graph(mesh, child_parts, bridges=mask.bridges)
            field = _item_field(grow_field, child_parts)
            k = cnt[lvl]
            cent = np.array([mesh.points[p].mean(0) for p in child_parts])
            cent /= np.linalg.norm(cent, axis=1, keepdims=True)
            seeds = _fps(cent, k)
            targets = _grow_targets(sizes.sum(), k, spec.size_variance, rng)
            assign = region_grow(neighbors, sizes, targets, seeds, rng,
                                 field=field, lam=lam)
            parts[lvl] = [
                np.concatenate([child_parts[i] for i in np.flatnonzero(assign == c)])
                for c in range(k)
            ]
            parent_of[lvl + 1] = assign  # child level -> its parent index (this level)

        # --- append to global level_nodes with per-group parent offsets ---
        base = [len(level_nodes[lvl]) for lvl in range(n_levels)]
        for lvl in range(n_levels):
            for i, atoms in enumerate(parts[lvl]):
                node = {"atoms": atoms, "landmass": g if lvl == 0 else None}
                if lvl == 0:
                    node["parent"] = None
                else:
                    node["parent"] = base[lvl - 1] + int(parent_of[lvl][i])
                level_nodes[lvl].append(node)

    return level_nodes


def _fps(points, k):
    """Farthest-point sampling: k well-spread indices into points."""
    chosen = [0]
    d = np.linalg.norm(points - points[0], axis=1)
    while len(chosen) < k:
        nxt = int(d.argmax())
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return chosen
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/src/mimesis_earth/agglomerate.py python/tests/test_agglomerate.py
git commit -m "feat: partition_world bottom-up driver (level_nodes)"
```

---

## Task 7: Spec changes (remove count knobs, scalar roughness, version bump)

**Files:**
- Modify: `python/src/mimesis_earth/spec.py`
- Modify: `python/tests/test_spec.py`

- [ ] **Step 1: Write the failing test**

```python
# append to python/tests/test_spec.py
import pytest
from mimesis_earth.spec import WorldSpec


def test_spec_rejects_removed_and_list_fields():
    with pytest.raises(Exception):
        WorldSpec(count_coupling=0.5)          # removed field, extra=forbid
    with pytest.raises(Exception):
        WorldSpec(count_variance=0.5)          # removed field
    with pytest.raises(Exception):
        WorldSpec(border_roughness=[0.2, 0.5]) # now scalar-only


def test_spec_border_roughness_scalar_ok():
    s = WorldSpec(border_roughness=0.9)
    assert s.border_roughness == 0.9
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_spec.py::test_spec_rejects_removed_and_list_fields -q`
Expected: FAIL (constructing with `count_coupling` currently succeeds).

- [ ] **Step 3: Edit `spec.py`**

- Bump version: change `GENERATOR_VERSION = "0.5.0"` to `GENERATOR_VERSION = "0.6.0"`.
- Delete the `count_variance` and `count_coupling` field lines.
- Change `border_roughness: Union[float, list[float]] = 0.7` to:

```python
    border_roughness: float = Field(default=0.7, ge=0.0, le=2.0)
```

- Delete the `border_roughness_per_level` method (no longer used).
- In `_validate`, delete the block that validates `border_roughness` as a list
  (the `isinstance(self.border_roughness, list)` check and the `values`/`all(...)`
  lines). Keep the `levels`, `level_names`, `n_landmasses`, and resolution checks.
- Remove the now-unused `Union` import if present.

- [ ] **Step 4: Run to verify it passes**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_spec.py -q`
Expected: PASS. (Other spec tests that reference the removed fields/method are updated in Task 10.)

- [ ] **Step 5: Commit**

```bash
git add python/src/mimesis_earth/spec.py python/tests/test_spec.py
git commit -m "feat: retire count_coupling/count_variance; scalar border_roughness; v0.6.0"
```

---

## Task 8: Wire `generate.py` to the new driver

**Files:**
- Modify: `python/src/mimesis_earth/generate.py:30-145` (the partitioning core)
- Test: `python/tests/test_world.py` (existing property tests are the check)

- [ ] **Step 1: Confirm the existing world property tests describe the contract**

Read `python/tests/test_world.py`. The tests `test_unit_counts_exact_when_variance_zero`, `test_ids_and_parents`, `test_children_tile_parent_exactly`, `test_siblings_do_not_overlap`, `test_landmass_count`, `test_deterministic_and_seed_sensitive`, and `test_low_level_units_contiguous` define the contract the new core must satisfy. They will be the pass/fail signal for this task.

- [ ] **Step 2: Rewrite the partitioning core**

In `generate.py`, replace the block from the `atom_cost` comment through the end of the level loop (currently the `atom_cost_for` def and both `partition_atoms` loops that build `level_nodes` and `counts_by_level`) with:

```python
    # --- cost fields ------------------------------------------------------
    land_elevation = mask.elevation[mask.land]
    elev_z = (mask.elevation - land_elevation.mean()) / land_elevation.std()
    # leaf-border texture (atom scale): elevation crests + coherent noise, clipped
    leaf_noise = sphere_noise(
        mesh.points, np.random.default_rng([spec.seed, 0xB0DE]),
        octaves=BORDER_NOISE_OCTAVES, base_freq=BORDER_NOISE_FREQ,
        persistence=BORDER_NOISE_PERSISTENCE,
    )
    atom_cost = np.exp(np.clip(
        1.5 * spec.border_meander * elev_z
        + BORDER_NOISE_COST * spec.border_roughness * leaf_noise,
        -COST_EXPONENT_CLIP, COST_EXPONENT_CLIP,
    ))
    # macro ridge field for region-grow bias: low frequency so borders wander at
    # country scale; elevation term makes meander propagate to higher levels.
    grow_noise = sphere_noise(
        mesh.points, np.random.default_rng([spec.seed, 0x6600]),
        octaves=3, base_freq=4.0, persistence=0.7,
    )
    grow_field = spec.border_meander * elev_z + spec.border_roughness * grow_noise

    # --- bottom-up partition ---------------------------------------------
    level_nodes = partition_world(mesh, mask, spec, atom_cost, grow_field, rng)
```

- [ ] **Step 3: Update imports and remove dead references in `generate.py`**

- Add `from mimesis_earth.agglomerate import partition_world`.
- Remove `counts_by_level` construction. Update the `_capture` block to:

```python
    if _capture is not None:
        _capture["mesh"] = mesh
        _capture["level_nodes"] = level_nodes
        _capture["elevation"] = mask.elevation
```

- Remove now-unused imports from `mimesis_earth.partition` (keep only what the
  back half uses — after this task `generate.py` no longer calls
  `allocate_counts`, `coupled_counts`, `honor_minimums`, `plan_islands`,
  `redistribute_counts`, or `_island_analysis` directly). Keep `sphere_noise`
  and the `BORDER_NOISE_*` / `COST_EXPONENT_CLIP` constants.

- [ ] **Step 4: Run the world tests**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_world.py -q`
Expected: The count/nesting/contiguity/determinism tests PASS. If `test_low_level_units_contiguous` fails on its `counts_by_level` capture, note that key is removed — that test is rewritten in Task 10; for now run the subset that does not reference `counts_by_level`:

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_world.py -q -k "not low_level_units_contiguous and not border_roughness and not border_meander"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/src/mimesis_earth/generate.py
git commit -m "feat: generate uses bottom-up partition_world"
```

---

## Task 9: Delete dead code from `partition.py`

**Files:**
- Modify: `python/src/mimesis_earth/partition.py`

- [ ] **Step 1: Delete the unused functions and constant**

Delete these definitions entirely: `coupled_counts`, `honor_minimums`,
`_island_analysis`, `count_sizeable_islands`, `_cluster_islands`, `plan_islands`,
and the `ISLET_MAX_ATOMS` constant with its comment block.

Keep: `pick_seeds`, `_subgraph`, `_assign_labels`, `_repair_contiguity`,
`partition_atoms`, `allocate_counts`, `redistribute_counts`, `BRIDGE_COST_FACTOR`.
Remove the now-unused `cKDTree` import if nothing else uses it.

- [ ] **Step 2: Verify partition.py still imports**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -c "import mimesis_earth.partition"`
Expected: no error.

- [ ] **Step 3: Run the agglomerate + generate paths**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest tests/test_agglomerate.py -q && PYTHONPATH=src ../.venv/bin/python -c "from mimesis_earth.spec import WorldSpec; from mimesis_earth.generate import generate; generate(WorldSpec(resolution=8000, seed=1)); print('ok')"`
Expected: agglomerate tests PASS; generate prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add python/src/mimesis_earth/partition.py
git commit -m "refactor: remove top-down count/island machinery from partition.py"
```

---

## Task 10: Update the remaining tests

**Files:**
- Modify: `python/tests/test_partition.py`
- Modify: `python/tests/test_world.py`
- Modify: `python/tests/test_spec.py`

- [ ] **Step 1: Delete tests bound to removed functions**

In `test_partition.py` delete: `test_allocate_counts` may stay (function kept);
delete `test_coupled_counts_*`, `test_honor_minimums`, `test_plan_islands_*`,
`test_partition_cost_field_bounded_stays_balanced` only if it referenced removed
symbols (it uses `partition_atoms` + `sphere_noise`, so keep it). Delete any test
importing `coupled_counts`, `honor_minimums`, `plan_islands`,
`count_sizeable_islands`, `_island_analysis`, or `ISLET_MAX_ATOMS`.

In `test_spec.py` delete/adjust any test asserting `count_coupling`,
`count_variance`, `border_roughness_per_level`, or list `border_roughness`.

- [ ] **Step 2: Rewrite `test_low_level_units_contiguous` in `test_world.py`**

Replace it with a contiguity check that does not depend on `counts_by_level` or
`plan_islands` (contiguity is now guaranteed by construction, but islands may
legitimately make a country/province a MultiPolygon):

```python
def test_every_leaf_is_contiguous():
    from scipy.sparse.csgraph import connected_components
    for spec in (
        WorldSpec(levels=[4, 4, 3], n_landmasses=3, resolution=8000, seed=21),
        WorldSpec(levels=[6, 2, 2], n_landmasses=4, resolution=8000,
                  land_fraction=0.35, seed=7),
    ):
        cap = {}
        generate(spec, _capture=cap)
        mesh, level_nodes = cap["mesh"], cap["level_nodes"]
        for node in level_nodes[-1]:            # leaf districts
            atoms = node["atoms"]
            sub = mesh.adjacency[atoms][:, atoms]
            assert connected_components(sub, directed=False)[0] == 1, spec.seed
```

- [ ] **Step 3: Update the border tests in `test_world.py`**

`test_border_roughness_wiggles_coarse_borders` and
`test_border_meander_changes_borders_only_when_on` still apply (both knobs still
exist and now act through the leaf cost field + grow field). Change any
`border_roughness=[...]` list usage to a scalar. If
`test_border_roughness_wiggles_coarse_borders` was measuring level-0 tortuosity,
it will now measure the bottom-up borders — keep the assertion `rough > smooth * 1.1`.

- [ ] **Step 4: Run the full suite**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest -q`
Expected: PASS (all remaining tests).

- [ ] **Step 5: Commit**

```bash
git add python/tests/
git commit -m "test: update suite for bottom-up partition; drop removed-fn tests"
```

---

## Task 11: Web UI — remove coupling/counts sliders, scalar roughness, strip old fields

**Files:**
- Modify: `web/index.html`
- Modify: `web/src/api.ts`
- Modify: `web/src/panel.ts`

- [ ] **Step 1: Remove the sliders in `web/index.html`**

Delete these two lines:

```html
      <label>coupling <input id="p-coupling" type="range" min="0" max="1" step="0.05" value="0.85" /></label>
      <label>counts <input id="p-counts" type="range" min="0" max="2" step="0.05" value="0.2" /></label>
```

- [ ] **Step 2: Update `web/src/api.ts` Spec type**

Remove `count_coupling` and `count_variance` from the `Spec` interface. Leave
`border_roughness: number` (already scalar).

- [ ] **Step 3: Update `web/src/panel.ts` `readSpec`**

Remove the `count_coupling` and `count_variance` lines. Confirm no `$('p-coupling')`
or `$('p-counts')` remain.

- [ ] **Step 4: Strip unknown fields for old shared links**

If `readSpec` ever merges query-string params (search `web/src` for
`URLSearchParams` / `location.search`), drop unknown keys before building the
`Spec` so an old link carrying `count_coupling`/`count_variance` loads with
defaults instead of a 422. If no query-param loading exists, add a one-line note
in `readSpec` that only the known fields above are sent (no action needed).

- [ ] **Step 5: Build and smoke-test**

```bash
cd web && npm run build
```
Expected: `tsc` passes (proves the type changes are consistent), vite build succeeds.

```bash
cd .. && ./scripts/build_web.sh
```
Expected: webapp embedded.

- [ ] **Step 6: Commit**

```bash
git add web/index.html web/src/api.ts web/src/panel.ts python/src/mimesis_earth/webapp
git commit -m "feat(web): drop coupling/counts sliders; scalar border_roughness"
```

---

## Task 12: Acceptance — tortuosity + balance measurement, then visual gate

**Files:**
- Create: `python/scripts/measure_borders.py`

- [ ] **Step 1: Write the measurement script**

```python
# python/scripts/measure_borders.py
"""Acceptance gate: pooled interior-country-border macro tortuosity and country
size balance for the bottom-up partitioner. Prints numbers; exits non-zero if
below the spec thresholds (macro tortuosity >= 1.6, country size CV <= 0.45)."""
import sys
import numpy as np
from shapely.ops import linemerge
from mimesis_earth.spec import WorldSpec
from mimesis_earth.generate import generate


def lines_of(sh):
    if sh.geom_type == "LineString":
        return [sh]
    m = linemerge(sh)
    return list(getattr(m, "geoms", [m]))


def macro_tortuosity(world):
    units = world.units_at(0)
    lm = {u.id: u.landmass for u in units}
    t, w = [], []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            a, b = units[i], units[j]
            if lm[a.id] != lm[b.id] or not a.geometry.intersects(b.geometry):
                continue
            sh = a.geometry.boundary.intersection(b.geometry.boundary)
            if sh.is_empty or sh.length == 0:
                continue
            for ln in lines_of(sh):
                if ln.geom_type != "LineString" or len(ln.coords) < 2:
                    continue
                p0, p1 = ln.coords[0], ln.coords[-1]
                span = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
                if span < 3:
                    continue
                t.append(ln.simplify(1.2).length / span)
                w.append(span)
    if not w:
        return None
    w = np.array(w)
    return float((np.array(t) * w).sum() / w.sum())


def country_cv(world):
    a = np.array([u.area_km2 for u in world.units_at(0)])
    return float(a.std() / a.mean())


torts, cvs = [], []
for seed in range(6):
    w = generate(WorldSpec(n_landmasses=3, levels=[6, 5, 6], resolution=20000,
                           land_fraction=0.35, seed=seed))
    mt = macro_tortuosity(w)
    if mt is not None:
        torts.append(mt)
    cvs.append(country_cv(w))

mt = float(np.mean(torts))
cv = float(np.mean(cvs))
print(f"pooled macro tortuosity (interior country borders): {mt:.3f}  (>= 1.60)")
print(f"country area CV: {cv:.3f}  (<= 0.45)")
ok = mt >= 1.60 and cv <= 0.45
print("ACCEPTANCE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
```

- [ ] **Step 2: Run the acceptance gate**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python scripts/measure_borders.py`
Expected: `ACCEPTANCE: PASS` (macro tortuosity ≥ 1.6, CV ≤ 0.45). If tortuosity is
low, raise `GROW_BIAS` in `agglomerate.py` (measured sweet spot ≈ 3; try 4) and
re-run; if CV is high, that indicates a balance regression to investigate before
proceeding.

- [ ] **Step 3: Visual gate**

```bash
cd .. && .venv/bin/mimesis-earth serve --port 8000
```
Open `http://localhost:8000`, set borders and meander to max, roll several worlds
(spacebar), and confirm interior country borders visibly wander (not straight).
Compare province/district levels look organic. Stop the server when done.

- [ ] **Step 4: Run the full suite once more**

Run: `cd python && PYTHONPATH=src ../.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/scripts/measure_borders.py
git commit -m "test: border-meander + balance acceptance measurement"
```

---

## Notes for the implementer

- **`level_nodes` contract:** each entry is `{"atoms": np.ndarray, "parent": int|None, "landmass": int|None}`. Leaves are the last level. The back half of `generate.py` (geometry union, population, elevation, naming, `Unit` build) is unchanged and depends only on this shape — do not touch it.
- **Determinism:** `region_grow` iterates frontiers in sorted item-id order; do not switch to set iteration. The two coherent fields draw from independent `default_rng([seed, tag])` streams so `border_roughness=0` reproduces the pre-field layout for the leaf texture field.
- **Feasibility errors are expected** for extreme spec/landmask combinations (tiny lopsided groups); they should raise the clear `allocate_group_counts` message, never emit sub-`MIN_ATOMS` districts.
- **`partition_atoms` is reused as-is** for the leaf partition; its substantial-island seeding and starved-part escape are inert on single connected islands (harmless). Do not refactor it in this plan.
