# Partition Realism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Heavy-tailed unit sizes (`size_variance`), territory-coupled child counts (`count_coupling`, reworked `count_variance` semantics with exact per-level totals), and island-aware quotas so units below country level are contiguous (islet exception).

**Architecture:** All changes live in `spec.py` (two new fields, version bump), `partition.py` (weighted growth in `partition_atoms`, `coupled_counts`, `honor_minimums`, `plan_islands`, `count_sizeable_islands`), `generate.py` (rewired deeper-level loop: island plans instead of bridges), and the panel (two sliders). Spec: `docs/superpowers/specs/2026-07-27-partition-realism-addendum.md` — read it first.

**Tech stack:** unchanged (numpy/scipy/shapely/pydantic; Vite+TS).

**Working conventions:** venv at repo root (`cd python && ../.venv/bin/pytest`); branch `feature/partition-realism`; commit per task. Baseline suite: **83 passed**. RNG discipline: single rng stream; `_subgraph(..., roughness=0.0, rng)` draws nothing (already guarded).

---

### Task 1: WorldSpec fields + version bump

**Files:** Modify `python/src/mimesis_earth/spec.py`, `python/tests/test_spec.py`

- [ ] **Step 1: Failing tests** — append to `python/tests/test_spec.py`:

```python
def test_new_realism_fields_defaults():
    spec = WorldSpec()
    assert spec.size_variance == 0.4
    assert spec.count_coupling == 0.7
    assert spec.generator_version == "0.2.0"


def test_rejects_bad_size_variance_and_coupling():
    with pytest.raises(ValidationError):
        WorldSpec(size_variance=1.5)
    with pytest.raises(ValidationError):
        WorldSpec(count_coupling=-0.1)
```

Run: `cd python && ../.venv/bin/pytest tests/test_spec.py -v` → 2 new tests FAIL (default missing / version mismatch).

- [ ] **Step 2: Implement** — in `python/src/mimesis_earth/spec.py`: change `GENERATOR_VERSION = "0.1.0"` to `"0.2.0"`; add fields after `count_variance`:

```python
    size_variance: float = Field(default=0.4, ge=0.0, le=1.0)
    count_coupling: float = Field(default=0.7, ge=0.0, le=1.0)
```

Also update the `count_variance` field's neighborhood comment semantics by replacing the docstring line of the class if present — no other changes.

- [ ] **Step 3: Verify** — `tests/test_spec.py` all pass; full suite 85 passed.
- [ ] **Step 4: Commit** — `git add python && git commit -m "feat: size_variance and count_coupling spec fields; generator 0.2.0"`

---

### Task 2: coupled_counts + honor_minimums (+ redistribute minimums clamp)

**Files:** Modify `python/src/mimesis_earth/partition.py`, `python/tests/test_partition.py`

- [ ] **Step 1: Failing tests** — append to `python/tests/test_partition.py`:

```python
def test_coupled_counts_exact_total_and_min_one():
    from mimesis_earth.partition import coupled_counts

    sizes = np.array([1000.0, 100.0, 10.0])
    for variance in (0.0, 0.5, 1.0):
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
```

Run → FAIL with ImportError.

- [ ] **Step 2: Implement** — append to `python/src/mimesis_earth/partition.py`:

```python
def coupled_counts(
    total: int,
    sizes: np.ndarray,
    coupling: float,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Split `total` children among parents. Weights follow each parent's
    territory share raised to `coupling` (0 = uniform, 1 = proportional),
    jittered log-normally by `variance`. Total exact, each parent >= 1."""
    weights = np.asarray(sizes, dtype=float) ** coupling
    if variance > 0:
        weights = weights * rng.lognormal(0.0, variance, size=len(weights))
    return allocate_counts(total, weights)


def honor_minimums(counts: np.ndarray, minimums: np.ndarray) -> np.ndarray:
    """Best-effort: raise counts toward per-parent minimums by borrowing from
    parents above their own minimum. Preserves the total exactly; never takes
    a donor below max(1, its minimum). Shortfalls remain when donors run out
    (callers degrade gracefully via island clustering)."""
    counts = np.asarray(counts, dtype=int).copy()
    minimums = np.asarray(minimums, dtype=int)
    while True:
        need = minimums - counts
        needy = np.flatnonzero(need > 0)
        surplus = counts - np.maximum(minimums, 1)
        donors = np.flatnonzero(surplus > 0)
        if len(needy) == 0 or len(donors) == 0:
            return counts
        counts[needy[int(np.argmax(need[needy]))]] += 1
        counts[donors[int(np.argmax(surplus[donors]))]] -= 1
```

(Loop terminates: every iteration reduces total unmet need by 1.)

- [ ] **Step 3: Verify** — new tests pass; full suite 89 passed. Note: `child_counts` stays for now (generate.py still uses it); it is removed in Task 4.
- [ ] **Step 4: Commit** — `"feat: coupled_counts allocation and honor_minimums borrowing"`

---

### Task 3: Weighted growth in partition_atoms

**Files:** Modify `python/src/mimesis_earth/partition.py`, `python/tests/test_partition.py`

- [ ] **Step 1: Failing tests** — append:

```python
def test_partition_weighted_sizes():
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
            vals.append(sizes.std() / sizes.mean())
        cvs[sv] = float(np.mean(vals))
    assert cvs[0.0] < 0.15
    assert cvs[0.8] > 0.3


def test_partition_weighted_deterministic(mesh):
    atom_idx = np.arange(1500)
    a = partition_atoms(mesh, atom_idx, 5, None, 0.5,
                        np.random.default_rng(64), size_variance=0.7)
    b = partition_atoms(mesh, atom_idx, 5, None, 0.5,
                        np.random.default_rng(64), size_variance=0.7)
    for pa, pb in zip(a, b):
        np.testing.assert_array_equal(pa, pb)
```

Run → FAIL (unexpected keyword `size_variance`).

- [ ] **Step 2: Implement** — in `python/src/mimesis_earth/partition.py`:

1. `_assign_labels` takes weights and scales distances:

```python
def _assign_labels(adj, seeds, pts, weights):
    dist = np.asarray(dijkstra(adj, directed=False, indices=seeds))
    labels = (dist / weights[:, None]).argmin(axis=0)
    unreachable = ~np.isfinite(dist.min(axis=0))
    if unreachable.any():
        chord = np.linalg.norm(
            pts[unreachable][:, None, :] - pts[seeds][None, :, :], axis=2
        )
        labels[unreachable] = (chord / weights[None, :]).argmin(axis=1)
    return labels
```

2. `partition_atoms(mesh, atom_idx, k, extra_edges, roughness, rng, size_variance: float = 0.0)`:
   - after `seeds = ...`, draw once: `weights = rng.lognormal(0.0, size_variance, size=k) if size_variance > 0 else np.ones(k)` (guard keeps rng stream identical at 0).
   - pass `weights` to every `_assign_labels` call.
   - starved-part test becomes weight-aware — replace the `sizes[i] < mean_size / 8.0` check with:

```python
        expected = len(atom_idx) * weights / weights.sum()
        for i in range(k):
            if part_sizes[i] < max(2.0, expected[i] / 8.0):
```

   (keep the largest-part relocation + duplicate-seed guard unchanged; `part_sizes` is the existing per-part size array — match the current variable name in the code.)

- [ ] **Step 3: Verify** — new tests pass AND all existing partition tests still pass (default `size_variance=0.0` preserves old behavior exactly, including `test_partition_deterministic` and `test_partition_balance`). Full suite 91 passed.
- [ ] **Step 4: Commit** — `"feat: weighted growth (size_variance) in partition_atoms"`

---

### Task 4: Island-aware quotas + generate() rewiring

**Files:** Modify `python/src/mimesis_earth/partition.py`, `python/src/mimesis_earth/generate.py`, `python/tests/test_partition.py`, `python/tests/test_world.py`

- [ ] **Step 1: Failing tests** — append to `python/tests/test_partition.py`:

```python
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
```

And to `python/tests/test_world.py`:

```python
def test_low_level_units_contiguous():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    from mimesis_earth.landmask import build_landmask
    from mimesis_earth.mesh import build_mesh
    from mimesis_earth.partition import ISLET_MAX_ATOMS

    # rugged multi-island world with ample quota
    spec = WorldSpec(
        levels=[4, 4, 3], n_landmasses=3, coast_ruggedness=0.8,
        resolution=8000, seed=21,
    )
    world = generate(spec)
    # rebuild the mesh/landmask deterministically to recover atom geometry
    # (same seed + spec -> identical mesh) and map units to atoms via area:
    # instead, verify via geometry: for levels >= 1, all polygon parts other
    # than the largest must be islet-sized
    atom_area_km2 = 4 * 3.141592653589793 * 6371.0**2 / spec.resolution
    islet_bound = 2.5 * ISLET_MAX_ATOMS * atom_area_km2
    for level in (1, 2):
        for u in world.units_at(level):
            if u.geometry.geom_type != "MultiPolygon":
                continue
            # planar-degree areas are meaningless; use spherical km2 shares by
            # ranking parts by area fraction of the unit's own area_km2
            parts = sorted(u.geometry.geoms, key=lambda p: p.area, reverse=True)
            total = sum(p.area for p in parts)
            for extra in parts[1:]:
                extra_km2 = u.area_km2 * (extra.area / total)
                assert extra_km2 < islet_bound, (u.id, extra_km2, islet_bound)


def test_exact_totals_at_any_variance_and_coupled_counts_vary():
    spec = WorldSpec(
        levels=[5, 4, 4], count_variance=0.8, count_coupling=1.0,
        resolution=8000, seed=23,
    )
    world = generate(spec)
    assert len(world.units_at(0)) == 5
    assert len(world.units_at(1)) == 20
    assert len(world.units_at(2)) == 80
    per_parent: dict = {}
    for u in world.units_at(1):
        per_parent[u.parent_id] = per_parent.get(u.parent_id, 0) + 1
    assert len(set(per_parent.values())) > 1
```

Run → FAIL (no `plan_islands`; totals test may fail against old generate).

NOTE on the contiguity test: the planar-area-ratio proxy (`extra.area / total` of lon/lat degree areas scaled by the unit's true `area_km2`) is approximate near poles; the 2.5x slack on the bound absorbs it. If it flakes, tighten the spec's latitude spread (lower `spread`) rather than raising slack past 2.5x, and report.

- [ ] **Step 2: Implement plan_islands + count_sizeable_islands** — append to `python/src/mimesis_earth/partition.py`:

```python
ISLET_MAX_ATOMS = 8


def _island_analysis(mesh: Mesh, atom_idx: np.ndarray, rng: np.random.Generator):
    """Connected components of atom_idx over mesh edges only (no bridges).
    Returns (n_comp, comp_labels, comp_sizes, sizeable_component_ids)."""
    sub = _subgraph(mesh, atom_idx, None, 0.0, rng)  # roughness 0: no rng draws
    n_comp, comp = connected_components(sub, directed=False)
    sizes = np.bincount(comp)
    sizeable = np.flatnonzero(sizes >= ISLET_MAX_ATOMS)
    if len(sizeable) == 0:
        sizeable = np.array([int(sizes.argmax())])
    return n_comp, comp, sizes, sizeable


def count_sizeable_islands(
    mesh: Mesh, atom_idx: np.ndarray, rng: np.random.Generator
) -> int:
    _, _, _, sizeable = _island_analysis(mesh, np.asarray(atom_idx), rng)
    return len(sizeable)


def plan_islands(
    mesh: Mesh,
    atom_idx: np.ndarray,
    k: int,
    coupling: float,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, int]]:
    """Split a parent's atoms into island groups with per-group child counts
    summing to k. Sizeable islands each host >= 1 child when quota allows;
    islets attach to the nearest sizeable island. With more sizeable islands
    than quota, islands are clustered by proximity (each cluster = 1 child)."""
    atom_idx = np.asarray(atom_idx)
    n_comp, comp, sizes, sizeable = _island_analysis(mesh, atom_idx, rng)
    if n_comp == 1:
        return [(atom_idx, k)]
    centroids = np.stack(
        [mesh.points[atom_idx[comp == c]].mean(axis=0) for c in range(n_comp)]
    )
    # attach every non-sizeable component to its nearest sizeable island
    owner = np.empty(n_comp, dtype=int)
    for c in range(n_comp):
        if c in set(sizeable.tolist()):
            owner[c] = c
        else:
            d = np.linalg.norm(centroids[sizeable] - centroids[c], axis=1)
            owner[c] = int(sizeable[int(d.argmin())])
    m = len(sizeable)
    if m <= k:
        group_sizes = np.array(
            [float(sizes[owner == s].sum()) for s in sizeable]
        )
        alloc = allocate_counts(k, group_sizes**coupling)
        alloc = redistribute_counts(alloc, group_sizes.astype(int))
        groups = list(zip(sizeable.tolist(), alloc.tolist()))
    else:
        order = sizeable[np.argsort(-sizes[sizeable])]
        cluster_seeds = order[:k]
        seed_set = np.asarray(cluster_seeds)
        cluster_of = {int(s): int(s) for s in cluster_seeds}
        for c in order[k:]:
            d = np.linalg.norm(centroids[seed_set] - centroids[int(c)], axis=1)
            cluster_of[int(c)] = int(seed_set[int(d.argmin())])
        for c in range(n_comp):
            owner[c] = cluster_of[int(owner[c])] if int(owner[c]) in cluster_of else int(owner[c])
        # any owner not itself a cluster seed maps through cluster_of
        owner = np.array([cluster_of.get(int(o), int(o)) for o in owner])
        groups = [(int(s), 1) for s in cluster_seeds]
    out = []
    for s, count in groups:
        members = atom_idx[owner[comp] == s]
        out.append((members, int(count)))
    return out
```

(`group_sizes**coupling` with coupling 0 gives uniform weights — fine for `allocate_counts`. Bincount note: `sizes[owner == s].sum()` — verify shapes; `sizes` is per-component, `owner` per-component: `sizes[owner == s].sum()` sums the component sizes owned by island s. Implementer: sanity-check this line, it's the subtle one.)

- [ ] **Step 3: Rewire generate()** — in `python/src/mimesis_earth/generate.py`:

1. Update imports: remove `child_counts`, add `coupled_counts, count_sizeable_islands, honor_minimums, plan_islands` (keep `allocate_counts, partition_atoms, redistribute_counts`).
2. Level 0: apply coupling + weighted growth:

```python
    counts0 = allocate_counts(
        spec.levels[0], group_sizes**spec.count_coupling
    )
    ...
        parts = partition_atoms(
            mesh, idx, int(counts0[g]), mask.bridges, roughness[0], rng,
            size_variance=spec.size_variance,
        )
```

3. Deeper levels — replace the whole loop body:

```python
    for level in range(1, n_levels):
        prev = level_nodes[level - 1]
        parent_sizes = np.array([len(p["atoms"]) for p in prev], dtype=float)
        level_total = spec.levels[level] * len(prev)
        counts = coupled_counts(
            level_total, parent_sizes, spec.count_coupling,
            spec.count_variance, rng,
        )
        capacities = parent_sizes.astype(int)
        counts = redistribute_counts(counts, capacities)
        # island-rich parents need enough children for one per island where
        # the level total allows; shortfalls degrade to island clustering
        minimums = np.array(
            [
                min(count_sizeable_islands(mesh, p["atoms"], rng), int(capacities[i]))
                for i, p in enumerate(prev)
            ]
        )
        counts = honor_minimums(counts, minimums)
        current: list[dict] = []
        for parent_index, parent in enumerate(prev):
            k = int(counts[parent_index])
            for group_atoms, group_k in plan_islands(
                mesh, parent["atoms"], k, spec.count_coupling, rng
            ):
                parts = partition_atoms(
                    mesh, group_atoms, group_k, None, roughness[level], rng,
                    size_variance=spec.size_variance,
                )
                for atoms in parts:
                    current.append({"atoms": atoms, "parent": parent_index})
        level_nodes.append(current)
```

(Bridges no longer passed below level 0.)
4. Remove `child_counts` from `python/src/mimesis_earth/partition.py` and delete its two tests (`test_child_counts_exact_when_variance_zero`, `test_child_counts_varies_and_positive`) from `python/tests/test_partition.py`.

- [ ] **Step 4: Verify** — the world/partition suites pass, INCLUDING all pre-existing invariants (nesting, populations, determinism, export validity, winding) — these must hold under the new defaults. Expected total: 83 baseline + 2 (T1) + 4 (T2) + 2 (T3) + 3 (plan_islands) + 2 (world) − 2 (removed) = **94 passed**. `test_unit_counts_exact_when_variance_zero` (existing) must still pass — counts semantics at variance 0, coupling default 0.7 with the test's equal-ish specs: NOTE the existing module fixture uses `count_variance=0.0`; totals stay exact by construction, but PER-PARENT counts may now legitimately vary with coupling=0.7. That test asserts only per-level totals (4, 12, 36) — re-read it to confirm; if it asserts per-parent counts anywhere, report DONE_WITH_CONCERNS instead of editing it silently.
- [ ] **Step 5: Measure** — time `generate(WorldSpec())` and `generate(WorldSpec(levels=[8,6,9], resolution=30000, seed=42))` (3 runs each, report medians). The island analysis adds ~2 subgraph scans per parent; if the default world exceeds 2.0s, report timings rather than optimizing ad hoc.
- [ ] **Step 6: Commit** — `"feat: island-aware quotas; exact totals at any count variance"`

---

### Task 5: Frontend sliders, docs, end-to-end

**Files:** Modify `web/index.html`, `web/src/panel.ts`, `web/src/api.ts`, `docs/superpowers/specs/2026-07-26-synthetic-geography-design.md`, `README.md`

- [ ] **Step 1: Panel** — in `web/index.html`, after the `borders` row add:

```html
      <label>sizes <input id="p-sizes" type="range" min="0" max="1" step="0.05" value="0.4" /></label>
      <label>coupling <input id="p-coupling" type="range" min="0" max="1" step="0.05" value="0.7" /></label>
```

In `web/src/api.ts` `Spec` interface add `size_variance: number` and `count_coupling: number`. In `web/src/panel.ts` `readSpec()` add:

```typescript
    size_variance: parseFloat($('p-sizes').value),
    count_coupling: parseFloat($('p-coupling').value),
```

- [ ] **Step 2: Docs** — in the main design doc's decisions table, update the hierarchy-topology row's bridge note to reference the addendum's contiguity contract (one sentence + link to `2026-07-27-partition-realism-addendum.md`). In README's Python API example add `size_variance=0.4, count_coupling=0.7,` lines to the WorldSpec example.
- [ ] **Step 3: Verify** — `cd web && npx tsc --noEmit && npm run build`; `./scripts/build_web.sh`; full python suite (94 passed); live smoke: serve on :8010, POST a spec including the two new fields → 200; POST with `size_variance: 2` → 422 naming the field. Kill the server.
- [ ] **Step 4: Commit** — `"feat: panel sliders for size variance and count coupling; docs"`

---

## Self-review notes (applied)

- Spec addendum coverage: size-coupled counts (T2/T4), weighted growth (T3), island-aware quotas + contiguity contract (T4), new fields + version (T1), panel/docs (T5). `count_variance` semantic change lands in T4 (generate rewiring) — its new guarantee is tested by `test_exact_totals_at_any_variance_and_coupled_counts_vary`.
- Type consistency: `partition_atoms(..., size_variance=0.0)` keyword used by generate() T4 and tests T3; `plan_islands -> list[(ndarray, int)]` consumed by generate() T4; `honor_minimums(counts, minimums)` and `coupled_counts(total, sizes, coupling, variance, rng)` signatures match all call sites.
- Known sharp edges called out inline for the implementer: the `sizes[owner == s].sum()` line in plan_islands, the planar-area proxy in the contiguity test, and the per-parent-counts caveat in Task 4 Step 4.
