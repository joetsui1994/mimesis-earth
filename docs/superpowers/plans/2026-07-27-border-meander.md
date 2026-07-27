# Border Meander Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Borders that meander at all scales by routing unit growth through a smooth per-world cost field (`border_meander`, default 0.5), replacing the straight-bisector look at country/province level.

**Architecture & spec:** docs/superpowers/specs/2026-07-27-border-meander-addendum.md. Touched: `spec.py` (field + version 0.3.0), `partition.py` (`atom_cost` plumbing in `_subgraph`/`partition_atoms`), `generate.py` (field draw + wiring), panel (one slider). Baseline suite: **96 passed**. Branch: `feature/border-meander` off main (96fcafb).

---

### Task 1: Spec field + version bump

**Files:** Modify `python/src/mimesis_earth/spec.py`, `python/tests/test_spec.py`

- [ ] **Step 1: Failing tests** — append to test_spec.py:

```python
def test_border_meander_field():
    spec = WorldSpec()
    assert spec.border_meander == 0.5
    assert spec.generator_version == "0.3.0"
    with pytest.raises(ValidationError):
        WorldSpec(border_meander=1.5)
    assert WorldSpec(border_meander=0.0).border_meander == 0.0
```

Run `cd python && ../.venv/bin/pytest tests/test_spec.py -v` → new test FAILS.

- [ ] **Step 2: Implement** — spec.py: `GENERATOR_VERSION = "0.3.0"`; add after `count_coupling`:

```python
    border_meander: float = Field(default=0.5, ge=0.0, le=1.0)
```

- [ ] **Step 3: Verify** — full suite 97 passed.
- [ ] **Step 4: Commit** — `"feat: border_meander spec field; generator 0.3.0"`

---

### Task 2: atom_cost plumbing in partition

**Files:** Modify `python/src/mimesis_earth/partition.py`, `python/tests/test_partition.py`

- [ ] **Step 1: Failing tests** — append to test_partition.py:

```python
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


def test_partition_cost_field_deterministic(mesh):
    atom_cost = np.exp(np.linspace(-1, 1, len(mesh.points)))
    a = partition_atoms(mesh, np.arange(1000), 4, None, 0.4,
                        np.random.default_rng(101), atom_cost=atom_cost)
    b = partition_atoms(mesh, np.arange(1000), 4, None, 0.4,
                        np.random.default_rng(101), atom_cost=atom_cost)
    for pa, pb in zip(a, b):
        np.testing.assert_array_equal(pa, pb)
```

Run → FAIL (unexpected keyword `atom_cost`).

- [ ] **Step 2: Implement** — partition.py:
  - `_subgraph(mesh, atom_idx, extra_edges, roughness, rng, atom_cost=None)`: after computing mesh-edge weights `w` (BEFORE the roughness jitter, and NOT applied to bridge weights `bw`):

```python
    if atom_cost is not None:
        w = w * np.sqrt(atom_cost[e[m, 0]] * atom_cost[e[m, 1]])
```

  (note `e[m, ...]` are GLOBAL atom indices — atom_cost is a full-length per-atom array. Read the current function; the mesh-edge weight variable and mask names must match what's there.)
  - `partition_atoms(..., size_variance: float = 0.0, atom_cost: np.ndarray | None = None)`: pass `atom_cost` through to its `_subgraph` call. Do NOT pass it anywhere in `plan_islands`/`_island_analysis` (their `_subgraph` calls stay `None`-cost).

- [ ] **Step 3: Verify** — new tests pass; ALL existing tests pass unchanged (default `atom_cost=None` leaves every weight untouched). Full suite 100 passed.
- [ ] **Step 4: Commit** — `"feat: atom_cost edge re-weighting in partitioning"`

---

### Task 3: generate() wiring, panel, docs, e2e

**Files:** Modify `python/src/mimesis_earth/generate.py`, `python/tests/test_world.py`, `web/index.html`, `web/src/api.ts`, `web/src/panel.ts`, `README.md`

- [ ] **Step 1: Failing test** — append to test_world.py:

```python
def test_border_meander_changes_borders_only_when_on():
    base = WorldSpec(levels=[4, 3], n_landmasses=2, resolution=6000, seed=31)
    w_on = generate(base)
    w_on2 = generate(base)
    w_off = generate(base.model_copy(update={"border_meander": 0.0}))
    j_on = json.dumps(w_on.geojson_dict(1), sort_keys=True)
    assert j_on == json.dumps(w_on2.geojson_dict(1), sort_keys=True)
    assert j_on != json.dumps(w_off.geojson_dict(1), sort_keys=True)
```

Run → FAILS (meander not wired; on/off worlds identical).

- [ ] **Step 2: Implement** — generate.py:
  - Import `sphere_noise` from `mimesis_earth.noise`.
  - Immediately after `mask = build_landmask(...)` add (unconditional draw — stream layout must not depend on the knob):

```python
    # phantom-terrain cost field: borders settle on its crests (watersheds).
    # Drawn unconditionally so the rng stream layout is knob-independent.
    terrain = sphere_noise(mesh.points, rng, octaves=6, base_freq=2.0)
    atom_cost = np.exp(1.5 * spec.border_meander * terrain)
```

  - Pass `atom_cost=atom_cost` to every `partition_atoms` call (level 0 and deeper). Note at meander 0 the array is all-ones — multiplying by sqrt(1*1) is a no-op numerically, so no gating needed in partition.
  - Wait — all-ones multiplication is a float no-op but `atom_cost is not None` branches still execute; that's fine and keeps behavior uniform. Do NOT special-case meander==0.
- [ ] **Step 3: Panel + API** — index.html after the `counts` row: `<label>meander <input id="p-meander" type="range" min="0" max="1" step="0.05" value="0.5" /></label>`; api.ts Spec += `border_meander: number`; panel.ts readSpec += `border_meander: parseFloat($('p-meander').value),`. README WorldSpec example gains `border_meander=0.5,`.
- [ ] **Step 4: Verify** — full python suite 101 passed (ALL existing invariants — contiguity, totals, determinism, export validity — must hold under the new default); `cd web && npx tsc --noEmit && npm run build`; `./scripts/build_web.sh`; live smoke on a spare port: POST with border_meander 0 and 1 → 200 both, differing outputs; 1.5 → 422 naming the field. Also report median generate(WorldSpec()) timing (expect ~unchanged; the field draw is milliseconds).
- [ ] **Step 5: Commit** — `"feat: wire border meander through generation; panel slider"`

---

## Self-review notes (applied)

- Addendum coverage: field+version (T1), cost plumbing with land-only/bridge-exempt semantics (T2), per-world field + all-level wiring + unconditional draw (T3), panel (T3). Contiguity-without-repair proof pinned by a dedicated test (T2).
- Type consistency: `partition_atoms(..., atom_cost=None)` keyword used by generate() T3 and tests T2; `_subgraph(..., atom_cost=None)` internal only.
- Sharp edges flagged inline: global-index indexing of atom_cost inside `_subgraph`; the crest-locking test's 2.0-ratio threshold is conservative (measured ~4.6x) but if it flakes, investigate rather than loosen.
