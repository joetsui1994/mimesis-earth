# Terrain Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Elevation-first land: coastlines are the sea-level contour of an explicit per-atom elevation field; borders meander along the same field's ridges; per-unit mean elevation exported. Spec: docs/superpowers/specs/2026-07-27-terrain-phase1-addendum.md.

**Baseline:** suite 101 passed; branch `feature/terrain-phase1` off main (39caf82 + ocean/land color commits). Version → 0.4.0.

---

### Task 1: elevation.py

**Files:** Create `python/src/mimesis_earth/elevation.py`; test `python/tests/test_elevation.py`

- [ ] **Step 1: Failing tests** — create test_elevation.py:

```python
import numpy as np

from mimesis_earth.elevation import build_elevation, ridged_noise
from mimesis_earth.mesh import build_mesh
from mimesis_earth.noise import unit_vectors
from mimesis_earth.spec import WorldSpec


def test_ridged_noise_deterministic_and_crisp():
    pts = unit_vectors(4000, np.random.default_rng(1))
    a = ridged_noise(pts, np.random.default_rng(2))
    b = ridged_noise(pts, np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert abs(a.mean()) < 1e-9 and abs(a.std() - 1.0) < 1e-9
    # ridged fields are asymmetric: sharp crests, broad valleys
    assert abs(np.median(a) - a.mean()) > 0.05


def test_build_elevation_shape_and_determinism():
    mesh = build_mesh(4000, np.random.default_rng(3))
    seeds = unit_vectors(3, np.random.default_rng(4))
    spec = WorldSpec(levels=[3, 3], n_landmasses=3, resolution=4000)
    e1 = build_elevation(mesh, seeds, spec, np.random.default_rng(5))
    e2 = build_elevation(mesh, seeds, spec, np.random.default_rng(5))
    np.testing.assert_array_equal(e1, e2)
    assert e1.shape == (4000,)
    assert np.isfinite(e1).all()


def test_elevation_peaks_near_seeds():
    mesh = build_mesh(4000, np.random.default_rng(6))
    seeds = unit_vectors(2, np.random.default_rng(7))
    spec = WorldSpec(levels=[2, 3], n_landmasses=2, coast_ruggedness=0.3,
                     resolution=4000)
    elev = build_elevation(mesh, seeds, spec, np.random.default_rng(8))
    angle = np.arccos(np.clip(mesh.points @ seeds.T, -1, 1)).min(axis=1)
    near = elev[angle < 0.5].mean()
    far = elev[angle > 1.5].mean()
    assert near > far + 0.5


def test_ruggedness_scales_relief():
    mesh = build_mesh(4000, np.random.default_rng(9))
    seeds = unit_vectors(3, np.random.default_rng(10))
    smooth = build_elevation(
        mesh, seeds,
        WorldSpec(levels=[3, 3], n_landmasses=3, coast_ruggedness=0.0,
                  resolution=4000),
        np.random.default_rng(11),
    )
    rough = build_elevation(
        mesh, seeds,
        WorldSpec(levels=[3, 3], n_landmasses=3, coast_ruggedness=1.0,
                  resolution=4000),
        np.random.default_rng(11),
    )
    # relief = residual variance after removing the continent base trend;
    # proxy: local roughness via edge-difference std
    def edge_std(e):
        return np.abs(e[:-1] - e[1:]).std()

    assert edge_std(rough) > 2.0 * edge_std(smooth)
```

Run `cd python && ../.venv/bin/pytest tests/test_elevation.py -v` → ModuleNotFoundError.

- [ ] **Step 2: Implement** — `python/src/mimesis_earth/elevation.py`:

```python
"""Per-atom elevation on the sphere: continent bases + ridged mountains.

Phase-1 terrain: land is whatever rises above sea level (landmask.py picks
the sea-level quantile); border costs follow the same field's ridges."""

import numpy as np

from mimesis_earth.mesh import Mesh
from mimesis_earth.noise import sphere_noise
from mimesis_earth.spec import WorldSpec


def ridged_noise(
    points: np.ndarray, rng: np.random.Generator,
    octaves: int = 5, base_freq: float = 2.5,
) -> np.ndarray:
    """Mountain-chain noise: sharp crests, broad valleys. Zero-mean, unit-std."""
    n = sphere_noise(points, rng, octaves=octaves, base_freq=base_freq)
    r = (1.0 - np.abs(n) / np.abs(n).max()) ** 1.7
    return (r - r.mean()) / r.std()


def build_elevation(
    mesh: Mesh, seeds: np.ndarray, spec: WorldSpec, rng: np.random.Generator
) -> np.ndarray:
    """Unitless elevation per atom. Continent bumps around the landmass seeds
    preserve the islands/spread semantics; coast_ruggedness scales relief."""
    angle = np.arccos(np.clip(mesh.points @ seeds.T, -1.0, 1.0)).min(axis=1)
    base = -angle
    base = (base - base.mean()) / base.std()
    mountains = ridged_noise(mesh.points, rng)
    detail = sphere_noise(mesh.points, rng, octaves=4, base_freq=6.0)
    return base + spec.coast_ruggedness * (0.9 * mountains + 0.5 * detail)
```

- [ ] **Step 3: Verify** — elevation tests pass; full suite 105 passed.
- [ ] **Step 4: Commit** — `"feat: elevation field (continent bases + ridged mountains)"`

---

### Task 2: landmask rework + version bump

**Files:** Modify `python/src/mimesis_earth/spec.py`, `python/src/mimesis_earth/landmask.py`, `python/src/mimesis_earth/generate.py`, `python/tests/test_spec.py`, `python/tests/test_landmask.py`

- [ ] **Step 1: Failing tests** — spec.py test: update the "0.3.0" pin in test_spec.py to expect "0.4.0" (run → fails). test_landmask.py: `build_landmask` gains a required `elevation` argument — update EVERY call site in test_landmask.py to build it explicitly:

```python
# helper at top of test_landmask.py (after imports):
from mimesis_earth.elevation import build_elevation
from mimesis_earth.noise import sample_vmf, unit_vectors


def _mask(mesh, spec, seed):
    rng = np.random.default_rng(seed)
    return build_landmask(mesh, spec, rng)
```

Wait — seeds are drawn INSIDE build_landmask (vMF retry loop), so elevation must be built inside too. DESIGN DECISION (follow it): `build_landmask(mesh, spec, rng)` keeps its signature; it builds elevation internally per retry attempt (seeds redraw → elevation rebuild) and RETURNS it: `LandMask` gains an `elevation: np.ndarray` field. Test call sites stay unchanged; only assertions about scoring internals (none exist) would change. Add one new test:

```python
def test_land_is_high_ground(mesh):
    spec = WorldSpec(levels=[3, 3], n_landmasses=3, resolution=4000)
    mask = build_landmask(mesh, spec, np.random.default_rng(60))
    assert mask.elevation.shape == (len(mesh.points),)
    # land sits above sea: mean land elevation > mean sea elevation
    assert mask.elevation[mask.land].mean() > mask.elevation[~mask.land].mean() + 0.5
```

Run: new test fails (no elevation attr); version test fails.

- [ ] **Step 2: Implement**
  - spec.py: `GENERATOR_VERSION = "0.4.0"`.
  - landmask.py: import `build_elevation`; add `elevation: np.ndarray` to the `LandMask` dataclass; in `build_landmask`'s retry loop, after drawing `seeds`, replace the score construction:

```python
        elevation = build_elevation(mesh, seeds, spec, rng)
        angle = np.arccos(np.clip(mesh.points @ seeds.T, -1.0, 1.0))
        nearest = angle.argmin(axis=1)
        score = elevation
```

    (the old z-scored -angle + coast noise lines are deleted; kernels, budget fill by top score, group assignment, `present` check, bridges, retry loop all stay verbatim). Return `LandMask(land=land, group=group, bridges=bridges, elevation=elevation)`.
  - generate.py: no changes in this task (next task rewires atom_cost).

- [ ] **Step 3: Verify** — full suite must pass: pay special attention to the landmask regression tests (`test_many_landmasses_low_spread`, `test_many_landmasses_low_land_fraction`, `test_land_fraction_respected`, `test_bridges_connect_within_groups`, determinism) — the guarantee kernels are elevation-independent so they must hold; if any fails, investigate (likely the retry loop needs elevation rebuilt per attempt — it must be inside the loop). Expected 106 passed.
- [ ] **Step 4: Commit** — `"feat: land is high ground - elevation-based landmask; generator 0.4.0"`

---

### Task 3: cost coherence, elevation export, docs, e2e

**Files:** Modify `python/src/mimesis_earth/generate.py`, `python/src/mimesis_earth/world.py`, `python/tests/test_world.py`, `README.md`

- [ ] **Step 1: Failing tests** — append to test_world.py:

```python
def test_elevation_exported_and_coherent(tmp_path):
    spec = WorldSpec(levels=[4, 3], n_landmasses=2, coast_ruggedness=0.8,
                     border_meander=1.0, resolution=8000, seed=41)
    cap: dict = {}
    world = generate(spec, _capture=cap)
    for u in world.units:
        assert isinstance(u.elevation_m, int)
        assert -100 <= u.elevation_m <= 5000
    assert max(u.elevation_m for u in world.units_at(1)) > 300
    out = tmp_path / "elev"
    world.to_geojson(out)
    fc = json.loads((out / "level0_country.geojson").read_text())
    assert all("elevation_m" in f["properties"] for f in fc["features"])
    world.to_csv(tmp_path / "units.csv")
    assert "elevation_m" in (tmp_path / "units.csv").read_text().splitlines()[0]
    # coherence: with meander=1, country borders sit on high ground
    mesh, nodes = cap["mesh"], cap["level_nodes"]
    elevation = cap["elevation"]
    label = np.full(len(mesh.points), -1)
    for i, node in enumerate(nodes[0]):
        label[node["atoms"]] = i
    e = mesh.edges
    border = (label[e[:, 0]] >= 0) & (label[e[:, 1]] >= 0) & (
        label[e[:, 0]] != label[e[:, 1]]
    )
    border_atoms = np.unique(np.concatenate([e[border, 0], e[border, 1]]))
    land_atoms = np.flatnonzero(label >= 0)
    assert elevation[border_atoms].mean() > elevation[land_atoms].mean() + 0.2
```

Run → fails (Unit has no elevation_m; cap has no "elevation" key).

- [ ] **Step 2: Implement**
  - generate.py: delete the phantom `terrain = sphere_noise(...)` draw; use the landmask's field: `atom_cost = np.exp(1.5 * spec.border_meander * ((mask.elevation - mask.elevation.mean()) / mask.elevation.std()))`. Add `"elevation"` to the `_capture` dict (`mask.elevation`). Compute per-unit mean elevation in meters: after `sea_level = np.quantile(mask.elevation, 1.0 - spec.land_fraction)`... careful: the landmask's land set is kernel-adjusted, not a pure quantile — define `sea_level` as the minimum elevation over land atoms minus epsilon? DESIGN DECISION (follow it): `sea_level = np.quantile(mask.elevation, 1.0 - spec.land_fraction)` (the nominal contour) and scale: `scale = 4500.0 / max(mask.elevation.max() - sea_level, 1e-9)`; per-unit `elevation_m = int(round(scale * ((area-weighted mean of mask.elevation over unit atoms) - sea_level)))`. Clamp to >= -100 for kernel-forced low atoms: `max(-100, ...)`. Pass into Unit.
  - world.py: `Unit` gains `elevation_m: int = 0`; `_feature` properties gain `"elevation_m": u.elevation_m`; CSV column appended at END (after landmass); gdf records gain it.
  - README: WorldSpec example unchanged; add `elevation_m` to the one-line attribute description if present.
- [ ] **Step 3: Verify** — full suite (expect 107 passed; all export-path and invariant tests must stay green — test_exports asserts the csv header PREFIX so an appended column passes). `cd web && npx tsc --noEmit && npm run build` (frontend untouched but keep the gate); `./scripts/build_web.sh`; live smoke on spare port: generate default world → 200 and features carry elevation_m; timing: median generate(WorldSpec()) of 3 — expect ≤1.3s.
- [ ] **Step 4: Commit** — `"feat: elevation-driven border costs and per-unit elevation export"`

---

## Self-review notes (applied)

- Addendum coverage: elevation module (T1), elevation-based land + version (T2), cost coherence + exports (T3). Ruggedness reinterpretation covered by T1's relief test + T2 keeping the ruggedness-sensitive landmask regressions.
- Sharp edges flagged: elevation must be rebuilt per retry attempt inside build_landmask's loop; sea_level uses the nominal quantile while land uses kernel-adjusted membership — the -100m clamp absorbs the mismatch.
- Deferred explicitly: rivers, tectonics, rendering, population coupling.
