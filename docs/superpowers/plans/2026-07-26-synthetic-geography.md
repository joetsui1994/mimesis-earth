# Synthetic Geography Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `mimesis-earth`: a Python package that deterministically generates synthetic nested administrative geographies (real WGS84 coordinates, strictly nested polygons, demographics), plus a FastAPI server and a minimalist paper-globe web frontend.

**Architecture:** Everything is built from a fine mesh of "atoms" (jittered Fibonacci points on the unit sphere + their spherical Voronoi cells). A land mask classifies atoms into landmass groups; competitive flood-fill (noisy Dijkstra) recursively partitions atoms into countries → provinces → districts, guaranteeing strict nesting by construction. Unit polygons are unions of atom cells converted to lon/lat (with antimeridian/pole handling). A stateless FastAPI server exposes `POST /api/generate`; a Vite + TypeScript + d3-geo canvas frontend renders a draggable/zoomable orthographic globe (spacebar = new world).

**Tech Stack:** Python ≥3.10, numpy, scipy, shapely 2, pydantic 2, FastAPI, uvicorn; optional geopandas. Frontend: Vite, TypeScript, d3-geo, fflate. Tests: pytest, httpx.

**Spec:** `docs/superpowers/specs/2026-07-26-synthetic-geography-design.md` — read it first.

**Working conventions for all tasks:**
- Python work happens in `python/`; run tests as `cd python && ../.venv/bin/pytest tests/... -v`.
- The venv lives at repo root: `.venv` (created in Task 1).
- Commit after every task (steps include the commands).
- Determinism rule: all randomness flows from a single `numpy.random.default_rng(seed)` created in `generate()`; never call `np.random.*` module functions or `random`.

---

## File structure (final state)

```
python/
  pyproject.toml
  src/mimesis_earth/
    __init__.py        # public API: WorldSpec, World, generate
    spec.py            # WorldSpec (pydantic) + validation
    mesh.py            # Fibonacci lattice, spherical Voronoi, adjacency graph
    noise.py           # spectral noise on the sphere, von Mises–Fisher sampling
    landmask.py        # landmass seeds, land classification, island bridges
    partition.py       # seed picking, noisy-Dijkstra flood-fill partitioning
    geometry.py        # atom cells → lon/lat polygons; antimeridian & poles
    naming.py          # seeded syllable name generator
    attributes.py      # population density field, sum-preserving rounding
    world.py           # Unit + World (gdf/to_geojson/to_csv/geojson_dict)
    generate.py        # pipeline orchestrator
    server.py          # FastAPI app (POST /api/generate + static frontend)
    cli.py             # `mimesis-earth serve`
    webapp/            # built frontend (copied by scripts/build_web.sh; gitignored)
  tests/
    test_spec.py  test_mesh.py  test_noise.py  test_landmask.py
    test_partition.py  test_geometry.py  test_naming.py  test_attributes.py
    test_world.py  test_server.py
web/
  index.html  package.json  tsconfig.json  vite.config.ts
  src/
    main.ts  api.ts  globe.ts  panel.ts  inspect.ts  exporter.ts  style.css
scripts/
  build_web.sh         # npm build + copy dist into python package
```

---

### Task 1: Python package scaffolding

**Files:**
- Create: `python/pyproject.toml`
- Create: `python/src/mimesis_earth/__init__.py`
- Create: `python/tests/test_import.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create pyproject**

`python/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mimesis-earth"
version = "0.1.0"
description = "Rapid synthetic geography generator: nested admin units on a real sphere"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "shapely>=2.0",
    "pydantic>=2.5",
]

[project.optional-dependencies]
server = ["fastapi>=0.110", "uvicorn>=0.27"]
geo = ["geopandas>=0.14"]
dev = [
    "pytest>=8",
    "httpx>=0.27",
    "fastapi>=0.110",
    "uvicorn>=0.27",
    "geopandas>=0.14",
]

[project.scripts]
mimesis-earth = "mimesis_earth.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.hatch.build.targets.wheel]
packages = ["src/mimesis_earth"]
```

- [ ] **Step 2: Create package init and a sanity test**

`python/src/mimesis_earth/__init__.py`:

```python
"""mimesis-earth: rapid synthetic geography generator."""

__version__ = "0.1.0"
```

`python/tests/test_import.py`:

```python
def test_import():
    import mimesis_earth

    assert mimesis_earth.__version__ == "0.1.0"
```

- [ ] **Step 3: Create venv, install editable, run test**

```bash
cd /Users/user/Documents/work/mimesis-earth
python3 -m venv .venv
.venv/bin/pip install -e './python[dev]'
cd python && ../.venv/bin/pytest tests/test_import.py -v
```

Expected: 1 passed. (If python3 is 3.14 and a dependency lacks wheels, install Python 3.12 via `brew install python@3.12` and use `python3.12 -m venv .venv` instead — note which interpreter you used in the commit message.)

- [ ] **Step 4: Update .gitignore**

Append to `.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
python/src/mimesis_earth/webapp/
web/node_modules/
web/dist/
.pytest_cache/
```

- [ ] **Step 5: Commit**

```bash
git add python .gitignore
git commit -m "chore: scaffold python package"
```

---

### Task 2: WorldSpec (pydantic) with validation

**Files:**
- Create: `python/src/mimesis_earth/spec.py`
- Modify: `python/src/mimesis_earth/__init__.py`
- Test: `python/tests/test_spec.py`

- [ ] **Step 1: Write failing tests**

`python/tests/test_spec.py`:

```python
import math

import pytest
from pydantic import ValidationError

from mimesis_earth.spec import GENERATOR_VERSION, WorldSpec


def test_defaults_are_valid():
    spec = WorldSpec()
    assert spec.levels == [6, 5, 6]
    assert spec.level_names == ["country", "province", "district"]
    assert spec.seed == 0
    assert spec.generator_version == GENERATOR_VERSION


def test_level_names_default_matches_levels_length():
    spec = WorldSpec(levels=[4, 3])
    assert spec.level_names == ["country", "province"]


def test_rejects_bad_spread():
    with pytest.raises(ValidationError):
        WorldSpec(spread=1.5)


def test_rejects_fewer_countries_than_landmasses():
    with pytest.raises(ValidationError, match="n_landmasses"):
        WorldSpec(levels=[2, 3], n_landmasses=5)


def test_rejects_resolution_too_low_for_leaf_count():
    with pytest.raises(ValidationError, match="resolution"):
        WorldSpec(levels=[20, 20, 20], resolution=2000)


def test_rejects_mismatched_border_roughness_list():
    with pytest.raises(ValidationError, match="border_roughness"):
        WorldSpec(levels=[4, 4], border_roughness=[0.1, 0.2, 0.3])


def test_rejects_mismatched_level_names():
    with pytest.raises(ValidationError, match="level_names"):
        WorldSpec(levels=[4, 4], level_names=["only-one"])


def test_json_roundtrip():
    spec = WorldSpec(levels=[8, 6, 9], seed=42, spread=0.6)
    spec2 = WorldSpec.model_validate_json(spec.model_dump_json())
    assert spec2 == spec


def test_leaf_count_helper():
    assert WorldSpec(levels=[8, 6, 9]).max_leaf_count() == math.prod([8, 6, 9])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && ../.venv/bin/pytest tests/test_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mimesis_earth.spec'`

- [ ] **Step 3: Implement WorldSpec**

`python/src/mimesis_earth/spec.py`:

```python
"""World specification: the full parameter set that (with a seed) defines a world."""

import math
from typing import Optional, Union

from pydantic import BaseModel, Field, model_validator

GENERATOR_VERSION = "0.1.0"

_DEFAULT_LEVEL_NAMES = ["country", "province", "district", "ward", "block"]

# Each leaf unit needs at least this many atoms to have a drawable polygon.
MIN_ATOMS_PER_LEAF = 8


class WorldSpec(BaseModel):
    levels: list[int] = Field(default=[6, 5, 6], min_length=1, max_length=5)
    level_names: Optional[list[str]] = None
    n_landmasses: int = Field(default=3, ge=1, le=64)
    spread: float = Field(default=0.5, ge=0.0, le=1.0)
    land_fraction: float = Field(default=0.3, gt=0.0, lt=0.8)
    coast_ruggedness: float = Field(default=0.5, ge=0.0, le=1.0)
    border_roughness: Union[float, list[float]] = 0.4
    count_variance: float = Field(default=0.2, ge=0.0, le=1.0)
    total_population: int = Field(default=50_000_000, gt=0)
    resolution: int = Field(default=20_000, ge=2_000, le=200_000)
    seed: int = 0
    generator_version: str = GENERATOR_VERSION

    def max_leaf_count(self) -> int:
        return math.prod(self.levels)

    def border_roughness_per_level(self) -> list[float]:
        if isinstance(self.border_roughness, list):
            return self.border_roughness
        return [self.border_roughness] * len(self.levels)

    @model_validator(mode="after")
    def _validate(self) -> "WorldSpec":
        if any(c < 1 for c in self.levels):
            raise ValueError("every entry in levels must be >= 1")
        if self.level_names is None:
            self.level_names = _DEFAULT_LEVEL_NAMES[: len(self.levels)]
        if len(self.level_names) != len(self.levels):
            raise ValueError(
                f"level_names has {len(self.level_names)} entries but levels has "
                f"{len(self.levels)}"
            )
        if self.levels[0] < self.n_landmasses:
            raise ValueError(
                f"levels[0]={self.levels[0]} must be >= n_landmasses="
                f"{self.n_landmasses} (each landmass needs at least one "
                f"top-level unit); lower n_landmasses or raise levels[0]"
            )
        if isinstance(self.border_roughness, list) and len(
            self.border_roughness
        ) != len(self.levels):
            raise ValueError("border_roughness list must match levels length")
        expected_land_atoms = self.resolution * self.land_fraction
        if self.max_leaf_count() * MIN_ATOMS_PER_LEAF > expected_land_atoms:
            need = int(self.max_leaf_count() * MIN_ATOMS_PER_LEAF / self.land_fraction)
            raise ValueError(
                f"resolution={self.resolution} is too low for "
                f"~{self.max_leaf_count()} leaf units at land_fraction="
                f"{self.land_fraction}; raise resolution to >= {need} or reduce "
                f"level counts"
            )
        return self
```

Update `python/src/mimesis_earth/__init__.py`:

```python
"""mimesis-earth: rapid synthetic geography generator."""

from mimesis_earth.spec import WorldSpec

__version__ = "0.1.0"

__all__ = ["WorldSpec", "__version__"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && ../.venv/bin/pytest tests/test_spec.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "feat: WorldSpec parameter model with validation"
```

---

### Task 3: Sphere mesh (atoms)

**Files:**
- Create: `python/src/mimesis_earth/mesh.py`
- Test: `python/tests/test_mesh.py`

- [ ] **Step 1: Write failing tests**

`python/tests/test_mesh.py`:

```python
import numpy as np

from mimesis_earth.mesh import Mesh, build_mesh, fibonacci_points


def test_fibonacci_points_on_unit_sphere():
    rng = np.random.default_rng(1)
    pts = fibonacci_points(500, rng)
    assert pts.shape == (500, 3)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-12)


def test_fibonacci_points_deterministic():
    a = fibonacci_points(200, np.random.default_rng(7))
    b = fibonacci_points(200, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_build_mesh_structure():
    mesh = build_mesh(500, np.random.default_rng(3))
    assert isinstance(mesh, Mesh)
    assert mesh.points.shape == (500, 3)
    assert len(mesh.regions) == 500
    # every region references valid voronoi vertices
    for region in mesh.regions:
        assert len(region) >= 3
        assert max(region) < len(mesh.vertices)
    # areas tile the sphere
    np.testing.assert_allclose(mesh.areas.sum(), 4 * np.pi, rtol=1e-6)
    # adjacency is symmetric, no self-loops, connected-ish degree
    assert (mesh.adjacency != mesh.adjacency.T).nnz == 0
    assert mesh.adjacency.diagonal().sum() == 0
    degrees = np.diff(mesh.adjacency.indptr)
    assert degrees.min() >= 3


def test_edges_are_unique_and_undirected():
    mesh = build_mesh(300, np.random.default_rng(5))
    e = mesh.edges
    assert (e[:, 0] < e[:, 1]).all()
    assert len(np.unique(e, axis=0)) == len(e)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && ../.venv/bin/pytest tests/test_mesh.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement mesh**

`python/src/mimesis_earth/mesh.py`:

```python
"""The atom mesh: jittered Fibonacci points and their spherical Voronoi cells."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull, SphericalVoronoi


@dataclass
class Mesh:
    points: np.ndarray  # (n, 3) atom centers on the unit sphere
    vertices: np.ndarray  # (m, 3) Voronoi vertices on the unit sphere
    regions: list  # per atom: ordered Voronoi vertex indices (closed ring)
    edges: np.ndarray  # (e, 2) unique undirected atom adjacencies, col0 < col1
    adjacency: csr_matrix  # symmetric (n, n), weights = geodesic edge length
    areas: np.ndarray  # (n,) spherical cell areas on the unit sphere


def fibonacci_points(n: int, rng: np.random.Generator, jitter: float = 0.35) -> np.ndarray:
    i = np.arange(n)
    golden = (1 + 5**0.5) / 2
    theta = 2 * np.pi * i / golden
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(np.clip(1 - z**2, 0, None))
    pts = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    # tangential jitter scaled to typical point spacing, then renormalize
    spacing = np.sqrt(4 * np.pi / n)
    noise = rng.normal(scale=jitter * spacing, size=(n, 3))
    noise -= pts * np.sum(noise * pts, axis=1, keepdims=True)
    pts = pts + noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def build_mesh(n: int, rng: np.random.Generator) -> Mesh:
    pts = fibonacci_points(n, rng)
    sv = SphericalVoronoi(pts, radius=1.0)
    sv.sort_vertices_of_regions()
    hull = ConvexHull(pts)  # Delaunay triangulation on the sphere
    tri = hull.simplices
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.arccos(np.clip(np.sum(pts[e[:, 0]] * pts[e[:, 1]], axis=1), -1.0, 1.0))
    adjacency = csr_matrix(
        (
            np.concatenate([w, w]),
            (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])),
        ),
        shape=(n, n),
    )
    return Mesh(
        points=pts,
        vertices=sv.vertices,
        regions=sv.regions,
        edges=e,
        adjacency=adjacency,
        areas=sv.calculate_areas(),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && ../.venv/bin/pytest tests/test_mesh.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "feat: atom mesh (fibonacci lattice + spherical voronoi + adjacency)"
```

---

### Task 4: Noise and von Mises–Fisher sampling

**Files:**
- Create: `python/src/mimesis_earth/noise.py`
- Test: `python/tests/test_noise.py`

- [ ] **Step 1: Write failing tests**

`python/tests/test_noise.py`:

```python
import numpy as np

from mimesis_earth.noise import sample_vmf, sphere_noise, unit_vectors


def test_unit_vectors():
    v = unit_vectors(100, np.random.default_rng(1))
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-12)


def test_sphere_noise_normalized_and_deterministic():
    pts = unit_vectors(2000, np.random.default_rng(2))
    a = sphere_noise(pts, np.random.default_rng(3))
    b = sphere_noise(pts, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)
    assert abs(a.mean()) < 1e-9
    assert abs(a.std() - 1.0) < 1e-9


def test_sphere_noise_is_smooth():
    # nearby points must have similar noise values
    rng = np.random.default_rng(4)
    base = unit_vectors(500, rng)
    eps = base + rng.normal(scale=1e-4, size=base.shape)
    eps /= np.linalg.norm(eps, axis=1, keepdims=True)
    na = sphere_noise(np.vstack([base, eps]), np.random.default_rng(5))
    diff = np.abs(na[:500] - na[500:])
    assert diff.max() < 0.05


def test_vmf_concentration():
    mu = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(6)
    tight = sample_vmf(mu, kappa=200.0, n=500, rng=rng)
    loose = sample_vmf(mu, kappa=1e-9, n=500, rng=np.random.default_rng(7))
    np.testing.assert_allclose(np.linalg.norm(tight, axis=1), 1.0, atol=1e-9)
    # tight samples hug mu; loose samples cover the sphere
    assert (tight @ mu).min() > 0.8
    assert (loose @ mu).min() < -0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && ../.venv/bin/pytest tests/test_noise.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement noise**

`python/src/mimesis_earth/noise.py`:

```python
"""Seeded random fields on the sphere: spectral noise and vMF direction sampling."""

import numpy as np


def unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def sphere_noise(
    points: np.ndarray,
    rng: np.random.Generator,
    octaves: int = 4,
    waves_per_octave: int = 6,
    base_freq: float = 3.0,
) -> np.ndarray:
    """Smooth zero-mean unit-variance noise sampled at `points` on the unit sphere.

    Sum of random plane waves; smooth by construction, fully determined by rng.
    """
    out = np.zeros(len(points))
    for octave in range(octaves):
        freq = base_freq * 2.0**octave
        amp = 0.55**octave
        dirs = unit_vectors(waves_per_octave, rng)
        phases = rng.uniform(0, 2 * np.pi, waves_per_octave)
        for u, phase in zip(dirs, phases):
            out += amp * np.cos(freq * np.pi * (points @ u) + phase)
    return (out - out.mean()) / out.std()


def sample_vmf(
    mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample n unit vectors from a von Mises–Fisher distribution around mu.

    kappa -> 0 is uniform on the sphere; large kappa concentrates near mu.
    """
    if kappa < 1e-6:
        return unit_vectors(n, rng)
    u = rng.uniform(size=n)
    w = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    angle = rng.uniform(0, 2 * np.pi, n)
    helper = (
        np.array([1.0, 0.0, 0.0]) if abs(mu[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    e1 = np.cross(mu, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(mu, e1)
    r = np.sqrt(np.clip(1.0 - w**2, 0.0, None))
    return (
        w[:, None] * mu
        + (r * np.cos(angle))[:, None] * e1
        + (r * np.sin(angle))[:, None] * e2
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && ../.venv/bin/pytest tests/test_noise.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "feat: spectral sphere noise and von Mises-Fisher sampling"
```

---

### Task 5: Land mask

**Files:**
- Create: `python/src/mimesis_earth/landmask.py`
- Test: `python/tests/test_landmask.py`

- [ ] **Step 1: Write failing tests**

`python/tests/test_landmask.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && ../.venv/bin/pytest tests/test_landmask.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement landmask**

`python/src/mimesis_earth/landmask.py`:

```python
"""Classify atoms into sea and landmass groups; bridge islands within a group."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from mimesis_earth.mesh import Mesh
from mimesis_earth.noise import sample_vmf, sphere_noise, unit_vectors
from mimesis_earth.spec import WorldSpec


@dataclass
class LandMask:
    land: np.ndarray  # (n,) bool
    group: np.ndarray  # (n,) int landmass index; -1 for sea
    bridges: np.ndarray  # (b, 2) atom index pairs linking islands within a group


def build_landmask(mesh: Mesh, spec: WorldSpec, rng: np.random.Generator) -> LandMask:
    # spread=0 -> tightly clustered landmass seeds; spread=1 -> uniform
    kappa = 100.0 * (1.0 - spec.spread) ** 4
    for _ in range(10):
        center = unit_vectors(1, rng)[0]
        seeds = sample_vmf(center, kappa, spec.n_landmasses, rng)
        angle = np.arccos(np.clip(mesh.points @ seeds.T, -1.0, 1.0))  # (n, K)
        nearest = angle.argmin(axis=1)
        score = -angle.min(axis=1)
        score = (score - score.mean()) / score.std()
        score = score + 1.2 * spec.coast_ruggedness * sphere_noise(mesh.points, rng)
        threshold = np.quantile(score, 1.0 - spec.land_fraction)
        land = score > threshold
        group = np.where(land, nearest, -1)
        present = np.unique(group[land])
        if len(present) == spec.n_landmasses:
            bridges = _bridge_islands(mesh, group, spec.n_landmasses)
            return LandMask(land=land, group=group, bridges=bridges)
    raise RuntimeError(
        "could not place all landmasses; try a different seed or raise land_fraction"
    )


def _bridge_islands(mesh: Mesh, group: np.ndarray, n_groups: int) -> np.ndarray:
    """Within each landmass group, link secondary islands to the largest island so
    flood-fill partitioning can cross the (small) sea gaps inside an archipelago."""
    bridges = []
    for g in range(n_groups):
        idx = np.flatnonzero(group == g)
        if len(idx) == 0:
            continue
        sub = mesh.adjacency[idx][:, idx]
        n_comp, labels = connected_components(sub, directed=False)
        if n_comp <= 1:
            continue
        sizes = np.bincount(labels)
        main_label = int(sizes.argmax())
        main = np.flatnonzero(labels == main_label)
        tree = cKDTree(mesh.points[idx[main]])
        for c in range(n_comp):
            if c == main_label:
                continue
            comp = np.flatnonzero(labels == c)
            dist, j = tree.query(mesh.points[idx[comp]])
            k = int(dist.argmin())
            bridges.append((int(idx[comp[k]]), int(idx[main[j[k]]])))
    return np.array(bridges, dtype=int).reshape(-1, 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && ../.venv/bin/pytest tests/test_landmask.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "feat: land mask with landmass groups and island bridges"
```

---

### Task 6: Partitioning (competitive flood-fill)

**Files:**
- Create: `python/src/mimesis_earth/partition.py`
- Test: `python/tests/test_partition.py`

- [ ] **Step 1: Write failing tests**

`python/tests/test_partition.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && ../.venv/bin/pytest tests/test_partition.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement partition**

`python/src/mimesis_earth/partition.py`:

```python
"""Competitive flood-fill partitioning of atoms over the adjacency graph."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from mimesis_earth.mesh import Mesh

BRIDGE_COST_FACTOR = 3.0


def pick_seeds(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Farthest-point sampling: k well-spaced local indices into `points`."""
    first = int(rng.integers(len(points)))
    chosen = [first]
    d = np.linalg.norm(points - points[first], axis=1)
    while len(chosen) < k:
        nxt = int(d.argmax())
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen)


def _subgraph(
    mesh: Mesh,
    atom_idx: np.ndarray,
    extra_edges: Optional[np.ndarray],
    roughness: float,
    rng: np.random.Generator,
) -> csr_matrix:
    pos = -np.ones(len(mesh.points), dtype=int)
    pos[atom_idx] = np.arange(len(atom_idx))
    e = mesh.edges
    m = (pos[e[:, 0]] >= 0) & (pos[e[:, 1]] >= 0)
    local = np.column_stack([pos[e[m, 0]], pos[e[m, 1]]])
    w = np.arccos(
        np.clip(np.sum(mesh.points[e[m, 0]] * mesh.points[e[m, 1]], axis=1), -1, 1)
    )
    if extra_edges is not None and len(extra_edges) > 0:
        bm = (pos[extra_edges[:, 0]] >= 0) & (pos[extra_edges[:, 1]] >= 0)
        be = extra_edges[bm]
        if len(be) > 0:
            bw = BRIDGE_COST_FACTOR * np.arccos(
                np.clip(
                    np.sum(mesh.points[be[:, 0]] * mesh.points[be[:, 1]], axis=1),
                    -1,
                    1,
                )
            )
            local = np.vstack([local, np.column_stack([pos[be[:, 0]], pos[be[:, 1]]])])
            w = np.concatenate([w, bw])
    # symmetric per-edge noise makes borders wiggly; same draw for both directions
    w = w * (1.0 + roughness * rng.uniform(0.0, 3.0, size=len(w)))
    n = len(atom_idx)
    return csr_matrix(
        (
            np.concatenate([w, w]),
            (
                np.concatenate([local[:, 0], local[:, 1]]),
                np.concatenate([local[:, 1], local[:, 0]]),
            ),
        ),
        shape=(n, n),
    )


def partition_atoms(
    mesh: Mesh,
    atom_idx: np.ndarray,
    k: int,
    extra_edges: Optional[np.ndarray],
    roughness: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Split atom_idx into k non-empty contiguous parts. Returns global index arrays."""
    atom_idx = np.asarray(atom_idx)
    assert 1 <= k <= len(atom_idx), f"cannot cut {len(atom_idx)} atoms into {k} parts"
    if k == 1:
        return [atom_idx]
    adj = _subgraph(mesh, atom_idx, extra_edges, roughness, rng)
    seeds = pick_seeds(mesh.points[atom_idx], k, rng)
    dist = dijkstra(adj, directed=False, indices=seeds)
    labels = np.asarray(dist).argmin(axis=0)
    # atoms unreachable from every seed (disconnected slivers with no bridge):
    # attach to the nearest seed by straight-line distance
    unreachable = ~np.isfinite(np.asarray(dist).min(axis=0))
    if unreachable.any():
        pts = mesh.points[atom_idx]
        chord = np.linalg.norm(
            pts[unreachable][:, None, :] - pts[seeds][None, :, :], axis=2
        )
        labels[unreachable] = chord.argmin(axis=1)
    return [atom_idx[labels == i] for i in range(k)]


def child_counts(
    mean: int, n_parents: int, variance: float, rng: np.random.Generator
) -> np.ndarray:
    """How many children each parent gets. variance=0 -> exactly `mean` each."""
    if variance <= 0:
        return np.full(n_parents, mean, dtype=int)
    counts = np.round(rng.normal(mean, variance * mean, n_parents)).astype(int)
    return np.clip(counts, 1, None)


def allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    """Split `total` units among groups proportionally to weights, each >= 1."""
    assert total >= len(weights)
    share = weights / weights.sum()
    counts = np.maximum(1, np.floor(share * total)).astype(int)
    while counts.sum() > total:
        counts[counts.argmax()] -= 1
    remainder = share * total - counts
    while counts.sum() < total:
        i = int(remainder.argmax())
        counts[i] += 1
        remainder[i] -= 1.0
    return counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && ../.venv/bin/pytest tests/test_partition.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "feat: noisy flood-fill partitioning, seed picking, count allocation"
```

---

### Task 7: Geometry (cells → lon/lat polygons, antimeridian & poles)

**Files:**
- Create: `python/src/mimesis_earth/geometry.py`
- Test: `python/tests/test_geometry.py`

- [ ] **Step 1: Write failing tests**

`python/tests/test_geometry.py`:

```python
import numpy as np
from shapely.geometry import Point

from mimesis_earth.geometry import (
    R_EARTH_KM,
    atoms_polygon,
    cell_polygon,
    xyz_to_lonlat,
)
from mimesis_earth.mesh import build_mesh


def lonlat_to_xyz(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def test_xyz_to_lonlat_roundtrip():
    lon, lat = xyz_to_lonlat(np.array([lonlat_to_xyz(45.0, 30.0)]))
    np.testing.assert_allclose([lon[0], lat[0]], [45.0, 30.0], atol=1e-9)


def test_all_cells_valid_and_in_range():
    mesh = build_mesh(1000, np.random.default_rng(30))
    for i in range(len(mesh.points)):
        poly = cell_polygon(mesh.vertices[mesh.regions[i]])
        assert poly.is_valid, f"cell {i} invalid"
        assert not poly.is_empty
        minx, miny, maxx, maxy = poly.bounds
        assert -180.0001 <= minx <= maxx <= 180.0001
        assert -90.0001 <= miny <= maxy <= 90.0001


def test_pole_cell_contains_pole():
    # hexagon of vertices at lat 85 -> cell encloses the north pole
    lons = np.array([0.0, 60.0, 120.0, 180.0, -120.0, -60.0])
    verts = np.stack([lonlat_to_xyz(lo, 85.0) for lo in lons])
    poly = cell_polygon(verts)
    assert poly.is_valid
    assert poly.contains(Point(10.0, 89.5))


def test_antimeridian_cell_split():
    # small square straddling lon=180
    corners = [(179.0, 10.0), (-179.0, 10.0), (-179.0, 12.0), (179.0, 12.0)]
    verts = np.stack([lonlat_to_xyz(lo, la) for lo, la in corners])
    poly = cell_polygon(verts)
    assert poly.is_valid
    assert poly.geom_type == "MultiPolygon"
    assert poly.bounds[0] >= -180.0001 and poly.bounds[2] <= 180.0001


def test_atoms_polygon_union():
    mesh = build_mesh(1000, np.random.default_rng(31))
    ids = np.arange(40)
    merged = atoms_polygon(mesh, ids)
    assert merged.is_valid
    # union area equals sum of parts (no double-counting, no gaps)
    parts = sum(
        cell_polygon(mesh.vertices[mesh.regions[i]]).area for i in ids
    )
    np.testing.assert_allclose(merged.area, parts, rtol=1e-6)


def test_earth_radius_constant():
    assert R_EARTH_KM == 6371.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && ../.venv/bin/pytest tests/test_geometry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement geometry**

`python/src/mimesis_earth/geometry.py`:

```python
"""Convert spherical Voronoi cells to WGS84 lon/lat shapely polygons.

Handles the two classic sphere-to-plane traps:
- antimeridian: cells straddling lon=+-180 are split into a MultiPolygon
- poles: cells enclosing a pole get an explicit closure ring over the pole
"""

import numpy as np
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

R_EARTH_KM = 6371.0


def xyz_to_lonlat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points)
    lon = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(points[:, 2], -1.0, 1.0)))
    return lon, lat


def _unwrapped_ring(verts3d: np.ndarray) -> list[tuple[float, float]]:
    """Ring in (lon, lat) with longitudes unwrapped into a continuous sequence.
    Cells that enclose a pole get extra points closing the ring over the pole."""
    lon, lat = xyz_to_lonlat(verts3d)
    deltas = (np.diff(lon) + 180.0) % 360.0 - 180.0
    ulon = np.concatenate([[lon[0]], lon[0] + np.cumsum(deltas)])
    closing = ((lon[0] - ulon[-1]) + 180.0) % 360.0 - 180.0
    winding = (ulon[-1] + closing) - ulon[0]
    ring = list(zip(ulon.tolist(), lat.tolist()))
    if abs(winding) > 180.0:  # ring wraps fully around a pole
        sign = 1.0 if winding > 0 else -1.0
        pole_lat = 90.0 if float(np.mean(lat)) > 0 else -90.0
        start_lon, start_lat = ring[0]
        ring = ring + [
            (start_lon + 360.0 * sign, start_lat),
            (start_lon + 360.0 * sign, pole_lat),
            (start_lon, pole_lat),
        ]
    return ring


def _normalize_lon(poly):
    """Clip an unwrapped-longitude polygon into [-180, 180], splitting across
    the antimeridian if needed."""
    minx, _, maxx, _ = poly.bounds
    k0 = int(np.floor((minx + 180.0) / 360.0))
    k1 = int(np.floor((maxx + 180.0) / 360.0))
    if k0 == 0 and k1 == 0:
        return poly
    parts = []
    for k in range(k0, k1 + 1):
        piece = poly.intersection(
            box(k * 360.0 - 180.0, -90.0, k * 360.0 + 180.0, 90.0)
        )
        if not piece.is_empty:
            parts.append(translate(piece, xoff=-360.0 * k))
    return unary_union(parts)


def _polygons_only(geom):
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        return unary_union(polys)
    return geom


def cell_polygon(verts3d: np.ndarray):
    """Lon/lat polygon (or MultiPolygon if split) for one Voronoi cell."""
    poly = Polygon(_unwrapped_ring(verts3d))
    if not poly.is_valid:
        poly = _polygons_only(make_valid(poly))
    return _polygons_only(_normalize_lon(poly))


def atoms_polygon(mesh, atom_ids, cell_cache: dict | None = None):
    """Union of the given atoms' cell polygons. Optional cache: atom id -> polygon."""
    geoms = []
    for i in atom_ids:
        i = int(i)
        if cell_cache is not None and i in cell_cache:
            geoms.append(cell_cache[i])
            continue
        g = cell_polygon(mesh.vertices[mesh.regions[i]])
        if cell_cache is not None:
            cell_cache[i] = g
        geoms.append(g)
    return _polygons_only(unary_union(geoms))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && ../.venv/bin/pytest tests/test_geometry.py -v`
Expected: all PASS. If `test_all_cells_valid_and_in_range` fails on a handful of cells, debug with the failing cell index — the usual cause is near-duplicate Voronoi vertices; fix by deduplicating consecutive identical ring points inside `_unwrapped_ring` (append only when the point differs from the previous by >1e-12), not by loosening the test.

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "feat: cell-to-lonlat geometry with antimeridian and pole handling"
```

---

### Task 8: Naming

**Files:**
- Create: `python/src/mimesis_earth/naming.py`
- Test: `python/tests/test_naming.py`

- [ ] **Step 1: Write failing tests**

`python/tests/test_naming.py`:

```python
import numpy as np

from mimesis_earth.naming import make_namer


def test_names_unique_and_wordlike():
    namer = make_namer(np.random.default_rng(40))
    names = [namer() for _ in range(300)]
    assert len(set(names)) == 300
    for n in names:
        assert n[0].isupper()
        assert 3 <= len(n) <= 20


def test_names_deterministic():
    a = [make_namer(np.random.default_rng(41))() for _ in range(10)]
    b = [make_namer(np.random.default_rng(41))() for _ in range(10)]
    assert a == b


def test_different_seeds_differ():
    a = [make_namer(np.random.default_rng(42))() for _ in range(10)]
    b = [make_namer(np.random.default_rng(43))() for _ in range(10)]
    assert a != b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && ../.venv/bin/pytest tests/test_naming.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement naming**

`python/src/mimesis_earth/naming.py`:

```python
"""Seeded syllable-based place-name generator. Each world draws its own
sound inventory, giving worlds distinct linguistic flavor."""

import numpy as np

ONSETS = [
    "b", "br", "c", "cr", "d", "dr", "f", "g", "gr", "h", "k", "kh", "l",
    "m", "n", "p", "pr", "r", "s", "sh", "st", "t", "th", "tr", "v", "z",
]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ai", "ea", "ia", "ou"]
CODAS = ["", "", "", "n", "r", "l", "s", "th", "m", "nd", "rk"]
SUFFIXES = ["ia", "a", "or", "un", "eth", "ara", "is", "ov", "ane", "und"]


def make_namer(rng: np.random.Generator):
    """Returns a zero-arg function producing unique capitalized names."""
    onsets = [str(x) for x in rng.choice(ONSETS, size=10, replace=False)]
    vowels = [str(x) for x in rng.choice(VOWELS, size=5, replace=False)]
    codas = [str(x) for x in rng.choice(CODAS, size=6, replace=False)]
    suffixes = [str(x) for x in rng.choice(SUFFIXES, size=4, replace=False)]
    used: set[str] = set()

    def pick(seq: list[str]) -> str:
        return seq[int(rng.integers(len(seq)))]

    def namer() -> str:
        name = ""
        for _ in range(50):
            n_syllables = int(rng.integers(2, 4))
            parts = [pick(onsets) + pick(vowels) for _ in range(n_syllables - 1)]
            if rng.random() < 0.6:
                parts.append(pick(onsets) + pick(suffixes))
            else:
                parts.append(pick(onsets) + pick(vowels) + pick(codas))
            name = "".join(parts).capitalize()
            if name not in used:
                used.add(name)
                return name
        name = f"{name}-{len(used)}"  # exhausted retries: disambiguate
        used.add(name)
        return name

    return namer
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && ../.venv/bin/pytest tests/test_naming.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "feat: seeded syllable name generator"
```

---

### Task 9: Population attributes

**Files:**
- Create: `python/src/mimesis_earth/attributes.py`
- Test: `python/tests/test_attributes.py`

- [ ] **Step 1: Write failing tests**

`python/tests/test_attributes.py`:

```python
import numpy as np

from mimesis_earth.attributes import population_density, round_preserving_sum
from mimesis_earth.mesh import build_mesh


def test_density_positive_and_deterministic():
    mesh = build_mesh(2000, np.random.default_rng(50))
    land_idx = np.arange(600)
    a = population_density(mesh, land_idx, np.random.default_rng(51))
    b = population_density(mesh, land_idx, np.random.default_rng(51))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (600,)
    assert (a > 0).all()


def test_density_spatially_varied():
    mesh = build_mesh(2000, np.random.default_rng(52))
    land_idx = np.arange(800)
    d = population_density(mesh, land_idx, np.random.default_rng(53))
    # cities exist: the densest atom is much denser than the median
    assert d.max() / np.median(d) > 3.0


def test_round_preserving_sum_exact():
    rng = np.random.default_rng(54)
    x = rng.uniform(0.1, 10.0, size=1000)
    out = round_preserving_sum(x, 1_000_000)
    assert out.sum() == 1_000_000
    assert (out >= 0).all()
    # proportions roughly preserved
    big, small = x.argmax(), x.argmin()
    assert out[big] > out[small]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && ../.venv/bin/pytest tests/test_attributes.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement attributes**

`python/src/mimesis_earth/attributes.py`:

```python
"""Synthetic demographics: a spatially correlated population density field."""

import numpy as np

from mimesis_earth.mesh import Mesh


def population_density(
    mesh: Mesh, land_idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Relative population density per land atom (aligned with land_idx).

    A few gaussian 'city' kernels on top of a rural base, times lognormal
    noise -> log-normal-ish unit sizes with spatial autocorrelation."""
    n_cities = min(12, len(land_idx))
    centers = land_idx[rng.integers(0, len(land_idx), size=n_cities)]
    weights = rng.lognormal(mean=0.0, sigma=1.0, size=n_cities)
    sigmas = rng.uniform(0.04, 0.18, size=n_cities)  # radians
    density = np.full(len(land_idx), 0.05)
    pts = mesh.points[land_idx]
    for c, w, s in zip(centers, weights, sigmas):
        ang = np.arccos(np.clip(pts @ mesh.points[int(c)], -1.0, 1.0))
        density += w * np.exp(-0.5 * (ang / s) ** 2)
    density *= rng.lognormal(mean=0.0, sigma=0.4, size=len(density))
    return density


def round_preserving_sum(x: np.ndarray, total: int) -> np.ndarray:
    """Scale positive weights x to integers summing exactly to `total`
    (largest-remainder method)."""
    scaled = x * (total / x.sum())
    base = np.floor(scaled).astype(np.int64)
    remainder = int(total - base.sum())
    order = np.argsort(-(scaled - base))
    base[order[:remainder]] += 1
    return base
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && ../.venv/bin/pytest tests/test_attributes.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "feat: population density field and sum-preserving rounding"
```

---

### Task 10: World model, generate() pipeline, exports, invariants

This is the integration task — the heart of the package.

**Files:**
- Create: `python/src/mimesis_earth/world.py`
- Create: `python/src/mimesis_earth/generate.py`
- Modify: `python/src/mimesis_earth/__init__.py`
- Test: `python/tests/test_world.py`

- [ ] **Step 1: Write failing tests**

`python/tests/test_world.py`:

```python
import json

import numpy as np
import pytest
from shapely.ops import unary_union

from mimesis_earth import World, WorldSpec, generate

# small, fast spec used across tests
SPEC = WorldSpec(
    levels=[4, 3, 3],
    n_landmasses=2,
    resolution=6000,
    count_variance=0.0,
    seed=7,
)


@pytest.fixture(scope="module")
def world() -> World:
    return generate(SPEC)


def test_unit_counts_exact_when_variance_zero(world):
    assert len(world.units_at(0)) == 4
    assert len(world.units_at(1)) == 4 * 3
    assert len(world.units_at(2)) == 4 * 3 * 3


def test_ids_and_parents(world):
    for level in range(1, 3):
        parent_ids = {u.id for u in world.units_at(level - 1)}
        for u in world.units_at(level):
            assert u.parent_id in parent_ids
            assert u.id.startswith(u.parent_id + ".")
    for u in world.units_at(0):
        assert u.parent_id is None
    all_ids = [u.id for u in world.units]
    assert len(set(all_ids)) == len(all_ids)


def test_landmass_count(world):
    landmasses = {u.landmass for u in world.units_at(0)}
    assert landmasses == {0, 1}


def test_geometries_valid_and_in_range(world):
    for u in world.units:
        assert u.geometry.is_valid, u.id
        assert not u.geometry.is_empty, u.id
        minx, miny, maxx, maxy = u.geometry.bounds
        assert -180.0001 <= minx <= maxx <= 180.0001
        assert -90.0001 <= miny <= maxy <= 90.0001


def test_children_tile_parent_exactly(world):
    for level in range(1, 3):
        for parent in world.units_at(level - 1):
            children = [u for u in world.units_at(level) if u.parent_id == parent.id]
            merged = unary_union([c.geometry for c in children])
            assert parent.geometry.symmetric_difference(merged).area < 1e-9


def test_siblings_do_not_overlap(world):
    districts = world.units_at(2)
    by_parent: dict = {}
    for u in districts:
        by_parent.setdefault(u.parent_id, []).append(u)
    for sibs in by_parent.values():
        for i in range(len(sibs)):
            for j in range(i + 1, len(sibs)):
                inter = sibs[i].geometry.intersection(sibs[j].geometry)
                assert inter.area < 1e-9


def test_population_sums(world):
    assert sum(u.population for u in world.units_at(2)) == SPEC.total_population
    for level in range(1, 3):
        for parent in world.units_at(level - 1):
            child_sum = sum(
                u.population for u in world.units_at(level) if u.parent_id == parent.id
            )
            assert child_sum == parent.population


def test_areas_positive_and_consistent(world):
    for u in world.units:
        assert u.area_km2 > 0
    for parent in world.units_at(0):
        child_area = sum(
            u.area_km2 for u in world.units_at(1) if u.parent_id == parent.id
        )
        np.testing.assert_allclose(child_area, parent.area_km2, rtol=1e-6)


def test_deterministic_and_seed_sensitive():
    w1 = generate(SPEC)
    w2 = generate(SPEC)
    j1 = json.dumps(w1.geojson_dict(2), sort_keys=True)
    j2 = json.dumps(w2.geojson_dict(2), sort_keys=True)
    assert j1 == j2
    w3 = generate(SPEC.model_copy(update={"seed": 8}))
    assert json.dumps(w3.geojson_dict(2), sort_keys=True) != j1


def test_geojson_dict_structure(world):
    fc = world.geojson_dict(0)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 4
    f = fc["features"][0]
    props = f["properties"]
    for key in (
        "id", "name", "level", "level_name", "parent_id",
        "population", "area_km2", "centroid_lon", "centroid_lat",
    ):
        assert key in props
    assert f["geometry"]["type"] in ("Polygon", "MultiPolygon")


def test_exports(tmp_path, world):
    out = tmp_path / "w"
    world.to_geojson(out)
    files = sorted(p.name for p in out.iterdir())
    assert files == [
        "level0_country.geojson",
        "level1_province.geojson",
        "level2_district.geojson",
        "spec.json",
    ]
    loaded = json.loads((out / "level2_district.geojson").read_text())
    assert len(loaded["features"]) == 36
    world.to_csv(tmp_path / "units.csv")
    lines = (tmp_path / "units.csv").read_text().strip().split("\n")
    assert len(lines) == 1 + len(world.units)
    assert lines[0].startswith("id,level,level_name,parent_id,name,population")


def test_gdf_roundtrip(tmp_path, world):
    geopandas = pytest.importorskip("geopandas")
    gdf = world.gdf(level=2)
    assert len(gdf) == 36
    assert gdf.crs.to_epsg() == 4326
    path = tmp_path / "districts.gpkg"
    gdf.to_file(path)
    back = geopandas.read_file(path)
    assert len(back) == 36
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && ../.venv/bin/pytest tests/test_world.py -v`
Expected: FAIL with `ImportError` (no `World`/`generate` in package)

- [ ] **Step 3: Implement world.py**

`python/src/mimesis_earth/world.py`:

```python
"""World and Unit: the generated product and its export methods."""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from shapely.geometry import mapping

from mimesis_earth.spec import WorldSpec

COORD_DECIMALS = 6  # ~0.1 m; applied identically to shared borders


def _round_coords(obj):
    if isinstance(obj, float):
        return round(obj, COORD_DECIMALS)
    if isinstance(obj, (list, tuple)):
        return [_round_coords(x) for x in obj]
    return obj


@dataclass
class Unit:
    id: str
    level: int
    level_name: str
    parent_id: Optional[str]
    name: str
    population: int
    area_km2: float
    centroid_lon: float
    centroid_lat: float
    geometry: object  # shapely Polygon or MultiPolygon, WGS84 lon/lat
    landmass: Optional[int] = None  # set for level-0 units


@dataclass
class World:
    spec: WorldSpec
    units: list[Unit] = field(default_factory=list)

    def units_at(self, level: int) -> list[Unit]:
        return [u for u in self.units if u.level == level]

    def _feature(self, u: Unit) -> dict:
        geom = mapping(u.geometry)
        geom["coordinates"] = _round_coords(geom["coordinates"])
        return {
            "type": "Feature",
            "properties": {
                "id": u.id,
                "name": u.name,
                "level": u.level,
                "level_name": u.level_name,
                "parent_id": u.parent_id,
                "population": u.population,
                "area_km2": round(u.area_km2, 3),
                "centroid_lon": round(u.centroid_lon, COORD_DECIMALS),
                "centroid_lat": round(u.centroid_lat, COORD_DECIMALS),
            },
            "geometry": geom,
        }

    def geojson_dict(self, level: int) -> dict:
        feats = [self._feature(u) for u in self.units_at(level)]
        feats.sort(key=lambda f: f["properties"]["id"])
        return {"type": "FeatureCollection", "features": feats}

    def to_geojson(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for level, name in enumerate(self.spec.level_names):
            path = directory / f"level{level}_{name}.geojson"
            path.write_text(
                json.dumps(
                    self.geojson_dict(level), sort_keys=True, separators=(",", ":")
                )
            )
        (directory / "spec.json").write_text(self.spec.model_dump_json(indent=2))

    def to_csv(self, path) -> None:
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "id", "level", "level_name", "parent_id", "name",
                    "population", "area_km2", "centroid_lon", "centroid_lat",
                ]
            )
            for u in sorted(self.units, key=lambda u: (u.level, u.id)):
                writer.writerow(
                    [
                        u.id, u.level, u.level_name, u.parent_id or "", u.name,
                        u.population, round(u.area_km2, 3),
                        round(u.centroid_lon, COORD_DECIMALS),
                        round(u.centroid_lat, COORD_DECIMALS),
                    ]
                )

    def gdf(self, level: Optional[int] = None):
        import geopandas  # optional dependency, imported lazily

        units = self.units if level is None else self.units_at(level)
        units = sorted(units, key=lambda u: (u.level, u.id))
        records = [
            {
                "id": u.id,
                "level": u.level,
                "level_name": u.level_name,
                "parent_id": u.parent_id,
                "name": u.name,
                "population": u.population,
                "area_km2": u.area_km2,
                "centroid_lon": u.centroid_lon,
                "centroid_lat": u.centroid_lat,
            }
            for u in units
        ]
        return geopandas.GeoDataFrame(
            records, geometry=[u.geometry for u in units], crs="EPSG:4326"
        )
```

- [ ] **Step 4: Implement generate.py**

`python/src/mimesis_earth/generate.py`:

```python
"""The generation pipeline: mesh -> land mask -> partitions -> geometry -> attributes."""

import numpy as np
from shapely.ops import unary_union

from mimesis_earth.attributes import population_density, round_preserving_sum
from mimesis_earth.geometry import R_EARTH_KM, atoms_polygon, xyz_to_lonlat
from mimesis_earth.landmask import build_landmask
from mimesis_earth.mesh import build_mesh
from mimesis_earth.naming import make_namer
from mimesis_earth.partition import allocate_counts, child_counts, partition_atoms
from mimesis_earth.spec import WorldSpec
from mimesis_earth.world import Unit, World


def generate(spec: WorldSpec) -> World:
    rng = np.random.default_rng(spec.seed)
    mesh = build_mesh(spec.resolution, rng)
    mask = build_landmask(mesh, spec, rng)
    roughness = spec.border_roughness_per_level()
    n_levels = len(spec.levels)

    # --- partition atoms level by level ---------------------------------
    # each entry: {"atoms": ndarray, "parent": index into previous level or None,
    #              "landmass": int (level 0 only)}
    level_nodes: list[list[dict]] = []
    group_sizes = np.array(
        [(mask.group == g).sum() for g in range(spec.n_landmasses)], dtype=float
    )
    counts0 = allocate_counts(spec.levels[0], group_sizes)
    top: list[dict] = []
    for g in range(spec.n_landmasses):
        idx = np.flatnonzero(mask.group == g)
        parts = partition_atoms(
            mesh, idx, int(counts0[g]), mask.bridges, roughness[0], rng
        )
        for atoms in parts:
            top.append({"atoms": atoms, "parent": None, "landmass": g})
    level_nodes.append(top)

    for level in range(1, n_levels):
        prev = level_nodes[level - 1]
        counts = child_counts(spec.levels[level], len(prev), spec.count_variance, rng)
        current: list[dict] = []
        for parent_index, parent in enumerate(prev):
            k = int(min(counts[parent_index], len(parent["atoms"])))
            parts = partition_atoms(
                mesh, parent["atoms"], k, mask.bridges, roughness[level], rng
            )
            for atoms in parts:
                current.append({"atoms": atoms, "parent": parent_index})
        level_nodes.append(current)

    # --- population on leaves --------------------------------------------
    land_idx = np.flatnonzero(mask.land)
    atom_density = np.zeros(len(mesh.points))
    atom_density[land_idx] = population_density(mesh, land_idx, rng)
    leaf_weights = np.array(
        [
            float((atom_density[n["atoms"]] * mesh.areas[n["atoms"]]).sum())
            for n in level_nodes[-1]
        ]
    )
    leaf_pops = round_preserving_sum(leaf_weights, spec.total_population)

    # --- attributes + geometry, bottom-up --------------------------------
    namer = make_namer(rng)
    cell_cache: dict = {}
    unit_grids: list[list[Unit]] = [[] for _ in range(n_levels)]

    # names must be drawn in a deterministic order: level by level, node order
    names = [[namer() for _ in level_nodes[lvl]] for lvl in range(n_levels)]

    # leaf geometries from atoms; parent geometry = union of children
    geoms: list[list] = [[None] * len(level_nodes[lvl]) for lvl in range(n_levels)]
    for i, node in enumerate(level_nodes[-1]):
        geoms[-1][i] = atoms_polygon(mesh, node["atoms"], cell_cache)
    for lvl in range(n_levels - 2, -1, -1):
        children_of: list[list] = [[] for _ in level_nodes[lvl]]
        for i, node in enumerate(level_nodes[lvl + 1]):
            children_of[node["parent"]].append(geoms[lvl + 1][i])
        for i, childs in enumerate(children_of):
            geoms[lvl][i] = unary_union(childs)

    # populations bottom-up
    pops: list[np.ndarray] = [None] * n_levels
    pops[-1] = leaf_pops
    for lvl in range(n_levels - 2, -1, -1):
        agg = np.zeros(len(level_nodes[lvl]), dtype=np.int64)
        for i, node in enumerate(level_nodes[lvl + 1]):
            agg[node["parent"]] += pops[lvl + 1][i]
        pops[lvl] = agg

    # build Unit objects top-down so ids exist before children need them
    id_grids: list[list[str]] = [[None] * len(level_nodes[lvl]) for lvl in range(n_levels)]
    child_counter: list[dict] = [dict() for _ in range(n_levels)]
    for lvl in range(n_levels):
        letter = spec.level_names[lvl][0].upper()
        for i, node in enumerate(level_nodes[lvl]):
            if lvl == 0:
                index = i + 1
                uid = f"{letter}{index:02d}"
                parent_id = None
            else:
                parent_pos = node["parent"]
                parent_id = id_grids[lvl - 1][parent_pos]
                index = child_counter[lvl].get(parent_pos, 0) + 1
                child_counter[lvl][parent_pos] = index
                uid = f"{parent_id}.{letter}{index:02d}"
            id_grids[lvl][i] = uid
            atoms = node["atoms"]
            weights = mesh.areas[atoms]
            center = (mesh.points[atoms] * weights[:, None]).sum(axis=0)
            center /= np.linalg.norm(center)
            lon, lat = xyz_to_lonlat(center[None, :])
            unit_grids[lvl].append(
                Unit(
                    id=uid,
                    level=lvl,
                    level_name=spec.level_names[lvl],
                    parent_id=parent_id,
                    name=names[lvl][i],
                    population=int(pops[lvl][i]),
                    area_km2=float(mesh.areas[atoms].sum() * R_EARTH_KM**2),
                    centroid_lon=float(lon[0]),
                    centroid_lat=float(lat[0]),
                    geometry=geoms[lvl][i],
                    landmass=node.get("landmass"),
                )
            )

    units = [u for grid in unit_grids for u in grid]
    return World(spec=spec, units=units)
```

Update `python/src/mimesis_earth/__init__.py`:

```python
"""mimesis-earth: rapid synthetic geography generator."""

from mimesis_earth.generate import generate
from mimesis_earth.spec import WorldSpec
from mimesis_earth.world import Unit, World

__version__ = "0.1.0"

__all__ = ["WorldSpec", "World", "Unit", "generate", "__version__"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd python && ../.venv/bin/pytest tests/test_world.py -v`
Expected: all PASS. This is the integration point — failures here are real bugs, not test problems. Debug rather than weaken assertions. Likely trouble spots: `test_children_tile_parent_exactly` tolerance (parents are literal unions of children, so only float noise from rounding is possible — should be << 1e-9) and empty geometries if a partition produced an atom set whose polygon union collapsed (check the partition sizes).

- [ ] **Step 6: Run the full suite**

Run: `cd python && ../.venv/bin/pytest -v`
Expected: all tests pass, total runtime under ~2 minutes.

- [ ] **Step 7: Commit**

```bash
git add python
git commit -m "feat: World model and full generation pipeline with invariant tests"
```

---

### Task 11: FastAPI server + CLI

**Files:**
- Create: `python/src/mimesis_earth/server.py`
- Create: `python/src/mimesis_earth/cli.py`
- Test: `python/tests/test_server.py`

- [ ] **Step 1: Write failing tests**

`python/tests/test_server.py`:

```python
from fastapi.testclient import TestClient

from mimesis_earth.server import app

client = TestClient(app)

SPEC = {
    "levels": [3, 3],
    "n_landmasses": 2,
    "resolution": 4000,
    "seed": 1,
}


def test_generate_endpoint():
    resp = client.post("/api/generate", json=SPEC)
    assert resp.status_code == 200
    data = resp.json()
    assert data["spec"]["levels"] == [3, 3]
    assert data["spec"]["seed"] == 1
    assert len(data["levels"]) == 2
    assert data["levels"][0]["name"] == "country"
    fc = data["levels"][0]["geojson"]
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 3
    assert len(data["levels"][1]["geojson"]["features"]) == 9


def test_invalid_spec_names_field():
    resp = client.post("/api/generate", json={**SPEC, "spread": 3.0})
    assert resp.status_code == 422
    assert "spread" in resp.text


def test_impossible_spec_is_422_not_500():
    resp = client.post(
        "/api/generate", json={"levels": [30, 30, 30], "resolution": 2000}
    )
    assert resp.status_code == 422
    assert "resolution" in resp.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python && ../.venv/bin/pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement server and CLI**

`python/src/mimesis_earth/server.py`:

```python
"""Stateless HTTP wrapper: POST a WorldSpec, get a world back as GeoJSON."""

from importlib import resources

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mimesis_earth.generate import generate
from mimesis_earth.spec import WorldSpec

app = FastAPI(title="mimesis-earth", docs_url=None, redoc_url=None)


@app.post("/api/generate")
def generate_world(spec: WorldSpec) -> dict:
    world = generate(spec)
    return {
        "spec": spec.model_dump(),
        "levels": [
            {
                "level": level,
                "name": name,
                "geojson": world.geojson_dict(level),
            }
            for level, name in enumerate(spec.level_names)
        ],
    }


def _mount_frontend() -> None:
    webapp = resources.files("mimesis_earth") / "webapp"
    if webapp.is_dir():
        app.mount("/", StaticFiles(directory=str(webapp), html=True), name="webapp")


_mount_frontend()
```

`python/src/mimesis_earth/cli.py`:

```python
"""Command-line entry point: `mimesis-earth serve`."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="mimesis-earth")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the web app + API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        from mimesis_earth.server import app

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

Note: pydantic validation errors on `WorldSpec` (including the model-level checks like the resolution rule) are automatically returned by FastAPI as 422 with the failing field/message in the body — no extra handler needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python && ../.venv/bin/pytest tests/test_server.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add python
git commit -m "feat: FastAPI generate endpoint and serve CLI"
```

---

### Task 12: Frontend scaffold (Vite + TS, page shell, paper styling)

**Files:**
- Create: `web/package.json`, `web/tsconfig.json`, `web/vite.config.ts`, `web/index.html`, `web/src/style.css`, `web/src/main.ts` (stub)

No unit tests for frontend tasks — verification is `npm run build` succeeding plus visual checks at the end.

- [ ] **Step 1: Create configs**

`web/package.json`:

```json
{
  "name": "mimesis-earth-web",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "d3-geo": "^3.1.0",
    "fflate": "^0.8.2"
  },
  "devDependencies": {
    "@types/d3-geo": "^3.1.0",
    "@types/geojson": "^7946.0.14",
    "typescript": "^5.4.0",
    "vite": "^5.2.0"
  }
}
```

`web/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"]
  },
  "include": ["src"]
}
```

`web/vite.config.ts`:

```typescript
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 2: Create page shell**

`web/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>mimesis-earth</title>
  </head>
  <body>
    <canvas id="globe"></canvas>

    <aside id="panel">
      <div class="panel-title">NEXT WORLD</div>
      <label>units/level <input id="p-levels" value="6,5,6" /></label>
      <label>islands <input id="p-islands" type="number" value="3" min="1" max="20" /></label>
      <label>spread <input id="p-spread" type="range" min="0" max="1" step="0.05" value="0.5" /></label>
      <label>coast <input id="p-coast" type="range" min="0" max="1" step="0.05" value="0.5" /></label>
      <label>borders <input id="p-borders" type="range" min="0" max="1" step="0.05" value="0.4" /></label>
      <label>population <input id="p-pop" type="number" value="50000000" step="1000000" /></label>
      <label>detail <input id="p-res" type="number" value="20000" step="2000" min="4000" max="200000" /></label>
      <label class="seed-row">seed <input id="p-seed" type="number" value="0" />
        <span title="new random seed on each spacebar"><input id="p-autoseed" type="checkbox" checked />🎲</span>
      </label>
    </aside>

    <nav id="levels"></nav>

    <div id="inspect" hidden>
      <div id="inspect-name"></div>
      <div id="inspect-id" class="mono dim"></div>
      <div id="inspect-pop" class="mono"></div>
      <div id="inspect-area" class="mono"></div>
    </div>

    <div id="status" hidden>generating…</div>
    <button id="export">⤓ geojson</button>
    <footer id="hint">space — new world · drag — rotate · scroll — zoom · click — inspect</footer>

    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

- [ ] **Step 3: Create styles**

`web/src/style.css`:

```css
:root {
  --paper: #faf8f3;
  --card: #fffdf8;
  --ink: #2f3a45;
  --dim: #7a7466;
  --line: #c9c2b2;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: Georgia, 'Times New Roman', serif;
  overflow: hidden;
}

#globe { display: block; width: 100vw; height: 100vh; cursor: grab; }
#globe.dragging { cursor: grabbing; }

#panel {
  position: fixed;
  top: 24px;
  left: 24px;
  width: 190px;
  background: var(--card);
  border: 1px solid var(--ink);
  padding: 12px 14px;
  font-family: ui-monospace, Menlo, monospace;
  font-size: 12px;
}
.panel-title { color: var(--dim); letter-spacing: 2px; font-size: 10px; margin-bottom: 8px; }
#panel label { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin: 6px 0; }
#panel input { font-family: inherit; font-size: 12px; color: var(--ink); background: transparent; border: none; border-bottom: 1px solid var(--line); width: 80px; text-align: right; }
#panel input[type="range"] { width: 80px; accent-color: var(--ink); }
#panel input[type="checkbox"] { width: auto; }
#panel input:focus { outline: none; border-bottom-color: var(--ink); }
.seed-row span { display: inline-flex; align-items: center; gap: 2px; }

#levels {
  position: fixed;
  bottom: 56px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 14px;
  font-size: 13px;
}
#levels button {
  background: none; border: none; cursor: pointer;
  font: inherit; color: var(--dim); padding: 2px 4px;
}
#levels button.active { color: var(--ink); border-bottom: 1px solid var(--ink); }

#inspect {
  position: fixed;
  top: 24px;
  right: 24px;
  min-width: 200px;
  background: var(--card);
  border: 1px solid var(--ink);
  padding: 12px 16px;
}
#inspect-name { font-size: 16px; margin-bottom: 4px; }
.mono { font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
.dim { color: var(--dim); }

#status {
  position: fixed;
  top: 24px;
  left: 50%;
  transform: translateX(-50%);
  color: var(--dim);
  font-style: italic;
  font-size: 13px;
}

#export {
  position: fixed;
  bottom: 24px;
  right: 24px;
  background: none;
  border: none;
  cursor: pointer;
  font-family: ui-monospace, Menlo, monospace;
  font-size: 12px;
  color: var(--dim);
}
#export:hover { color: var(--ink); }

#hint {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  color: var(--dim);
  font-size: 13px;
}
```

- [ ] **Step 4: Stub main.ts and verify build**

`web/src/main.ts` (stub, replaced in Task 14):

```typescript
import './style.css'

console.log('mimesis-earth frontend stub')
```

Run:

```bash
cd web && npm install && npm run build
```

Expected: `vite build` completes, `web/dist/` created.

- [ ] **Step 5: Commit**

```bash
git add web
git commit -m "feat: frontend scaffold with paper styling"
```

---

### Task 13: Globe rendering (canvas, drag, zoom, pick)

**Files:**
- Create: `web/src/api.ts`
- Create: `web/src/globe.ts`

- [ ] **Step 1: Create api.ts**

`web/src/api.ts`:

```typescript
import type { FeatureCollection } from 'geojson'

export interface Spec {
  levels: number[]
  n_landmasses: number
  spread: number
  coast_ruggedness: number
  border_roughness: number
  total_population: number
  resolution: number
  seed: number
}

export interface LevelData {
  level: number
  name: string
  geojson: FeatureCollection
}

export interface WorldData {
  spec: Spec & { level_names: string[] }
  levels: LevelData[]
}

export async function generateWorld(spec: Spec): Promise<WorldData> {
  const resp = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(spec),
  })
  if (!resp.ok) {
    const body = await resp.text()
    throw new Error(`generate failed (${resp.status}): ${body}`)
  }
  return resp.json()
}
```

- [ ] **Step 2: Create globe.ts**

`web/src/globe.ts`:

```typescript
import { geoContains, geoGraticule10, geoOrthographic, geoPath } from 'd3-geo'
import type { GeoProjection } from 'd3-geo'
import type { Feature, FeatureCollection } from 'geojson'

const INK = '#2f3a45'
const SPHERE = '#f1ecdf'
const LAND = '#e5decb'
const GRID = '#cfc8b6'
const HILITE = 'rgba(214, 185, 140, 0.65)'

export class Globe {
  private ctx: CanvasRenderingContext2D
  private projection: GeoProjection
  private rotation: [number, number] = [20, -15]
  private zoomFactor = 1
  private fc: FeatureCollection | null = null
  private width = 0
  private height = 0
  private dpr = 1
  selected: Feature | null = null
  onPick: (f: Feature | null) => void = () => {}

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext('2d')!
    this.projection = geoOrthographic().clipAngle(90)
    window.addEventListener('resize', () => this.resize())
    this.bindPointer()
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault()
        this.zoomFactor = Math.min(
          60,
          Math.max(0.7, this.zoomFactor * Math.exp(-e.deltaY * 0.0015)),
        )
        this.draw()
      },
      { passive: false },
    )
    this.resize()
  }

  setWorld(fc: FeatureCollection) {
    this.fc = fc
    this.selected = null
    this.canvas.animate([{ opacity: 0.3 }, { opacity: 1 }], { duration: 220 })
    this.draw()
  }

  setSelected(f: Feature | null) {
    this.selected = f
    this.draw()
  }

  private resize() {
    this.dpr = window.devicePixelRatio || 1
    this.width = window.innerWidth
    this.height = window.innerHeight
    this.canvas.width = this.width * this.dpr
    this.canvas.height = this.height * this.dpr
    this.canvas.style.width = `${this.width}px`
    this.canvas.style.height = `${this.height}px`
    this.draw()
  }

  private bindPointer() {
    let dragging = false
    let moved = false
    let last: [number, number] = [0, 0]
    this.canvas.addEventListener('pointerdown', (e) => {
      dragging = true
      moved = false
      last = [e.clientX, e.clientY]
      this.canvas.setPointerCapture(e.pointerId)
      this.canvas.classList.add('dragging')
    })
    this.canvas.addEventListener('pointermove', (e) => {
      if (!dragging) return
      const dx = e.clientX - last[0]
      const dy = e.clientY - last[1]
      if (Math.abs(dx) + Math.abs(dy) > 2) moved = true
      last = [e.clientX, e.clientY]
      const k = 90 / this.projection.scale()
      this.rotation = [
        this.rotation[0] + dx * k,
        Math.max(-90, Math.min(90, this.rotation[1] - dy * k)),
      ]
      this.draw()
    })
    this.canvas.addEventListener('pointerup', (e) => {
      dragging = false
      this.canvas.classList.remove('dragging')
      if (!moved) this.pick(e.clientX, e.clientY)
    })
  }

  private pick(x: number, y: number) {
    if (!this.fc) return
    const ll = this.projection.invert?.([x * 1, y * 1]) ?? null
    if (!ll) {
      this.setSelected(null)
      this.onPick(null)
      return
    }
    // invert() returns a point even outside the disc; verify it re-projects nearby
    const back = this.projection([ll[0], ll[1]])
    if (!back || Math.hypot(back[0] - x, back[1] - y) > 2) {
      this.setSelected(null)
      this.onPick(null)
      return
    }
    for (const f of this.fc.features) {
      if (geoContains(f, [ll[0], ll[1]])) {
        this.setSelected(f)
        this.onPick(f)
        return
      }
    }
    this.setSelected(null)
    this.onPick(null)
  }

  draw() {
    const { ctx } = this
    const scale = 0.42 * Math.min(this.width, this.height) * this.zoomFactor
    this.projection
      .scale(scale)
      .translate([this.width / 2, this.height / 2])
      .rotate([this.rotation[0], this.rotation[1], 0])
    const path = geoPath(this.projection, ctx)

    ctx.save()
    ctx.scale(this.dpr, this.dpr)
    ctx.clearRect(0, 0, this.width, this.height)
    ctx.lineJoin = 'round'

    // sphere
    ctx.beginPath()
    path({ type: 'Sphere' })
    ctx.fillStyle = SPHERE
    ctx.fill()

    // graticule
    ctx.beginPath()
    path(geoGraticule10())
    ctx.strokeStyle = GRID
    ctx.lineWidth = 0.5
    ctx.stroke()

    // land units
    if (this.fc) {
      for (const f of this.fc.features) {
        ctx.beginPath()
        path(f)
        ctx.fillStyle = f === this.selected ? HILITE : LAND
        ctx.fill()
        ctx.strokeStyle = INK
        ctx.lineWidth = 0.7
        ctx.stroke()
      }
    }

    // sphere outline on top
    ctx.beginPath()
    path({ type: 'Sphere' })
    ctx.strokeStyle = INK
    ctx.lineWidth = 1.4
    ctx.stroke()
    ctx.restore()
  }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd web && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/api.ts web/src/globe.ts
git commit -m "feat: canvas globe with drag, zoom, and polygon picking"
```

---

### Task 14: Frontend wiring (panel, inspect, export, spacebar)

**Files:**
- Create: `web/src/panel.ts`, `web/src/inspect.ts`, `web/src/exporter.ts`
- Modify: `web/src/main.ts` (replace stub)

- [ ] **Step 1: Create panel.ts**

`web/src/panel.ts`:

```typescript
import type { Spec } from './api'

const $ = (id: string) => document.getElementById(id) as HTMLInputElement

export function readSpec(): Spec {
  const levels = $('p-levels')
    .value.split(',')
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n > 0)
  return {
    levels: levels.length ? levels : [6, 5, 6],
    n_landmasses: parseInt($('p-islands').value, 10) || 3,
    spread: parseFloat($('p-spread').value),
    coast_ruggedness: parseFloat($('p-coast').value),
    border_roughness: parseFloat($('p-borders').value),
    total_population: parseInt($('p-pop').value, 10) || 50_000_000,
    resolution: parseInt($('p-res').value, 10) || 20_000,
    seed: parseInt($('p-seed').value, 10) || 0,
  }
}

export function maybeRandomizeSeed(): void {
  if ($('p-autoseed').checked) {
    $('p-seed').value = String(Math.floor(Math.random() * 1_000_000))
  }
}

export function isTypingInPanel(target: EventTarget | null): boolean {
  return target instanceof HTMLInputElement
}
```

- [ ] **Step 2: Create inspect.ts**

`web/src/inspect.ts`:

```typescript
import type { Feature } from 'geojson'

const el = (id: string) => document.getElementById(id)!

export function showInspect(f: Feature): void {
  const p = f.properties ?? {}
  el('inspect-name').textContent = String(p.name ?? '?')
  el('inspect-id').textContent = String(p.id ?? '')
  el('inspect-pop').textContent = `pop   ${Number(p.population ?? 0).toLocaleString()}`
  el('inspect-area').textContent = `area  ${Math.round(Number(p.area_km2 ?? 0)).toLocaleString()} km²`
  el('inspect').hidden = false
}

export function hideInspect(): void {
  el('inspect').hidden = true
}
```

- [ ] **Step 3: Create exporter.ts**

`web/src/exporter.ts`:

```typescript
import { strToU8, zipSync } from 'fflate'
import type { WorldData } from './api'

export function downloadWorld(world: WorldData): void {
  const files: Record<string, Uint8Array> = {}
  for (const lvl of world.levels) {
    files[`level${lvl.level}_${lvl.name}.geojson`] = strToU8(
      JSON.stringify(lvl.geojson),
    )
  }
  files['spec.json'] = strToU8(JSON.stringify(world.spec, null, 2))

  const rows = ['id,level,level_name,parent_id,name,population,area_km2']
  for (const lvl of world.levels) {
    for (const f of lvl.geojson.features) {
      const p = f.properties ?? {}
      rows.push(
        [p.id, p.level, p.level_name, p.parent_id ?? '', p.name, p.population, p.area_km2].join(','),
      )
    }
  }
  files['units.csv'] = strToU8(rows.join('\n'))

  const zip = zipSync(files)
  const blob = new Blob([zip], { type: 'application/zip' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `world-seed${world.spec.seed}.zip`
  a.click()
  URL.revokeObjectURL(a.href)
}
```

- [ ] **Step 4: Replace main.ts**

`web/src/main.ts`:

```typescript
import './style.css'
import { generateWorld } from './api'
import type { WorldData } from './api'
import { Globe } from './globe'
import { hideInspect, showInspect } from './inspect'
import { downloadWorld } from './exporter'
import { isTypingInPanel, maybeRandomizeSeed, readSpec } from './panel'

const globe = new Globe(document.getElementById('globe') as HTMLCanvasElement)
const statusEl = document.getElementById('status')!
const levelsNav = document.getElementById('levels')!

let world: WorldData | null = null
let levelIndex = 0
let busy = false

globe.onPick = (f) => (f ? showInspect(f) : hideInspect())

function setLevel(i: number): void {
  if (!world) return
  levelIndex = Math.max(0, Math.min(i, world.levels.length - 1))
  globe.setWorld(world.levels[levelIndex].geojson)
  hideInspect()
  renderLevelNav()
}

function renderLevelNav(): void {
  if (!world) return
  levelsNav.innerHTML = ''
  world.levels.forEach((lvl, i) => {
    const btn = document.createElement('button')
    btn.textContent = lvl.name
    btn.className = i === levelIndex ? 'active' : ''
    btn.addEventListener('click', () => setLevel(i))
    levelsNav.appendChild(btn)
  })
}

async function newWorld(): Promise<void> {
  if (busy) return
  busy = true
  statusEl.hidden = false
  try {
    world = await generateWorld(readSpec())
    setLevel(Math.min(levelIndex, world.levels.length - 1))
  } catch (err) {
    statusEl.hidden = false
    statusEl.textContent = String(err)
    setTimeout(() => {
      statusEl.hidden = true
      statusEl.textContent = 'generating…'
    }, 4000)
    busy = false
    return
  }
  statusEl.hidden = true
  busy = false
}

window.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !isTypingInPanel(e.target)) {
    e.preventDefault()
    maybeRandomizeSeed()
    void newWorld()
  }
})

document.getElementById('export')!.addEventListener('click', () => {
  if (world) downloadWorld(world)
})

void newWorld()
```

- [ ] **Step 5: Build and smoke-test against the live server**

```bash
cd web && npm run build
cd /Users/user/Documents/work/mimesis-earth
.venv/bin/mimesis-earth serve --port 8000 &
sleep 2
curl -s -X POST localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"levels":[3,3],"resolution":4000,"seed":1}' | head -c 300
```

Expected: JSON starting with `{"spec":`. Then run the dev frontend:

```bash
cd web && npm run dev
```

Open http://localhost:5173 — expect: globe renders with a world; spacebar generates a new one; drag rotates; scroll zooms; clicking a polygon shows the inspect card; level buttons switch layers; export downloads a zip. Kill the background server afterwards (`kill %1`).

- [ ] **Step 6: Commit**

```bash
git add web
git commit -m "feat: frontend wiring - panel, inspect card, export, spacebar"
```

---

### Task 15: Build pipeline, packaged serving, README

**Files:**
- Create: `scripts/build_web.sh`
- Create: `README.md`

- [ ] **Step 1: Create build script**

`scripts/build_web.sh`:

```bash
#!/usr/bin/env bash
# Build the frontend and embed it in the python package so
# `pip install` + `mimesis-earth serve` needs no Node.
set -euo pipefail
cd "$(dirname "$0")/.."
(cd web && npm run build)
rm -rf python/src/mimesis_earth/webapp
cp -r web/dist python/src/mimesis_earth/webapp
echo "webapp embedded: $(du -sh python/src/mimesis_earth/webapp | cut -f1)"
```

```bash
chmod +x scripts/build_web.sh
./scripts/build_web.sh
```

Expected: prints `webapp embedded: ...`.

- [ ] **Step 2: End-to-end check of the packaged app**

```bash
.venv/bin/mimesis-earth serve --port 8001 &
sleep 2
curl -s localhost:8001/ | head -c 200        # expect the index.html
curl -s -o /dev/null -w "%{http_code}" -X POST localhost:8001/api/generate \
  -H 'Content-Type: application/json' -d '{"seed":3,"resolution":6000}'
kill %1
```

Expected: HTML fragment containing `<canvas id="globe">`, then `200`.

- [ ] **Step 3: Write README**

`README.md`:

````markdown
# mimesis-earth

Rapidly generate synthetic world geographies: strictly nested administrative
units (countries → provinces → districts) with organic coastlines, real WGS84
coordinates, and consistent synthetic demographics. Deterministic: a spec +
seed always reproduces the same world.

## Install & run

```bash
python -m venv .venv && .venv/bin/pip install -e './python[dev]'
.venv/bin/mimesis-earth serve      # open http://localhost:8000
```

Press **space** for a new world. Drag to rotate, scroll to zoom, click a
polygon to inspect. The top-left panel sets parameters for the next world.

## Python API

```python
from mimesis_earth import WorldSpec, generate

world = generate(WorldSpec(levels=[8, 6, 9], n_landmasses=4, seed=42))
world.gdf(level=2)          # geopandas GeoDataFrame of districts
world.to_geojson("out/")    # one FeatureCollection per level + spec.json
world.to_csv("out/units.csv")
```

## Development

- Python package: `python/` — `cd python && ../.venv/bin/pytest`
- Frontend: `web/` — `cd web && npm install && npm run dev` (proxies /api to :8000)
- Embed frontend in the package: `./scripts/build_web.sh`

Design docs: `docs/superpowers/specs/`.
````

- [ ] **Step 4: Full test suite one last time**

Run: `cd python && ../.venv/bin/pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts README.md
git commit -m "feat: web build pipeline and README"
```

---

## Self-review notes (already applied)

- Spec coverage: WorldSpec params (Task 2), atoms/mesh (3), spread via vMF + coast_ruggedness noise (4, 5), n_landmasses grouping + archipelago bridges (5), strict nesting via atom partitioning (6, 10), antimeridian/pole-safe WGS84 geometry (7), names (8), population with consistent sums (9, 10), IDs/centroid/area + exports + determinism (10), server 422s + serve CLI (11), minimalist globe frontend with all specced interactions (12–14), packaged single-command serving (15). Border roughness per level: `border_roughness_per_level()` consumed in `generate()`. Landmass count guarantee: retry loop in `build_landmask` + test.
- Deliberate deviations from spec examples: `count_variance` interacts with `child_counts` only for levels ≥ 1 (level-0 counts are exact — the user asked for "8 countries" and gets 8; variance there would fight `allocate_counts`). `level_names`-derived ID letters collide if two level names share a first letter — IDs remain unique regardless (position encodes level).
- Type consistency: `partition_atoms(mesh, atom_idx, k, extra_edges, roughness, rng)` signature matches all call sites; `World.geojson_dict(level)` used by both server and tests; `Spec` interface in `api.ts` matches `WorldSpec` field names exactly (pydantic fills omitted fields with defaults).
```
