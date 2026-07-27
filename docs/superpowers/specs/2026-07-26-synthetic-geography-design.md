# mimesis-earth — Synthetic Geography Generator

**Date:** 2026-07-26
**Status:** Approved design, pending implementation plan

## Purpose

Rapidly generate synthetic world geographies — nested administrative units
(countries → provinces → districts) with realistic-looking coastlines and
borders — as valid geospatial data. Two driving use cases:

1. **Test/demo data** for tools that consume real GeoJSON/CSV/shapefile admin
   data (dashboards, spatial pipelines, epi models).
2. **Research substrate**: controllable geography as an input to simulations
   (vary spatial structure, observe model behavior).

A minimalist web frontend lets the user generate and browse worlds on a
draggable globe by pressing the spacebar.

## Decisions made during brainstorming

| Question | Decision |
|---|---|
| Coordinates | Real WGS84 lon/lat on the sphere; output loads correctly in any GIS tool |
| Hierarchy topology | Strictly nested and gapless by construction (children exactly tile parent, no sibling overlaps). Controllable "messiness" is a possible later opt-in, not core. Contiguity below country level follows the [partition-realism addendum](2026-07-27-partition-realism-addendum.md)'s contiguity contract: bridges apply only at level 0 (countries may span islands); deeper units are contiguous except for attached islets and children of quota-starved archipelago parents |
| Visual/geometry style | Noise-perturbed organic borders and coastlines now ("B"); architecture leaves a slot to evolve to terrain-first elevation-derived land later ("C") |
| Scale | Medium: 3–4 levels, hundreds to ~12k leaf units (the practical ceiling at the resolution cap of 200k atoms with the 8-atoms-per-leaf validation floor; ~28s generation at that extreme, ~1s at defaults); per-level counts are user parameters |
| Core language | Python (source of truth). Dependencies restricted to numpy/scipy/shapely/pydantic — all four ship Pyodide wheels (pydantic via pydantic_core), so the core stays Pyodide-compatible; running the same code in-browser is a future experiment, not a requirement |
| Unit attributes | Identity (hierarchical ID, level, parent, generated name) + demographics (population, area, centroid). Extensible later (covariates etc.) |
| Persistence | In-memory; nothing written unless explicitly exported. Spec + seed fully reproduces a world (within a generator version, recorded in the spec) |
| Architecture | Python package + thin FastAPI server + static TS frontend (approach 1) |
| Frontend | Minimalist paper-and-ink globe UI (see Frontend section) |

## Core generation algorithm

Everything is built from a **fine mesh of atoms**: ~30k–100k jittered points
on the sphere (Fibonacci lattice + noise) and their spherical Voronoi cells.
Every geographic shape is a union of atoms, which is what guarantees strict
nesting and gapless topology with zero polygon clipping.

Pipeline (all randomness from one seeded `numpy` RNG):

1. **Land mask** *(pluggable step)*. Place K landmass seeds on the sphere;
   the `spread` parameter controls their dispersion (von Mises–Fisher
   sampling: low = clustered in one patch, high = scattered globally). Score
   each atom: `landness = distance-falloff from nearest seed + fractal
   noise`; atoms above a threshold (calibrated to hit `land_fraction`) are
   land. Connected land atoms form landmasses; noise makes coastlines ragged
   organically. *Future terrain-first mode replaces only this scoring with an
   elevation field + sea level; everything downstream is unchanged.*
2. **Countries.** Partition land atoms among N country seeds by competitive
   flood-fill over the atom adjacency graph (each seed grows outward,
   claiming atoms by lowest accumulated cost).
3. **Deeper levels.** Recursively partition each parent's atoms the same way.
   Every atom belongs to exactly one leaf unit, hence exactly one unit per
   level.
4. **Geometry.** A unit's polygon = shapely union of its atoms' cells, in
   WGS84 lon/lat.
5. **Attributes.** A smooth population-density field on the sphere is
   integrated per leaf unit (spatially correlated, log-normal-ish sizes) and
   scaled to `total_population`; parent populations are sums of children.
   Names from a seeded syllable generator; IDs hierarchical (`C03.P07.D12`).

### Boundaries

Borders are emergent, not modeled as objects. The border between sibling
units is the chain of shared atom edges between differently-assigned atoms;
both polygons contain vertex-identical border segments (no gaps/slivers,
TopoJSON-friendly). Border character is controlled by the flood-fill cost
function: pure geodesic distance → near-straight borders; distance + per-atom
noise → wiggly borders. `border_roughness` scales the noise (scalar or
per-level). Border/coastline vertex detail is bounded by atom size; the
`resolution` parameter trades detail for speed. Terrain-first mode can later
make the cost terrain-aware (borders following ridges/rivers) in the same
slot. Optional (non-core) export: dissolved boundary lines as a separate
layer.

## Python API

Package `mimesis_earth`. Public surface:

```python
from mimesis_earth import WorldSpec, generate

spec = WorldSpec(
    levels=[8, 6, 9],          # per-parent mean counts; ~432 leaf units here
    level_names=["country", "province", "district"],  # optional defaults
    n_landmasses=4,
    spread=0.6,                # 0 = one tight patch, 1 = scattered worldwide
    land_fraction=0.3,
    coast_ruggedness=0.5,
    border_roughness=0.4,      # scalar or per-level list
    count_variance=0.2,        # 0 = exact counts per parent
    total_population=80_000_000,
    resolution=30_000,         # atom count: detail vs speed
    seed=42,
)
world = generate(spec)

world.gdf(level=2)             # geopandas GeoDataFrame (id, parent_id, name,
                               #   population, area_km2, geometry); .gdf() = all levels
world.to_geojson("out/")       # one FeatureCollection per level
world.to_csv("out/units.csv")  # attributes without geometry
world.spec                     # full spec incl. seed and generator version
```

- `WorldSpec` is a pydantic model; every field has a sensible default, and
  impossible combinations fail fast with messages naming the parameter to
  change (e.g., resolution too low for requested leaf count).
- Determinism: same spec + seed + package version → byte-identical output.
  The generator version is recorded in the spec for this reason.
  `GENERATOR_VERSION` is deliberately independent of the package
  `__version__`: it tracks the generation *algorithm* and is bumped when the
  same seed starts producing different output, not on every release.
- geopandas is an optional extra (`mimesis-earth[geo]`); core requires only
  numpy, scipy, shapely (all Pyodide-available). Shapefile/GeoPackage export
  is delegated to geopandas (`world.gdf().to_file(...)`), not reimplemented.

## Server

FastAPI, stateless, ~100 lines:

- `POST /api/generate` — body: `WorldSpec` JSON (validated by the same
  pydantic class); response: world GeoJSON for all levels + attributes +
  echoed spec. Invalid specs → 422 naming the offending field.
- `GET /` — serves the built frontend.

CLI entry point `mimesis-earth serve` runs the single process. The built
frontend ships inside the Python package, so pip install needs no Node. No
job queue, caching, or persistence — worlds regenerate in roughly ≤1s at
default scale; slow heavy worlds just show the frontend's "generating" state.

## Frontend

Vite + TypeScript + d3-geo, no framework. Paper-and-ink aesthetic: off-white
background, thin dark strokes, no header/branding. Validated mockup:
minimalist orthographic globe, solid (paper-sphere, back side hidden), faint
front graticule.

- **Globe**: d3 orthographic projection on canvas (fast at 20k polygons).
  Drag rotates, scroll zooms, click a polygon → inspect card (name, ID,
  population, area). A small level switcher (country/province/district)
  selects the displayed layer.
- **Spacebar** generates the next world via `POST /api/generate`; short
  fade-in on arrival. Previous world is discarded (spec-only history could be
  added later).
- **Paper panel (top-left)**: spec fields (levels/counts, islands, spread,
  ruggedness, seed with random-seed-per-press toggle, on by default). Applies
  to the next world.
- **Export (`⤓ geojson`, bottom-right)**: downloads current in-browser world
  as zip of per-level GeoJSON + units.csv; no server round-trip.
- **Hint line (bottom-center)**: `space — new world · drag — rotate ·
  scroll — zoom · click — inspect`.

Repo shape: single repo, `python/` (package + server) and `web/` (frontend).

## Testing

- **Invariant tests** over a grid of specs × seeds: valid geometries;
  children exactly tile their parent; no sibling overlaps; single parent per
  unit; legal lon/lat; population sums consistent up the hierarchy; landmass
  and per-level counts respect the spec (exactly when `count_variance=0`).
- **Determinism**: same spec+seed → byte-identical GeoJSON; different seeds
  differ.
- **Ecosystem round-trip**: output loads cleanly through geopandas.
- **Server**: valid spec → well-formed world; invalid spec → 422 with field
  name.
- Frontend: no automated tests initially; iterated visually.

Error handling: front-loaded into `WorldSpec` validation. The generator
never emits an invalid world — internal assertion failures (e.g., unassigned
atom) raise loudly.

## Out of scope (for now)

- Terrain-first land generation (elevation, rivers, lakes) — designed-for
  future slot.
- Intentional topological messiness (slivers, exclaves, disputed borders) —
  possible later opt-in.
- Pyodide in-browser generation — enabled by the dependency constraint,
  explored later.
- Rich attribute generators (GDP-like covariates, urban/rural flags).
- World feed/gallery UI; world history.
