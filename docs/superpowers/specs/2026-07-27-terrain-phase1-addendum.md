# Terrain Phase 1 Addendum — Elevation & Sea-Level Coastlines

**Date:** 2026-07-27
**Status:** Approved design (session discussion), extends the 2026-07-26 design
and 2026-07-27 addenda. **Generator version → `0.4.0`** (land shapes and rng
stream change for all seeds).

## Scope

An explicit per-atom **elevation field** becomes the source of truth for land:
`land = elevation above sea level`, where sea level is the quantile hitting
`land_fraction` exactly. Borders' meander cost field switches from phantom
noise to this same elevation (one coherent geography). Per-unit mean elevation
is exported. **Deferred:** rivers/lakes via flow routing, tectonic simulation,
terrain rendering, population–elevation coupling.

## Elevation field (`elevation.py`, new module)

`build_elevation(mesh, seeds, spec, rng) -> np.ndarray` (per-atom, unitless):

```
base      = continent bumps: falloff of angular distance to nearest landmass
            seed (z-scored) — preserves the meaning of islands/spread
ridged    = mountain chains: r = (1 - |n|)^1.7 for a z-scored sphere_noise n
            (octaves=5, base_freq=2.5), then z-scored — crisp ridge lines
detail    = fine sphere_noise (octaves=4, base_freq=6.0), z-scored
elevation = base + coast_ruggedness * (0.9 * ridged + 0.5 * detail)
```

`coast_ruggedness` is REINTERPRETED as relief amplitude: 0 → smooth
continents with clean coasts; 1 → mountainous worlds with ragged, fjord-like
coasts and offshore island arcs (ridge crests piercing sea level). No new
spec fields.

## Land mask (`landmask.py` rework, same interface & guarantees)

`build_landmask(mesh, spec, rng, elevation)` — signature gains the elevation
array; scoring becomes: per-seed guarantee kernels first (unchanged
mechanism), remaining land budget = highest-elevation atoms. Sea level is
implicit in that quantile cut. Landmass groups (nearest seed), island
bridges, the retry loop, exact `land_fraction`, and the exactly-`n_landmasses`
guarantee all carry over unchanged. Interior depressions below sea level
enclosed by land become inland seas — allowed and desirable (they are simply
sea).

## Border cost coherence (`generate.py`)

The phantom `sphere_noise` draw for `atom_cost` is REMOVED; instead
`atom_cost = exp(1.5 * border_meander * zscore(elevation))`. Borders now lock
onto the same ridges that shape the coastline (crest/watershed mechanism,
proven in the border-meander work). One fewer rng draw → stream layout
changes → version bump covers it.

## Exports

`Unit` gains `elevation_m: int` — area-weighted mean over the unit's atoms,
scaled to meters-ish: after computing sea level, land elevation maps linearly
so the world's highest atom ≈ 4500 m and sea level = 0. Added to GeoJSON
properties, CSV column (end), and gdf. Deterministic.

## Testing

- elevation.py: determinism; ridged field has crisp maxima (kurtosis/quantile
  signature vs plain noise); shape/z-scoring.
- landmask: existing guarantee/regression tests pass with the new scoring
  (land_fraction exact, n_landmasses exact incl. the 63-landmass and
  low-spread regression zones, bridges intra-group, determinism).
- Coherence: at meander 1, mean elevation at border atoms substantially
  exceeds land mean (borders on ridges).
- Ruggedness: island count (land components) rises with coast_ruggedness;
  at 0 coastlines smooth (few components per landmass group).
- Exports: elevation_m present in all three export paths; sea-adjacent units
  low, interior mountain units high; full invariant suite green.

## Out of scope (deferred)

Rivers/lakes (flow routing), tectonic plates, hillshade/hypsometric
rendering, population–elevation coupling, exposing raw elevation rasters.
