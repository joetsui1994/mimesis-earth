# Border Meander Addendum

**Date:** 2026-07-27
**Status:** Approved design (discussed and accepted in session), extends
`2026-07-27-partition-realism-addendum.md`
**Generator version:** output changes for all worlds (new default-on field and
an rng-stream change) → bump `GENERATOR_VERSION` to `0.3.0`.

## Motivation (user-reported)

Borders between neighboring units — most visibly countries, noticeably
provinces — are straight lines with small-scale wiggle. Cause: the growth
frontier between two seeds settles on the equal-cost bisector, and per-edge
jitter is white noise (correlation length = one atom), so deviations regress
to the straight baseline. Real borders follow spatially *correlated* features
(rivers, ridges) with structure at every scale.

## Design: a correlated cost field ("phantom terrain")

- New `WorldSpec` field: `border_meander: float`, range 0.0–1.0, default 0.5.
  0 reproduces today's behavior exactly (no field influence).
- `generate()` draws one smooth multi-octave field per world, immediately
  after `build_landmask` (fixed rng draw order):
  `field = sphere_noise(mesh.points, rng, octaves=6, base_freq=2.0)`
  drawn **unconditionally** (stream layout independent of the knob), then
  converts it to a per-atom cost multiplier:
  `atom_cost = exp(1.5 * border_meander * field)`.
  At meander 1.0 regional cost ratios reach ~e^±3 (≈20×) — strong channeling;
  at the 0.5 default ≈4.5× — visible meanders without wrecking balance.
- `partition_atoms(..., atom_cost: np.ndarray | None = None)`: `_subgraph`
  multiplies each mesh-edge weight by the **geometric mean** of its two
  endpoints' costs (`w *= sqrt(atom_cost[a] * atom_cost[b])`). Applied at
  every level (same field world-wide, so province borders inherit the same
  terrain logic as country borders). NOT applied to bridge edges (they
  represent sea crossings) and not used by `_island_analysis` (connectivity
  only). `None` skips everything (direct callers/tests unaffected).
- Growth frontiers then follow valleys of the field → borders meander at all
  wavelengths. The existing `border_roughness` jitter stays as the
  fine-scale component.

## Guarantees (unchanged, with proof sketch)

- **Contiguity:** the multiplier is a symmetric positive re-weighting of
  edges; multi-source Dijkstra regions over any positive symmetric weights
  are unions of shortest-path trees → connected. (Only per-seed scaling —
  weighted growth — breaks this, and that already has the repair pass.)
- **Counts/totals:** untouched (counting machinery is independent of edge
  weights).
- **Determinism:** field drawn from the same seeded stream at a fixed point.
- **Terrain-first future:** the atom_cost slot is exactly where real
  elevation-derived cost plugs in later.

## Frontend

One new panel row: `meander` (range 0–1, step 0.05, default 0.5).

## Testing

- Spec: field default/bounds; version 0.3.0.
- Partition-level: with a synthetic high-cost band, borders preferentially
  avoid expensive atoms (mean field value on border atoms < global mean);
  contiguity fuzz at meander-equivalent cost fields with size_variance=0
  (repair off — proves the proof); determinism.
- World-level: existing invariant suite passes under new defaults;
  meander=0 vs >0 worlds differ; same spec+seed byte-identical.

## Out of scope

- Real terrain/rivers (terrain-first mode); deliberate straight "colonial"
  border segments as a stylistic knob.
