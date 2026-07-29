# Bottom-up partition: districts-first agglomeration

**Date:** 2026-07-29 (revised after design review + prototype measurement)
**Status:** Design refined; pending re-review, then implementation plan.

## Problem

Country (level-0) borders look unnaturally straight, while province/district
borders look organic — even though the same partitioner draws all of them.

Root cause: border "texture" is generated at the fixed atom scale (~1.6° cells
at `resolution=20000`), so its amplitude is roughly constant regardless of
level. Regions are not: a district spans ~8°, a country ~51°. The same ~1.6°
wiggle is ~20% of a district border (organic) but ~3% of a country border
(straight). Today's flow draws each country border *once*, at country scale (a
smooth Voronoi bisector between far-apart seeds), and fits districts inside it —
so districts never roughen the country's outer edge.

Tuning the atom-level cost field (frequency, persistence, amplitude, clip) moved
this only modestly and traded off against partition balance: the field controls
both border shape **and** Dijkstra reachability, so strong wander starves
regions into slivers. The fix must (a) make border detail scale with the region
and (b) decouple wander from balance.

## Core idea

Invert the partitioner to **bottom-up**: carve each landmass into many small
(jagged) leaf districts, then **agglomerate** districts into provinces and
provinces into countries by **field-biased region-growing**. Because every
province is a set of whole districts and every country a set of whole provinces,
higher borders are unions of district-edge borders. The field bias makes those
borders follow a low-frequency ridge field (macro-wander), while a separate
size-balancing rule keeps units even — decoupling wander from balance.

## Prototype evidence (resolves review point A)

A real region-grow prototype (`scratchpad/proto_grow2.py`), 6 seeds, measuring
length-weighted **macro** tortuosity of interior country borders (borders
simplified at 1.2° to strip atom-teeth, so this measures macro-shape, not
staircase):

| approach | macro tortuosity |
|----------|------------------|
| top-down (today) | 1.421 |
| bottom-up, **plain** region-grow | 1.512 (≈ top-down — not worth it) |
| bottom-up, **field-biased** region-grow (β≈3) | **1.785** (+26%) |

Plain region-grow balances to equal targets, so fronts meet on a ~bisector →
only staircase, no macro-wander. Field bias makes fronts meet on ridges of a
low-frequency field → genuine wander, confirmed visually. Contiguity was perfect
(0 stray components) in **all** cases, including strong bias — balance is
enforced independently of the field.

**Acceptance criterion:** the implementation must reach interior country-border
macro tortuosity ≥ ~1.7 pooled over ≥6 seeds (vs ~1.42 top-down), plus a visual
gate. This is a pooled measurement in a validation script, **not** a per-world
unit assertion (single-world tortuosity is too noisy — it swings with seed and
resolution).

## Decisions (from brainstorming, unchanged by review)

1. **Replace** the top-down partitioner entirely (no dual code path).
2. **Exact totals, organic distribution.** Exactly the totals implied by
   `levels`; per-parent child counts fall out of agglomeration.
3. **"Landmass" = seed group** (unchanged): each land atom → nearest of
   `n_landmasses` seeds; a group is a main island plus the smaller islands
   nearest the same seed, bridged (star topology) to the main island.
4. **Islands — prefer within-landmass.** Leaf districts are per physical island;
   a too-small island is absorbed upward into the nearest unit **within its own
   group**, never across groups.
5. **Order: leaves → provinces → countries.**
6. **Grouping primitive: seeded balanced region-growing** — contiguous by
   construction, now **field-biased** for macro-wander (added per review).
7. **Knobs:** retire `count_coupling` and `count_variance`; keep `size_variance`
   (unit-size spread) and `border_roughness` (now also drives the region-grow
   field bias). Bump `GENERATOR_VERSION`.

## Alternatives considered (review point G)

- **Keep tuning the atom-level cost field** (what we shipped: coherent noise,
  frequency, persistence, clip). Reached ~+17% raw tortuosity but the field
  controls reachability too, so pushing wander further caused slivers /
  mainland-hog. Rejected: capped by the wander-vs-balance coupling.
- **Post-process border displacement** (perturb finished shared borders along
  fractal noise). Would work but must keep sibling borders shared and polygons
  valid at 3-way junctions — significant new geometry code. Superseded: the
  bottom-up structure gives scale-appropriate borders without touching geometry.
- **Retrofit district-granularity onto the top-down partitioner.** A country's
  outer edge is fixed by the country partition before districts exist; making it
  follow district edges requires districts defined *across* country boundaries
  first — which is bottom-up. Not actually cheaper.
- **Plain (unbiased) bottom-up.** Measured ≈ top-down (see above). Rejected.

## Architecture & data flow

Unchanged ends, replaced middle:

```
mesh → landmask → atom_cost
     → [Phase 1: per-island leaf partition]          (partition_atoms, reused)
     → district adjacency graph (+ bridges, ε-weighted)
     → district ridge field (low-freq sphere_noise, independent rng)
     → [Phase 2: field-biased region-grow → provinces → countries]  (new)
     → level_nodes → geometry / population / attributes → World       (reused)
```

`level_nodes` keeps its shape (`list[level] of {atoms, parent, landmass}`);
leaves are districts, `parent` set by agglomeration; the back half is untouched.
`atom_cost` still shapes leaf borders (atom-scale texture); the new district
ridge field shapes higher borders (macro-wander).

## Component 1 — per-group count allocation

Runs once. Totals from `levels` (`[6,5,6]` → 6 / 30 / 180).
- `C_g = allocate_counts(levels[0], group_sizes)` — each group ≥ 1 country
  (`levels[0] ≥ n_landmasses` stays a validation rule); proportional to size.
- Derive lower totals to keep nesting and totals exact:
  `P_g = C_g * levels[1]`, `D_g = P_g * levels[2]`. Distribution *within* a group
  is organic; group totals are fixed.

## Component 2 — per-island leaf partition (Phase 1)

For each group, partition its atoms into `D_g` single-island districts:

1. Split the group into physical islands (connected components, mesh edges only).
2. **Cluster down if needed (resolves B + islet handling):** while the number of
   island-units exceeds `D_g`, or any unit is below `MIN_ATOMS_PER_LEAF` (8),
   attach the smallest unit to its nearest unit by chord distance. This
   generalizes the old islet-attach to the sizeable-island-surplus case. A
   clustered unit spans water — the one bounded exception to single-island
   districts.
3. **Allocate `D_g` districts across units ∝ size, clamped** so no unit gets more
   districts than `unit_atoms // MIN_ATOMS_PER_LEAF`; redistribute the overflow.
   This is exactly `redistribute_counts` — **so keep it** (resolves C).
4. Partition each unit independently with `partition_atoms` (cost field,
   `border_roughness`, `size_variance`; no bridges for single islands; chord
   fallback covers clustered multi-island units).

Result: `D_g` jagged districts per group, single-island except for the bounded
cluster exception.

## Component 3 — the field-biased region-grow primitive (Phase 2 core)

Group contiguous **items** (districts, then provinces) into `K` balanced,
contiguous groups with target sizes, biased to make borders wander.

Inputs: item adjacency (edge weight = shared-border arc length; **bridge edges
get weight ε** so bridged islands are eaten last — resolves the bridge
link-metric gap in D), per-item size, per-item **ridge-field value** (sampled
from a low-frequency `sphere_noise` at item centroids, independent rng stream),
bias strength `λ` (∝ `border_roughness`), `K` target sizes, rng.

```
1. Seeds:    K farthest-point seeds on item centroids.
2. Targets:  K sizes summing to total mass, log-normal spread ~ size_variance.
3. Grow:     until all items assigned —
               • group selection: lowest (filled/target) ratio with non-empty
                 frontier  → BALANCE;
               • item selection within that group: maximize
                 (link_weight − λ·field[item]) + tiny rng tie-break, iterating
                 the frontier in SORTED item-id order → WANDER + determinism.
4. Stranded: any item never reached attaches to the nearest assigned item by
             chord (safety net; with ε bridges this rarely fires).
```

Properties:
- **Contiguous by construction** — no repair pass.
- **Balanced** — feed-most-behind pulls sizes to targets (independent of field).
- **Macro-wander** — eating low-field items first leaves borders on the field's
  high ridges; low frequency ⇒ wander at country scale. Measured +26% macro
  tortuosity; contiguity unaffected.
- **Exact K**, deterministic (canonical sorted iteration — resolves F).

## Component 4 — hierarchical driver (Phase 2)

Per group `g`:
1. Districts from Phase 1.
2. District adjacency graph: mesh edges (arc-length weight) **plus** within-group
   bridges at weight ε. This is the single, coherent island-absorption mechanism
   (resolves D): a secondary island is reachable only via its ε bridge to the
   main island, so it is picked up — last — by the group owning that bridge
   endpoint. No separate "stranded vs frontier" ambiguity.
3. Field-biased region-grow districts → `P_g` provinces.
4. Province adjacency graph (province-province shared borders; bridges inherited);
   field-biased region-grow provinces → `C_g` countries.
5. Emit `level_nodes` with parent pointers and `landmass = g` on countries.

## Knob mapping (Option A)

| knob | role |
|------|------|
| `n_landmasses`, `spread`, `coast_ruggedness`, `land_fraction` | landmask only |
| `border_roughness` | atom-level leaf texture **and** region-grow field-bias strength λ (macro-wander) |
| `border_meander` | elevation term in the leaf cost field |
| `size_variance` | spread of target sizes at every level (see caveat below) |
| `resolution`, `total_population`, `seed`, `levels` | unchanged |
| ~~`count_coupling`~~, ~~`count_variance`~~ | removed |

Caveat (minor review point): `size_variance` acts through two transfer functions
— a log-normal *weight* in the leaf weighted-Voronoi vs a log-normal *target* in
region-grow — so the same numeric value won't produce identical spread at every
level. Acceptable, but the "one clean knob" framing is approximate.

## Size diversity (review point E)

Region-grow fills log-normal *targets*, so unit sizes are as heavy-tailed as
`size_variance` makes them — it does not force equal area (equalization happens
only at `size_variance=0`, as before). What is removed is the independent
size↔count tuning the old `coupled_counts` gave. **Validation target:** confirm
the level-0 area distribution's coefficient of variation at the default
`size_variance` is within ~20% of today's, so worlds don't read as uniformly
sized.

## Code changes

**Reused unchanged:** `partition_atoms` (+ helpers, seeds, cost field);
`allocate_counts`; `redistribute_counts`; all geometry/population/elevation/
naming/`Unit`/`World`.

**Removed:** `generate.py` top-down loop; `partition.py` `coupled_counts`,
`honor_minimums`, `plan_islands`, `_cluster_islands`, `count_sizeable_islands`,
`_island_analysis`, `ISLET_MAX_ATOMS`; `spec.py` `count_coupling`,
`count_variance` (+ `GENERATOR_VERSION` bump); web `coupling`/`counts` sliders
and their fields.

**Added — new module `agglomerate.py`:** `region_grow(...)` (Component 3), the
per-island leaf partition with island clustering (Component 2), and the
hierarchical driver (Component 4). `generate.py`'s core is rewritten to call it.

**Notes (minor review points):**
- *Performance:* Phase 1 makes many small `partition_atoms` calls (one per island
  unit) instead of a few large ones; each builds a csr_matrix + Dijkstra + 3
  Lloyd rounds. Expected fine at these sizes but must be timed against the
  current pipeline before merge.
- *`partition_atoms` reuse:* its substantial-island seeding and starved-part
  escape go inert when called per single connected island — harmless dead weight;
  simplifying it is optional cleanup, deferred (YAGNI).

## Testing

**Kept / adapted (property tests):** exact totals per level; every atom in
exactly one leaf; children exactly tile parents; every unit contiguous;
determinism; geometries valid; population sums.

**New:**
- `region_grow` returns exactly K contiguous groups covering all items, for both
  plain and field-biased modes, across seeds.
- leaf districts single-island except the bounded cluster exception; every
  district ≥ `MIN_ATOMS_PER_LEAF`.
- a too-small island is absorbed by a within-group neighbor and **never** across
  groups.
- determinism under canonical ordering (same seed → identical `level_nodes`).
- **macro-wander (replaces the tautological border-inheritance test):** a
  pooled validation script asserts interior country-border macro tortuosity
  ≥ ~1.7 over ≥6 seeds, materially above the top-down baseline. Documented as a
  validation gate, not a flaky per-world unit test.

**Deleted:** tests bound to removed functions.

## Backward compatibility (review point G)

`spec.py` uses `extra="forbid"`, so a saved spec/URL carrying `count_coupling` /
`count_variance` will 422 rather than degrade. To avoid breaking shared web
links, the **frontend strips unknown fields before POST** (and drops the retired
query params), so old links load with current defaults instead of erroring. The
Python API keeps `forbid` (explicit is correct for programmatic callers); the
version bump documents the break.

## Risks / edge cases

- **`#islands > D_g` / sub-MIN islets:** handled by the Component-2 clustering
  step; every district ends ≥ `MIN_ATOMS_PER_LEAF`.
- **Group too small for `D_g` drawable districts:** the global validation bounds
  this in aggregate; per-group, clustering reduces unit count until feasible, and
  if still infeasible the spec validation should catch it up front.
- **Determinism:** all seeds, targets, tie-breaks draw from `spec.seed` in fixed
  order; region-grow iterates items in sorted-id order.
- **Field-bias strength:** too high can over-fragment province shapes; `λ ≈ 3`
  (∝ `border_roughness`) was the measured sweet spot — tune during implementation
  and re-measure macro tortuosity + contiguity.
- **Visual validation** at max settings before calling it done.
