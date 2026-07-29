# Bottom-up partition: districts-first agglomeration

**Date:** 2026-07-29 (revised after two design reviews + prototype measurements)
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

A second measurement (`scratchpad/proto_bal.py`, full hierarchy) checked
**size balance**, which the second review flagged (H/I):

| metric | top-down | bottom-up (biased) |
|--------|----------|--------------------|
| country size CV (mean of 6 seeds) | 0.71 | 0.31 |
| country min/max ratio (mean) | 0.15 (worst seed 0.03) | 0.46 |
| provinces per country (range) | — | 2–8 |

Region-grow is *better* balanced than today's `partition_atoms` on average, and
provinces-per-country varies organically (2–8). But one seed stranded a single
district (isolated in the district graph) → an empty province → so a **straggler
guard** is required (see Component 3), even though the average balance premise of
review H is refuted. Full Lloyd/starved-escape rebalancing is **not** added
(YAGNI given these numbers); a single straggler-attach pass suffices.

## Decisions (from brainstorming, unchanged by review)

1. **Replace** the top-down partitioner entirely (no dual code path).
2. **Exact totals; organic distribution is real but not free (review J).**
   Exactly the totals implied by `levels`. Per-parent child counts *do* vary
   organically (measured: 2–8 provinces per country) — this arises from (a) the
   coarse-item quantization at the country level and (b) `size_variance`'s
   log-normal target spread, **not** as a cost-free structural property of
   agglomeration. `size_variance` is the knob that governs how much diversity
   there is; see the size-diversity target below.
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
- **Per-group feasibility (resolves review L).** Exact nesting forces the
  smallest group to host `D_g` districts, i.e. ≥ `D_g * MIN_ATOMS_PER_LEAF`
  atoms (240 at defaults). The current global check (`spec.py:71`) does not cover
  this. Add an up-front validation: after count allocation, every group must have
  `group_atoms >= D_g * MIN_ATOMS_PER_LEAF`. If not, the landmask retry loop
  (which already retries seed placement, `landmask.py:35`) is asked to produce
  less-lopsided groups; if it still fails after its retries, `generate` raises a
  clear error (lower `n_landmasses`, raise `resolution`/`land_fraction`, or lower
  `spread`) — matching the landmask's existing raise. Exact totals are preserved;
  infeasible configs fail loudly rather than emitting non-drawable districts.

## Component 2 — per-island leaf partition (Phase 1)

For each group, partition its atoms into `D_g` single-island districts:

1. Split the group into physical islands (connected components, mesh edges only).
2. **Cluster down if needed (resolves B + islet handling):** while the number of
   island-units exceeds `D_g`, or any unit is below `MIN_ATOMS_PER_LEAF` (8),
   attach the smallest unit to its nearest unit by chord distance. This
   generalizes the old islet-attach to the sizeable-island-surplus case. A
   clustered unit spans water — the one bounded exception to single-island
   districts. (Review Q: this intentionally collapses the old `ISLET_MAX_ATOMS`
   vs `MIN_ATOMS_PER_LEAF` distinction — `partition.py:14-20` kept them separate;
   bottom-up uses the single `MIN_ATOMS_PER_LEAF` threshold and drops
   `ISLET_MAX_ATOMS`.)
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
link-metric gap in D), per-item size, per-item **ridge-field value**, bias
strength `λ` (∝ `border_roughness`), `K` target sizes, rng.

**Ridge field (resolves review O).** Sampled at item centroids from a
low-frequency field with the same structure as `atom_cost` —
`meander·elevation_z + roughness·noise` — on an independent rng stream. Using
elevation (not just noise) makes `border_meander` propagate to macro borders:
country/province borders settle along real elevation crests, not only along
random ridges. (The prototype used noise-only and already hit +26%; adding the
elevation term is a strict improvement to validate.)

```
1. Seeds:    K farthest-point seeds on item centroids. (Seeding is a tunable;
             FPS already delivered the +26% wander, but jittered/interior seeds
             are worth trying — review N.)
2. Targets:  K sizes summing to total mass, log-normal spread ~ size_variance.
3. Grow:     until all items assigned —
               • group selection: lowest (filled/target) ratio with non-empty
                 frontier  → BALANCE;
               • item selection within that group: maximize
                 (link_weight − λ·field[item]) + tiny rng tie-break, iterating
                 the frontier in SORTED item-id order → WANDER + determinism.
4. Straggler guard (resolves review H/K): if the loop ends with any item still
   unassigned (an item isolated in the graph — measured to happen rarely), attach
   it to an ADJACENT assigned group (its strongest-link neighbor's group), never
   by chord distance. Chord-attach is banned because it produces geographically
   disconnected units (the failure that forced `_repair_contiguity` to exist).
   A truly neighborless item is a leaf-partition bug to fix upstream, not to
   paper over here.
```

Properties:
- **Contiguous by construction** — no repair pass; the straggler guard only ever
  attaches across an existing graph edge, so contiguity holds.
- **Balanced (best-effort, measured good)** — feed-most-behind pulls sizes to
  targets independent of the field. Measured country CV 0.31 vs top-down 0.71;
  min/max 0.46 vs 0.15. Not a hard guarantee (a boxed-in seed stops under
  target), but empirically better than the partitioner it replaces. No full
  Lloyd/starved-escape rebalance is added (YAGNI given the numbers).
- **Macro-wander** — eating low-field items first leaves borders on the field's
  high ridges; low frequency ⇒ wander at country scale. Measured +26% macro
  tortuosity; contiguity unaffected.
- **Exact K**, deterministic (canonical sorted iteration — resolves F).

## Component 4 — hierarchical driver (Phase 2)

Per group `g`:
1. Districts from Phase 1.
2. District adjacency graph: mesh edges (arc-length weight) **plus** within-group
   bridges at weight ε. Because the landmask bridges every secondary island to
   the group's main island (star topology, `landmask.py:75-84`), **the within-
   group district graph is always connected — a stated invariant** (resolves K).
   Island absorption is one coherent mechanism (resolves D/K): a secondary island
   is reachable only via its ε bridge, so it is picked up — last — by the group
   owning the district at the bridge's main-island endpoint (the nearest mainland
   region, not an arbitrary far country). No chord-attach; the invariant makes it
   unnecessary and the straggler guard is graph-adjacent only.
3. Field-biased region-grow districts → `P_g` provinces.
4. Province adjacency graph (province-province shared borders; bridges inherited);
   field-biased region-grow provinces → `C_g` countries.
5. Emit `level_nodes` with parent pointers and `landmass = g` on countries.
   Groups are built independently then concatenated into the flat global
   `level_nodes[level]`, so each group's parent indices must be **offset by that
   group's base position** in the global list (resolves review Q — off-by-one
   prone; assemble with explicit per-group offsets).

## Knob mapping (Option A)

| knob | role |
|------|------|
| `n_landmasses`, `spread`, `coast_ruggedness`, `land_fraction` | landmask only |
| `border_roughness` | atom-level leaf texture **and** region-grow field-bias strength λ (macro-wander). **Collapsed to a scalar** (review M): the per-level list form is removed since only the leaf field consumed it and the list mapping is undefined in bottom-up; passing a list is rejected in validation. |
| `border_meander` | elevation term in the leaf cost field **and** in the region-grow ridge field (so meander propagates to macro borders — review O) |
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
- **balance (review P/H/I):** country and province atom-size CV stays below a
  documented bound (measured ~0.31 for countries; assert ≤ ~0.45 pooled), and the
  straggler guard leaves no unassigned item and no empty group.
- **per-group feasibility (review P/L):** a small lopsided landmass either
  generates valid drawable districts or raises the clear feasibility error — a
  regression test with a deliberately small group asserts one of those two, never
  a silent sub-`MIN_ATOMS` district.
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
- **Group too small for `D_g` drawable districts (review L):** the new per-group
  feasibility check (Component 1) catches this up front — landmask retries for
  less-lopsided groups, else `generate` raises with actionable guidance. Exact
  totals preserved; no silent sub-`MIN_ATOMS` districts.
- **Region-grow balance is best-effort, not guaranteed (review H/I):** a boxed-in
  seed can stop under target; measured balance is nonetheless better than
  top-down (CV 0.31 vs 0.71). The straggler guard prevents unassigned items;
  balance is asserted by test, not by construction.
- **Determinism:** all seeds, targets, tie-breaks draw from `spec.seed` in fixed
  order; region-grow iterates items in sorted-id order.
- **Field-bias strength:** too high can over-fragment province shapes; `λ ≈ 3`
  (∝ `border_roughness`) was the measured sweet spot — tune during implementation
  and re-measure macro tortuosity + contiguity.
- **Visual validation** at max settings before calling it done.

## Implementation deviations (recorded post-build)

Two design details changed during implementation, surfaced by the review loop:

- **Leaf partition is whole-group, not per-island.** Component 2's per-island
  split + clustering gave the mainland only one district when a group had ~as
  many islands as districts (mainland-hog at the leaf level) and lost the
  `atom_cost` crest-following. `leaf_partition` now runs a single
  `partition_atoms` over the whole group **with within-group bridges** (small
  islands absorbed via bridge; mainland subdivided; meander restored), followed
  by a **count-preserving sliver-repair** pass (merge any sub-`MIN_ATOMS_PER_LEAF`
  district into its strongest-link neighbour, re-split the largest) so the
  "every district ≥ MIN" guarantee holds. `redistribute_counts` is still kept
  (used by `allocate_group_counts`).
- **Region-grow bias `lam` is the constant `GROW_BIAS`, not `GROW_BIAS * roughness`.**
  `grow_field` already scales with both knobs; multiplying by roughness again
  zeroed meander's macro effect when roughness=0. Constant `lam` lets
  `border_meander` bend macro borders on its own (matches the validated prototype).
- **`scripts/prebake.py`** curated specs were updated to drop the retired
  `count_coupling`/`count_variance` fields (add to the "Removed" surface).

Final acceptance on the real pipeline (6 seeds, res 20000): interior
country-border macro tortuosity **1.93** (≥1.60), country area CV **0.25**
(≤0.45), 0 sliver leaves; full test suite green (114).
