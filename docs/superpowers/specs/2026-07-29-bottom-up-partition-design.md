# Bottom-up partition: districts-first agglomeration

**Date:** 2026-07-29
**Status:** Design approved; ready for implementation plan.

## Problem

Country (level-0) borders look unnaturally straight, while province/district
borders look organic — even though the same partitioner draws all of them.

Root cause: border "texture" is generated at the fixed atom scale (~1.6° cells
at `resolution=20000`), so its amplitude is roughly constant regardless of
level. But regions are not constant: a district spans ~8°, a country ~51°. The
same ~1.6° wiggle is ~20% of a district border (reads as organic) but only ~3%
of a country border (reads as straight). Today's flow draws each country border
*once*, at country scale (a smooth Voronoi bisector between far-apart seeds),
and fits districts inside it — so districts never touch, and never roughen, the
country's outer edge.

Tuning the cost field (frequency, persistence, amplitude, clip) only moved this
modestly and traded off against partition balance. The fix must make border
detail **scale with the region** instead of being fixed at the atom scale.

## Core idea

Invert the partitioner to be **bottom-up**. Carve each landmass into many small
(jagged) leaf districts first, then **agglomerate** districts into provinces and
provinces into countries. Because every province is a set of whole districts and
every country a set of whole provinces, all higher borders are unions of
district-edge borders — so country borders inherit district-scale meander
(~8° wavelength) instead of atom-scale texture on a smooth line.

This dissolves, by construction, the three problems we fought under the
top-down model: straight country borders, mainland-hog provinces, and slivers.

## Decisions (settled during brainstorming)

1. **Replace** the top-down partitioner entirely (no dual code path).
2. **Exact totals, organic distribution.** The world has exactly the totals
   implied by `levels`; which parent gets how many children falls out of the
   agglomeration (soft, not a hard per-parent target).
3. **"Landmass" = seed group** (unchanged from today): the landmask assigns
   every land atom to its nearest of `n_landmasses` seeds; a group is a main
   island plus whatever smaller islands are nearest the same seed, bridged
   together within the group.
4. **Islands — prefer within-landmass (Option B).** Leaf districts are per
   physical island (never straddle water). A too-small island is absorbed
   upward into the nearest unit **within its own group**, never across groups.
5. **Order: leaves → provinces → countries** (true bottom-up merge).
6. **Grouping primitive: seeded balanced region-growing** — contiguous by
   construction (no repair pass).
7. **Knobs — Option A:** retire `count_coupling` and `count_variance`; keep
   `size_variance`. Count-follows-size emerges from region-growing to size
   targets, so the two count knobs are redundant. Bump `GENERATOR_VERSION`.

## Goals / non-goals

**Goals**
- Country and province borders that meander at the granularity of their
  children (organic at every level).
- Preserve every structural guarantee: exact per-level totals, children exactly
  tile parents, every unit contiguous, deterministic from `seed`.
- Keep the whole geometry/population/attribute back half untouched.

**Non-goals**
- No change to the landmask, elevation, or the meaning of `n_landmasses`,
  `spread`, `coast_ruggedness`, `land_fraction`.
- No post-process border displacement (the districts-first structure supersedes
  it).
- No backward compatibility with old specs that set `count_coupling` /
  `count_variance` (they are removed; version bump signals the break).

## Architecture & data flow

Unchanged ends, replaced middle:

```
mesh → landmask → atom_cost
     → [Phase 1: per-island leaf partition]         (partition_atoms, reused)
     → district adjacency graph (incl. within-group bridges)
     → [Phase 2: agglomerate → provinces → countries] (region-grow, new)
     → level_nodes  → geometry / population / attributes → World   (reused)
```

- **`level_nodes`** keeps its exact shape: `list[level] of {atoms, parent,
  landmass}`. Leaves are the districts; `parent` pointers are set by the
  agglomeration. The back half already consumes this and is not modified.
- **`atom_cost`** (elevation-meander + coherent border-noise, clipped) continues
  to shape the *leaf* borders. Since all higher borders are unions of leaf
  borders, `border_roughness` and `border_meander` now propagate up to every
  level — they are more effective than before, not less.

## Component 1 — per-group count allocation

Runs once, up front. Totals come from `levels` (e.g. `[6,5,6]` → 6 countries,
30 provinces, 180 districts).

- Countries per group: `C_g = allocate_counts(levels[0], group_sizes)` — each
  group ≥ 1 country (so `levels[0] ≥ n_landmasses` remains a validation rule).
  Allocation is proportional to group size (count-follows-size; no coupling
  exponent).
- Nesting is kept exact by deriving the lower totals from the country counts:
  `P_g = C_g * levels[1]`, `D_g = P_g * levels[2]`. Summed across groups these
  equal the global totals exactly. Distribution *within* a group is organic
  (see Component 3), but the group totals are fixed.

## Component 2 — per-island leaf partition (Phase 1)

For each landmass group, partition its atoms into `D_g` districts such that no
district straddles water:

- Split the group into physical islands (connected components over mesh edges
  only, ignoring bridges).
- Allocate `D_g` districts across islands ∝ island size, each island large
  enough to host a leaf getting ≥ 1 district.
- Partition each island independently with `partition_atoms` (cost field,
  `border_roughness`, `size_variance`; **no bridges**, so districts stay within
  one island).
- **Islet exception:** an island below `MIN_ATOMS_PER_LEAF` (8 atoms) cannot be
  a drawable leaf; it attaches to its nearest island (chord distance) and is
  partitioned with it. This is the only case where a district may include
  across-water specks, and it is bounded to sub-8-atom islets.

Result: `D_g` jagged districts per group, each a single physical island (modulo
the islet exception).

## Component 3 — the region-grow primitive (Phase 2 core)

Group a set of contiguous **items** (districts, or provinces) into `K` balanced,
contiguous groups with target sizes.

Inputs: item adjacency graph, per-item size (atom count), `K` target sizes, rng.

```
1. Seeds:    K well-spread seed items via farthest-point sampling on item
             centroids. Each seed starts one group.
2. Targets:  K target sizes summing to the total item mass, drawn with
             log-normal spread controlled by size_variance.
3. Frontier: each group tracks its unassigned adjacent items.
4. Grow:     until every item is assigned —
               • pick the group with the lowest (filled / target) ratio that
                 still has a non-empty frontier;
               • add its best frontier item = strongest link (largest shared
                 border length) with a small rng tie-break so borders wander;
               • update frontiers.
5. Stranded: any item never reached (e.g. a small island reachable only across
             water) attaches to the group owning its nearest item by chord
             distance.
```

Properties:
- **Contiguous by construction** — a group only ever gains a touching item; no
  contiguity-repair pass is needed.
- **Balanced** — always feeding the most-behind group pulls sizes to targets.
- **Exact K** — K seeds → exactly K groups, every item assigned once.
- **Jagged borders** — a group boundary is a chain of whole-item edges; the rng
  tie-break keeps it from growing in regular rings.

## Component 4 — hierarchical driver (Phase 2)

Per landmass group `g`:
1. Districts from Phase 1.
2. Build the district adjacency graph (mesh edges **plus** within-group bridges,
   so a small island's district can be reached across water).
3. Region-grow districts → `P_g` provinces.
4. Build the province adjacency graph; region-grow provinces → `C_g` countries.
5. Emit `level_nodes` entries with `parent` pointers (district→province,
   province→country) and `landmass = g` on countries.

Island absorption is automatic: a small island too small to host a province or
country seed simply gets no seed and is picked up by the region-grower's
"stranded → nearest within-group neighbor" step. No dedicated island machinery.

## Knob mapping (after Option A)

| knob | role in the new model |
|------|-----------------------|
| `n_landmasses`, `spread`, `coast_ruggedness`, `land_fraction` | landmask only — unchanged |
| `border_roughness`, `border_meander` | shape the leaf-district borders (cost field); propagate to all levels |
| `size_variance` | spread of target sizes at every level (districts via `partition_atoms`, provinces/countries via region-grow targets); count-follows-size is emergent |
| `resolution`, `total_population`, `seed`, `levels` | unchanged |
| ~~`count_coupling`~~, ~~`count_variance`~~ | **removed** |

## Code changes

**Reused unchanged:** `partition_atoms` and its helpers (`_subgraph`,
`_assign_labels`, `_repair_contiguity`, `pick_seeds`); `allocate_counts`; all of
geometry, population, elevation, naming, `Unit`/`World`.

**Removed:**
- `generate.py`: the top-down level loop.
- `partition.py`: `coupled_counts`, `honor_minimums`, `plan_islands`,
  `_cluster_islands`, `count_sizeable_islands`, `_island_analysis`,
  `ISLET_MAX_ATOMS`. (Keep `redistribute_counts` only if the count allocation
  needs a capacity clamp; otherwise remove.)
- `spec.py`: `count_coupling`, `count_variance`; bump `GENERATOR_VERSION`.
- Web: `coupling` and `counts` sliders (`index.html`), and their fields in
  `panel.ts` / `api.ts`.

**Added — new module `agglomerate.py`** (keeps `partition.py` focused):
- `region_grow(adj, sizes, targets, rng)` — Component 3.
- per-island leaf partition — Component 2.
- hierarchical driver — Component 4.
- `generate.py`'s partitioning core is rewritten to call these; the back half
  stays.

## Testing

**Kept / adapted (property tests):**
- exact totals per level; every atom in exactly one leaf; children exactly tile
  parents; every unit contiguous; determinism (same seed → same world);
  geometries valid and in range; population sums preserved.

**New:**
- `region_grow` returns exactly K contiguous groups covering all items.
- leaf districts are single-island (except sub-`MIN_ATOMS_PER_LEAF` islets).
- a too-small island is absorbed by a within-group neighbor and **never** merges
  across groups.
- structural border-inheritance check: the set of mesh edges on any
  province/country boundary is a subset of the district-boundary edges (higher
  borders are made of district borders).

**Deleted:** tests bound to removed functions (`coupled_counts`,
`plan_islands`, `honor_minimums`, island-analysis).

## Risks / edge cases

- **More physical islands than districts in a group** (`#islands > D_g`): the
  smallest islets attach to their nearest island for the leaf partition (islet
  exception), so each district still meets `MIN_ATOMS_PER_LEAF`.
- **A group with `C_g` countries but very few large islands**: region-growing on
  a sparse district graph must still yield `C_g` contiguous countries; the
  stranded step guarantees full assignment.
- **Determinism**: all seeds, target-size jitter, and tie-breaks draw from the
  `spec.seed` stream in a fixed order.
- **Balance under a strong cost field**: leaf partition still uses the clipped
  `atom_cost`; the region-grower balances at the district level independent of
  the cost field, so the old sliver/lopsided-split failure mode does not recur.
- **Visual validation**: before calling it done, render country/province borders
  at max settings and confirm the district-scale meander seen in the prototype.
