# Partition Realism Addendum

**Date:** 2026-07-27
**Status:** Approved design (discussed and accepted in session), extends
`2026-07-26-synthetic-geography-design.md`
**Generator version:** these changes alter generation output → bump
`GENERATOR_VERSION` to `0.2.0`.

## Motivation (user-reported)

1. **Disconnected units at low levels.** Island-bridge edges are currently
   used at every level, so provinces and districts can span open sea between
   comparably sized landmasses. Multi-island countries are realistic; a
   district straddling a strait is not. (Tiny islets attached to a coastal
   unit ARE realistic at every level and stay allowed.)
2. **Uniform unit sizes.** Lloyd rebalancing (added to guarantee exact unit
   counts) drove size variation down to CV ≈ 0.08; real admin units are
   heavy-tailed.
3. **Rectangular hierarchy.** `levels=[6,5,6]` gives every parent the same
   ~child count; real subdivision counts vary by orders of magnitude
   (Texas 254 counties, Delaware 3) and correlate with territory size.

## New WorldSpec parameters

| Field | Range | Default | Meaning |
|---|---|---|---|
| `size_variance` | 0.0–2.0 | 0.4 | Sigma of the log-normal growth weight per unit. 0 = current uniform-size behavior; higher = heavy-tailed unit areas. Above ~1.0 expect micro-units (1–2 atoms) as the size floor binds. |
| `count_coupling` | 0.0–1.0 | 0.7 | How strongly a parent's child count follows its territory share. 0 = uniform counts (old behavior); 1 = fully proportional. |

**Changed semantic — `count_variance`:** becomes the log-normal sigma
applied to *allocation weights* (jittering how the level's child total is
split among parents) instead of a normal jitter on per-parent counts.
Consequence (improvement): **per-level unit totals are now exact at every
`count_variance`**, not just at 0; variance moves counts *between* parents.

## Design

### 1. Size-coupled counts (levels ≥ 1, and landmass allocation at level 0)

Replace `child_counts` with weight-based allocation:

```
weights_i = (parent_atom_count_i) ** count_coupling
if count_variance > 0: weights_i *= lognormal(0, count_variance)
counts = allocate_counts(level_total, weights)      # sum exact, each >= 1
counts = redistribute_counts(counts, capacities, minimums)  # capacity/minimum clamp
```

`level_total = levels[l] × n_parents` — the `levels` list keeps its meaning
as per-parent averages.

### 2. Weighted growth (`size_variance`)

`partition_atoms` draws one log-normal weight per part
(`w = lognormal(0, size_variance, k)`, drawn once before the Lloyd loop) and
labels atoms by `argmin_i dist_i / w_i` instead of plain `argmin dist`.
The Lloyd loop keeps its two jobs — re-centering seeds to medoids and the
starved-part escape — but the starvation test becomes weight-aware:
a part is starved when its size < max(2, expected_share/8), where
expected_share = n × w_i / Σw. Weights drawn once, reused across Lloyd
iterations; determinism unchanged.

### 3. Island-aware quotas (levels ≥ 1)

Bridges (`mask.bridges`) are used **only at level 0** (countries may span
islands). For deeper levels, each parent is analyzed per landmass component:

- Components of the parent's atoms over mesh-only edges are classified:
  **sizeable** (≥ `ISLET_MAX_ATOMS = 8` atoms) vs **islets** (< 8).
- Islets attach to the nearest sizeable component (chord distance between
  closest atoms); their atoms partition together with that component and may
  end up in any of its children via the nearest-distance fallback — the
  sanctioned multi-part exception (mirrors real coastal municipalities).
- Let m = number of sizeable components, k = the parent's child count:
  - **m ≤ k:** allocate k across components via
    `allocate_counts(k, component_sizes ** count_coupling)` (each ≥ 1), then
    partition each component independently (weighted growth, no bridges).
    Every child lives on one island (+ attached islets).
  - **m > k:** group components into k proximity clusters (seeds = the k
    largest components; remaining components join the nearest seed by
    centroid chord distance); each cluster becomes exactly one child.
    Multi-island children return only in this quota-starved archipelago
    case — graceful, documented degradation.
- Per-parent minimum counts: a parent needs ≥ min(m, requested) children.
  `redistribute_counts` gains an optional `minimums` array (default all 1)
  so borrowing between parents preserves exact level totals while honoring
  per-parent island minimums when total quota allows.

### Contiguity contract (replaces "may span bridges at any level")

Units at levels ≥ 1 are contiguous except: (a) attached islets < 8 atoms,
and (b) children of quota-starved archipelago parents (m > k). Countries
(level 0) may span islands within their landmass group, unchanged.

## Frontend

Two new panel rows: `sizes` (size_variance, range 0–1 step 0.05, default
0.4) and `coupling` (count_coupling, range 0–1 step 0.05, default 0.7).

## Testing

- `coupled_counts`: exact totals at all variances; min 1; coupling 0 ↔ 1
  behavior; determinism.
- `redistribute_counts` with `minimums`.
- Weighted partitioning: size_variance 0 → CV small; 0.8 → CV > 0.3;
  non-empty/coverage/determinism preserved.
- Contiguity invariant (new, in test_world): for units at levels ≥ 1 in a
  rugged multi-landmass spec with ample quota, connected components of each
  unit's atoms (mesh edges only) — all but the largest have < 8 atoms.
- Exact per-level totals at count_variance 0.5 (new guarantee).
- Existing invariants (nesting, populations, determinism, export validity)
  must keep passing with new defaults.

## Out of scope (parked)

- Ragged hierarchies (variable depth per branch).
- Explicit per-parent count trees (`levels=[3,[12,4,7],...]`).
- Population-density-coupled unit sizes (urban compactness).
