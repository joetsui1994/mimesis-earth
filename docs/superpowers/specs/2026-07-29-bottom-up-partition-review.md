# Review: Bottom-up partition design spec

**Reviews:** `2026-07-29-bottom-up-partition-design.md`
**Date:** 2026-07-29
**Reviewer:** Claude (critical design review)

## Verdict

The core diagnosis is **correct and well-argued**, and the bottom-up inversion is a
legitimate, clean way to fix it. The root-cause analysis (country borders are drawn
*once* at country scale as a smooth bisector, so outer district edges are born smooth)
holds up against the code: `generate.py:200` already unions children into parents, so a
country polygon is *already* a union of district geometries. What's wrong today is *when*
the outer edge is drawn, not *that* it's a union.

But the document oversells its central claim, has two concrete correctness gaps in island
handling, and lacks an alternatives section. The status "Design approved; ready for
implementation plan" is **premature**. Resolve A–D before writing the plan.

---

## Critical issues

### A. The central claim is unquantified, and the test that "proves" it is tautological

The whole spec exists to make country borders meander, yet this is the least-evidenced
part of the doc.

- The **structural border-inheritance test** ("province/country boundary edges ⊆
  district-boundary edges") is **true by construction** the moment provinces are unions of
  whole districts. Every province-boundary mesh edge separates two districts in different
  provinces, so it *is* a district edge, always. The test passes trivially and validates
  nothing agglomeration doesn't already guarantee. It gives false confidence.
- The actual goal (does it *look* organic?) is punted entirely to eyeballing ("confirm the
  district-scale meander seen in the prototype").

Substantive worry the spec doesn't address: **region-grow balances to equal size
targets**, so two countries' growth fronts meet roughly along a Voronoi bisector between
their seeds — i.e. still *roughly straight at country scale*, just quantized into ~8°
district steps. That raises texture amplitude ~5× (1.6°→8°), which genuinely helps — but
it is *staircase roughness on a straight line*, not the *wandering macro-shape* the word
"meander" implies. The "small rng tie-break" produces only mild wander by design.

Contrast the rigor already in the codebase: `generate.py:36-37` and `:50-55` cite
*measured* effects ("+17% border tortuosity", "straight-pair fraction 18%→5%"). The spec
claims a prototype exists but gives **no measured tortuosity for the new approach**. For
the one number that matters most, the doc is weaker than the code it replaces.

**Action:** put the prototype's country-border tortuosity (or straight-pair fraction) in
the spec as an acceptance threshold. Replace the tautological border-inheritance test with
a real *tortuosity* assertion.

### B. Correctness gap: "#sizeable islands > D_g" has no handling

Component 2 allocates `D_g` districts across islands ∝ size, "each island large enough to
host a leaf getting ≥ 1 district," and explicitly uses **no bridges**. If a group has more
sizeable (≥ 8-atom) islands than `D_g` districts, some island gets **zero districts** —
yet every atom must land in exactly one district. With no bridges in Phase 1 you cannot
form a contiguous district spanning two islands, so this case either strands atoms (breaks
"every atom in exactly one leaf") or forces an across-water district (breaks "single-island
districts").

Today `_cluster_islands` / `plan_islands` (`partition.py:316-447`) handle exactly this, and
the spec removes them with **no replacement**. The risk section only covers sub-8-atom
*islets*, not the sizeable-island surplus.

**Action:** design a replacement — generalize the islet-attach rule to cluster the
smallest sizeable islands onto their nearest neighbor for the leaf partition.

### C. Per-island allocation doesn't guarantee each district ≥ MIN_ATOMS_PER_LEAF

The global validation (`spec.py:71-79`) only ensures leaves are resolvable *in aggregate*.
Island sizes are emergent from the landmask, so a 20-atom island allocated 3 districts
yields ~6.7 atoms/district < 8 — a non-drawable leaf. Component 2 allocates ∝ size but
describes no per-island capacity clamp, so the Risk-section claim "each district still
meets MIN_ATOMS_PER_LEAF" is **not guaranteed by the mechanism described**.

The clamp needed is `districts_on_island ≤ island_atoms // MIN_ATOMS` with overflow
redistributed — which is precisely `redistribute_counts`, the function the spec is unsure
whether to keep ("keep only if… otherwise remove"). The requirement resolves the
uncertainty: **it is needed.**

### D. Island absorption describes two mechanisms that disagree

- Component 4 step 2 puts **within-group bridges into the district graph** "so a small
  island's district can be reached across water" — meaning it is reached by *frontier
  growth*, absorbed by whichever country's front crosses the bridge first (balance-driven →
  possibly a far country).
- The absorption story says the island "gets no seed and is picked up by the region-grower's
  *stranded → nearest within-group neighbor* step" — i.e. chord-distance nearest.

If bridges make it reachable, it is **not stranded**, so the "nearest" step never fires;
frontier growth wins and may pick a non-nearest country. These are inconsistent.

Relatedly, Component 3's link metric is "**largest shared border length**," which is
**undefined for a bridge edge** (a bridge is not a shared border). Note the landmask uses a
star topology — every secondary island bridges only to the group's *main* island
(`landmask.py:75-84`), never to another secondary island.

**Action:** pick *one* absorption mechanism and give bridge edges an explicit link-strength
rule (e.g. ε, so bridged islands attach last, consistent with "stranded → nearest").

---

## Secondary issues

### E. Size-diversity regression (unvalidated)

Balanced region-grow **equalizes atom-mass at every level**. Combined with retiring
`count_coupling`, all cross-unit size variety now rests entirely on `size_variance`'s
symmetric log-normal jitter. The old `coupled_counts` (`partition.py:263`) produced
heavy-tailed size distributions; the new model drives countries toward *equal area*. Real
geographies are heavy-tailed. The spec calls this "organic distribution" but the mechanism
actively flattens size diversity. Whether `size_variance` alone restores enough spread is
untested and deserves a stated target.

### F. Determinism is more fragile than claimed

The spec asserts determinism but the new pipeline has far more RNG consumers in a more
complex order (per-island loops of variable count, then seeds + targets + tie-breaks at two
region-grow levels per group). Region-grow's frontier/tie-break draws must iterate items in
a **canonical order** (sorted by item id), or dict/set iteration order silently breaks
reproducibility. Mandate canonical ordering explicitly — today's `pick_seeds` already
depends on fixed draw order (`partition.py:24`).

### G. No alternatives section; hard backward-compat break

For an inversion this large there is no "alternatives considered" (e.g. why not retrofit
district-scale outer edges onto the top-down partitioner, keeping most of it). The
inversion is defensible, but the doc should say why it beats cheaper options.

Separately: `spec.py:17` sets `extra="forbid"`, so any saved spec/URL carrying
`count_coupling` / `count_variance` will now **hard-error (422)**, not degrade gracefully —
a UX cliff for shareable web URLs. "Version bump signals it" doesn't help a user with an old
link; consider stripping unknown fields client-side.

### Minor

- **`size_variance` overloaded across two transfer functions**: log-normal *weight* in
  weighted-Voronoi (leaf) vs. log-normal *target* in region-grow (province/country) respond
  differently to the same numeric value — the "one clean knob" framing is optimistic.
- **Performance unmentioned**: the shift from few-large to many-small `partition_atoms`
  calls (each builds a `csr_matrix` + dijkstra + 3 Lloyd rounds, `partition.py:160-225`) per
  island isn't discussed.
- **`partition_atoms` reused "unchanged"** carries multi-island machinery
  (substantial-island seeding `:166-173`, starved-part escape `:200-209`) that becomes
  near-inert when called per single connected island — dead weight, and its tuning was for
  the old caller.

---

## What's genuinely strong

- Root-cause analysis is correct and matches the code.
- "Contiguous by construction, no repair pass" is a real simplification over today's
  `_repair_contiguity` (`partition.py:97`).
- Preserving the back half and `level_nodes` shape correctly scopes the blast radius.
- Routing island containment through the region-grower rather than dedicated machinery is
  elegant — *if* B and D are resolved.

---

## Bottom line

Approve the direction. Before writing the implementation plan, the spec needs:

1. **(A)** a quantified border-meander acceptance criterion + a non-tautological test;
2. **(B)** a designed answer for sizeable-island surplus;
3. **(C)** the per-island MIN-atoms clamp (keep `redistribute_counts`);
4. **(D)** one coherent island-absorption mechanism with a defined bridge-edge link metric.

E–G each warrant a paragraph.
