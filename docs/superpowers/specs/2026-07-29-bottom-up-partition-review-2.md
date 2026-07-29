# Second review: Bottom-up partition design spec

**Reviews:** `2026-07-29-bottom-up-partition-design.md`
**Date:** 2026-07-29
**Reviewer:** Claude (second pass — deeper on the region-grow algorithm)

This is an independent second pass. It does **not** repeat the first review
(`2026-07-29-bottom-up-partition-review.md`, findings A–G). Everything below is new,
focused on the region-grow primitive's actual behavior, per-group feasibility, and internal
contradictions in the island-handling story.

---

## Critical issues

### H. Region-grow balance is best-effort, not a guarantee — and there is no escape hatch

Component 3 lists "**Balanced** — always feeding the most-behind group pulls sizes to
targets" as a *property*. It isn't one. The rule only feeds "the group with the lowest
filled/target ratio **that still has a non-empty frontier**." A seed that gets boxed in —
all its graph neighbors already taken by other groups — stops growing while still under
target, and the remaining mass is absorbed by whatever over-target groups still touch it.
Final sizes can then deviate arbitrarily from targets, in exactly the topology-constrained
cases (peninsulas, corners, narrow isthmuses) where it matters most.

The current `partition_atoms` survives this because it has **two** rebalancing mechanisms
the new primitive drops entirely:

- Lloyd relocation (`partition.py:185-196`) — moves each seed to its part's medoid and
  reassigns, 3 rounds.
- A **starved-part escape** (`partition.py:200-209`) — a part far below its expected size
  gets its seed teleported to the farthest atom of the largest part.

`region_grow` as specified is single-pass with no relocation and no starvation rescue. So
on constrained graphs it can produce worse balance than the top-down partitioner it
replaces — the opposite of the spec's framing. **The spec needs to either add a rebalance
pass or state (and test) an explicit worst-case balance bound.**

### I. Balance degrades up the hierarchy because item granularity coarsens

`region_grow` hits size targets by adding **whole items**. At the province level the items
are districts (small relative to a province) so the granularity error is small. At the
country level the items are **provinces** — few and large relative to a country. A country
target of ~5 provinces built from ~equal-mass provinces can be off by ±one province ≈ 20%,
before any boxed-in effect (H) is even considered.

So the coarsest, most visually prominent level (countries) is where balance is *weakest*,
and it degrades precisely as you climb. The spec treats region-grow as uniformly "balanced"
at every level; it is not. This compounds H and should be called out with a per-level
expectation.

### J. Decision 2 ("organic distribution") contradicts balanced region-grow

Decision 2 promises "**exact totals, organic distribution** … which parent gets how many
children falls out of the agglomeration (soft, not a hard per-parent target)." But
Component 3 **balances every level to equal-mass targets**. Equal-mass countries built from
equal-mass provinces have ~equal *province counts* too. So balancing actively *suppresses*
the children-per-parent variation Decision 2 sells as an emergent feature. The only surviving
source of variation is `size_variance`'s log-normal jitter — the same single knob doing all
the work flagged in the first review (finding E).

In other words: the "organic distribution falls out of the agglomeration" claim is
backwards. The agglomeration *homogenizes*; organic-ness has to be injected against it. The
spec should be honest that cross-unit size/child-count diversity is a tuned `size_variance`
effect, not a structural emergent property — and set a target for how much diversity is
wanted.

### K. The "stranded" step is dead when bridges are present and contiguity-breaking when absent — yet Component 4 depends on it

Two facts collide:

1. Component 4 step 2 builds the district graph with **mesh edges plus within-group
   bridges**. The landmask bridges *every* secondary island to the group's main island
   (`landmask.py:75-84`, star topology), so the within-group district graph is **always
   connected**. On a connected graph, `region_grow` never leaves an item unreached — so the
   Component 3 "stranded" step (step 5) **never fires**.
2. Yet Component 4's island-absorption story is explicitly "*the region-grower's stranded →
   nearest within-group neighbor step*." That step is the one that never runs.

So island absorption is actually done by **frontier growth across the bridge**, governed by
the balance rule — which can hand a small island to a *far* country, not its nearest. The
spec's stated mechanism and its actual mechanism are different (this sharpens first-review
finding D).

And the stranded step is not merely dead — it is *unsafe* in the one case it would run. If a
bridge were ever absent, "attach to the group owning its nearest item by **chord distance**"
attaches an item that touches nothing in that group across open water. The result is a
**geographically disconnected unit**, contradicting both the Component 3 headline
"Contiguous by construction — no repair pass needed" and the Testing section's "every unit
contiguous." This is exactly the multiplicative-Voronoi stranding that forced
`_repair_contiguity` to exist (`partition.py:97-102`). Resolve by removing the chord-attach
stranded step and relying on bridge connectivity — then state that connectivity as the
invariant it actually is.

### L. Per-group feasibility is never validated — only the global total is

Bottom-up makes the per-group nesting **rigid and exact**: `P_g = C_g · levels[1]`,
`D_g = P_g · levels[2]` (Component 1). Combined with "each group ≥ 1 country," the
*smallest* landmass is forced to host at least `levels[1] · levels[2]` districts. For the
default `levels=[6,5,6]` that is **30 districts × MIN_ATOMS_PER_LEAF (8) = 240 atoms**
minimum in the smallest group.

The only guard (`spec.py:71-79`) checks `max_leaf_count · MIN_ATOMS ≤ total land atoms` — a
**global** budget. It says nothing about whether the *smallest* group has 240 atoms. With
`spread` / `n_landmasses` producing lopsided groups, a small island group can be assigned a
country it cannot physically host, and Phase 1 would then be asked to cut, say, 120 atoms
into 30 districts — impossible under the ≥ 8-atom rule.

Today `honor_minimums` + `plan_islands` + clustering degrade gracefully. Bottom-up's exact
nesting removes that slack, so it **needs a new per-group validation** (or a documented
fallback that lets a small group carry fewer levels than the global spec). Neither is in the
doc.

---

## Secondary issues

### M. `border_roughness` as a per-level list becomes vestigial

Today `atom_cost_for(level)` scales the border field by `roughness[level]` — a *different*
cost field per level (`generate.py:96-101`), and `border_roughness` may be a **list**
(`spec.py:38`, `border_roughness_per_level`). In bottom-up the cost field shapes **only the
leaf level**, so only one entry of that list is ever consumed. The knob table says
`border_roughness` "shape[s] the leaf-district borders" but never resolves the list
semantics: if a user passes `[0.2, 0.5, 0.9]`, which value drives the leaf field, and are
the others silently ignored? Either collapse `border_roughness` to a scalar (and validate
against lists) or define exactly which entry wins. As written it is a silent footgun.

### N. FPS seeding pushes borders *toward* straightness — working against goal A

Component 3 step 1 seeds via farthest-point sampling on item centroids — i.e. the K most
*extreme* items. Seeds at the extremes grow inward and meet near the middle along
Voronoi-like bisectors (straight at macro scale), and corner seeds are the ones most prone
to the boxed-in balance failure (H). For a spec whose entire purpose is *less* straightness,
all-extreme seeding is a questionable default. Consider jittered or interior-biased seeds,
and treat this as a tunable that the prototype should have measured.

### O. `border_meander`'s propagation to province/country is indirect and unvalidated

The spec claims the cost field's meander is "more effective than before" at every level. But
`region_grow` is **blind to `atom_cost`** — it routes on shared-border length + rng, not on
elevation. Province/country borders follow crests *only* insofar as they are a subset of
district borders that already sit on crests, and the "strongest-link" heuristic may route
the boundary in ways that cut across the deepest crests rather than along them. This may be
fine, but "more effective, not less" is asserted, not shown. Flag it as an open question the
prototype must confirm, alongside the tortuosity number from first-review finding A.

### P. Testing section has no balance test and no feasibility test

The "New" tests cover totals, single-island leaves, absorption, and the (tautological, see
first review A) inheritance check — but **nothing asserts balance** (H, I) or **per-group
feasibility** (L). Given that balance is now a best-effort emergent, not a guaranteed
property, a test that provinces/countries land within X% of target (and a test that small
lopsided groups still generate) are the two most important additions and are both missing.

### Q. Minor

- **ISLET_MAX_ATOMS vs MIN_ATOMS_PER_LEAF conflation.** `partition.py:14-20` deliberately
  keeps these two numerically-equal-but-distinct constants separate ("not meant to be kept
  in lockstep"). Component 2's islet exception collapses them into `MIN_ATOMS_PER_LEAF`, and
  the code-changes section deletes `ISLET_MAX_ATOMS`. That may be fine, but the spec should
  acknowledge it is intentionally reversing a documented distinction.
- **Cross-group parent-index assembly.** `level_nodes[level]` is a flat global list and
  `parent` is an *index* into the previous level (`generate.py:151-162, 197-198`). Bottom-up
  builds per group then concatenates, so parent indices must be offset by each group's base
  position in the global list. Unspecified and off-by-one-prone; worth a sentence.

---

## Bottom line (second pass)

The first review's A–D still gate the plan. This pass adds two that are at least as serious:

- **H/I** — region-grow balance is best-effort and *worst* at the country level, with none
  of the rebalancing safety nets the current partitioner relies on.
- **L** — exact per-group nesting can be physically infeasible for a small landmass, and
  nothing validates it.

And a structural honesty fix:

- **J/K** — balancing suppresses the "organic distribution" Decision 2 claims to produce,
  and the stated island-absorption mechanism (the stranded step) is dead code that the doc
  nonetheless leans on.

Recommend a balance-bound + rebalance decision, a per-group feasibility rule, and rewrites
of Decision 2 and the Component 3/4 island story before this is "ready for implementation
plan."
