# The Case of the Vanishing Landmasses

*A plain-language account of two bugs found and fixed in the land-mask step of
the world generator (commits `f8f7434` and `90e1c9a`).*

## Background: what the land mask does

Every generated world starts as a sphere covered in roughly 20,000 tiny tiles
(we call them *atoms*). The land mask's job is to answer one question for each
atom: **is this land or sea?** — and, if it's land, **which landmass does it
belong to?**

The user asks for a specific number of landmasses (say, 4 islands). The
algorithm works like this:

1. **Drop seeds.** Pick 4 points on the globe — one per requested landmass.
   The user's *spread* setting controls how these seeds are placed: spread = 0
   means they huddle together in one corner of the globe, spread = 1 means
   they scatter anywhere.
2. **Score every atom.** Each atom gets a score that is basically "how close
   am I to the nearest seed?" — atoms near a seed score high, atoms in the
   middle of the future ocean score low. Then we add a layer of random,
   smoothly varying noise on top. The noise is what makes coastlines ragged
   and organic instead of perfectly round.
3. **Draw the sea level.** If the user asked for 30% land, we keep the
   top-scoring 30% of atoms as land and declare everything else sea.
4. **Group the land.** Each land atom joins the landmass of whichever seed
   it's closest to.

Simple, fast, and it produces nice-looking worlds. It also had two hidden
flaws, both variations of the same theme: **nothing in the design actually
guaranteed that all the requested landmasses would survive the process.**

## Bug #1: cramming too many seeds into a small corner

The first flaw was in step 1. The "how tightly do the seeds huddle" dial was
computed from the spread setting alone — it completely ignored *how many*
seeds there were.

Think of it like throwing darts at a dartboard: with spread = 0, the algorithm
aims all darts at a circle the size of a coin. That's fine when you're
throwing 2 or 3 darts. But ask for 32 landmasses at low spread, and you're
now trying to fit 32 darts into that same coin-sized circle. The seeds end up
so close together that they can't each claim their own patch of high-scoring
atoms — several seeds effectively sit on top of each other, and when the sea
level is drawn, some of them own no land at all. The world comes out with
fewer landmasses than requested, which the code (correctly) treats as a
failure.

There was a retry loop for failures — but it just re-threw the same 32 darts
at the same coin-sized circle, so it failed the same way every time. In
testing, the combination "32 landmasses, spread 0" failed **200 times out of
200 attempts**. This wasn't bad luck; it was geometry.

**The fix** has two parts:

- The huddling dial now accounts for the number of landmasses: more seeds →
  a proportionally bigger target circle, so each seed has room for a
  territory of its own. (Technically: the concentration parameter is divided
  by the number of landmasses, because the area a cluster of K patches needs
  grows with K.)
- Each retry now *loosens* the huddling a bit further instead of repeating
  the identical throw. Retries actually change the odds now.

After the fix, the same "32 landmasses, spread 0" case succeeded **200 out of
200 times** — and on the first attempt, no retries needed.

## Bug #2: the top-30% cut doesn't care about your landmasses

The second flaw was subtler, and it survived the first fix. It lives in
step 3 — the "keep the top 30% of scores" cut.

That cut is *global*: it looks at all 20,000 atom scores in one big pile and
keeps the best ones, with no regard for which seed they belong to. Remember
that random noise we add to make coastlines interesting? That noise varies
smoothly across the globe — it has broad "high-pressure" and "low-pressure"
regions, like a weather map. If one of those low-pressure regions happens to
sit on top of a seed, every atom near that seed gets dragged down the
rankings, and the entire landmass can fall below the sea level line and
drown.

With few landmasses this almost never happens — each landmass is big, and
big things are hard to drown. But ask for many landmasses *and* only a
little land (say, 63 landmasses covering 10% of the globe), and each
landmass is now a small target. Testing showed this combination failed
essentially **100% of the time — at any spread setting, and at any
resolution**, because the problem isn't where the seeds are or how many
atoms exist; it's that a global ranking is simply allowed to ignore some
seeds entirely.

**The fix** changes the guarantee from "hope for the best" to "reserved
seating." Before the global ranking runs, every seed is now handed a small
guaranteed kernel of land — its few nearest atoms are marked land
unconditionally. Only the *remaining* land budget is then distributed by the
usual score ranking. The kernels are deliberately small (at most a quarter of
the total land budget, shared among all seeds), so the noise still shapes
almost all of the coastline — but no landmass can ever be erased, because
its core territory is not up for debate.

A pleasant side effect: the amount of land is now *exactly* the requested
fraction (we count out the land budget atom by atom), rather than
approximately.

## Why this mattered

The whole point of this generator is that it's a controllable instrument:
you set the dials — number of landmasses, spread, land fraction — and you
get a world with exactly those properties, reproducibly, from a seed. A
several-percent chance of crashing (or silently producing 61 landmasses when
you asked for 63) breaks that contract, especially for research use where
someone might sweep hundreds of parameter combinations in a batch.

## How we know it's fixed

The reviewer re-ran the failure cases and a broad sweep of the whole valid
parameter space after the fix:

- Both previously failing scenarios: now succeed 100% of the time.
- A sweep of **13,680 combinations** across landmass counts (1–64), spreads
  (0–1), land fractions (0.02–0.79), coastline ruggedness (0–1), and
  resolutions (4,000–200,000): **zero failures**.
- Determinism verified: the same seed still produces byte-identical worlds,
  including when the algorithm needs a retry internally.
- Coastline character verified: the ruggedness dial still works (smooth
  single blobs at 0, fractured archipelagos at 1) — the reserved kernels
  didn't flatten the noise's effect.

Both bugs are also pinned by regression tests (`test_many_landmasses_low_spread`
and `test_many_landmasses_low_land_fraction` in `python/tests/test_landmask.py`),
so they can't quietly come back.

## The general lesson

Both bugs share a shape: a *statistical* process (random seeds, random noise,
a global ranking) was being trusted to deliver a *hard* guarantee ("you get
exactly N landmasses"). Statistics doesn't make promises — for the cases
where the odds got bad, the generator crashed. The fixes work because they
convert the guarantee into a *construction*: the seed cluster is sized so
every seed can own territory, and every landmass holds a reserved kernel no
ranking can take away. Randomness still paints the details; it just no longer
gets a vote on whether a landmass exists.
