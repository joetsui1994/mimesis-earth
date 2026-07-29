# python/src/mimesis_earth/agglomerate.py
"""Bottom-up agglomeration: leaf districts -> provinces -> countries."""

from collections import defaultdict

import numpy as np

BRIDGE_EPS = 1e-6   # link weight for cross-water bridge edges
GROW_BIAS = 3.0     # region-grow field-bias strength (multiplies border_roughness)


def region_grow(neighbors, sizes, targets, seeds, rng, field=None, lam=0.0):
    """Grow K contiguous groups over an item graph, contiguous by construction.

    neighbors: dict item -> list[(neighbor, link_weight)].
    sizes:     array[float] item mass.
    targets:   array[float] length K, desired group mass.
    seeds:     list[int] length K, one starting item per group.
    field/lam: if given, prefer eating LOW-field frontier items (borders settle
               on high-field ridges). lam=0 -> plain strongest-link growth.

    Returns assign: array[int] length n, each in 0..K-1 (straggler guard fills
    any item the frontier never reached via an adjacent assigned group).
    """
    n = len(sizes)
    K = len(seeds)
    assign = np.full(n, -1)
    filled = np.zeros(K)
    frontier = [set() for _ in range(K)]
    link = [defaultdict(float) for _ in range(K)]
    for g, s in enumerate(seeds):
        assign[s] = g
        filled[g] = sizes[s]
        for nb, w in neighbors[s]:
            if assign[nb] == -1:
                frontier[g].add(nb)
                link[g][nb] += w
    remaining = n - K
    while remaining > 0:
        cand = [g for g in range(K) if frontier[g]]
        if not cand:
            break
        g = min(cand, key=lambda g: filled[g] / targets[g])
        items = sorted(frontier[g])  # canonical order -> determinism
        if field is not None:
            scores = [link[g][it] - lam * field[it] + 1e-9 * rng.random() for it in items]
        else:
            scores = [link[g][it] + 1e-9 * rng.random() for it in items]
        best = items[int(np.argmax(scores))]
        assign[best] = g
        filled[g] += sizes[best]
        remaining -= 1
        for gg in range(K):
            frontier[gg].discard(best)
        for nb, w in neighbors[best]:
            if assign[nb] == -1:
                frontier[g].add(nb)
                link[g][nb] += w
    _attach_stragglers(neighbors, assign)
    return assign


def _attach_stragglers(neighbors, assign):
    """Attach any unassigned item to its strongest-link assigned neighbor's group
    (never by chord distance -> preserves contiguity). Iterates so a straggler
    that only touches other stragglers is resolved once a neighbor is placed."""
    while True:
        stragglers = np.flatnonzero(assign == -1)
        if len(stragglers) == 0:
            return
        progressed = False
        for it in sorted(stragglers.tolist()):
            best_g, best_w = -1, -1.0
            for nb, w in neighbors[it]:
                if assign[nb] >= 0 and w > best_w:
                    best_g, best_w = int(assign[nb]), w
            if best_g >= 0:
                assign[it] = best_g
                progressed = True
        if not progressed:
            raise RuntimeError(
                "region_grow: items isolated from all seeds "
                f"({len(stragglers)} left) -- disconnected item graph"
            )
