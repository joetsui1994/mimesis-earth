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
    # Two-phase seeding: assign ALL seeds first, then build frontiers. This
    # prevents a seed that is adjacent to another seed from being added to the
    # earlier group's frontier and later "stolen" (reassigned).
    for g, s in enumerate(seeds):
        assign[s] = g
        filled[g] = sizes[s]
    for g, s in enumerate(seeds):
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


def build_item_graph(mesh, parts, bridges=None):
    """Adjacency over 'parts' (lists of atom indices). Edge weight = summed
    shared-border arc length. Bridge atom-pairs add BRIDGE_EPS links so
    across-water neighbors are reachable but eaten last."""
    lab = np.full(len(mesh.points), -1)
    for i, p in enumerate(parts):
        lab[p] = i
    e = mesh.edges
    a_all, b_all = lab[e[:, 0]], lab[e[:, 1]]
    m = (a_all >= 0) & (b_all >= 0) & (a_all != b_all)
    a, b = a_all[m], b_all[m]
    w = np.arccos(
        np.clip(np.sum(mesh.points[e[m, 0]] * mesh.points[e[m, 1]], axis=1), -1, 1)
    )
    nbr = defaultdict(lambda: defaultdict(float))
    for i, j, ww in zip(a.tolist(), b.tolist(), w.tolist()):
        nbr[i][j] += ww
        nbr[j][i] += ww
    if bridges is not None and len(bridges):
        ba, bb = lab[bridges[:, 0]], lab[bridges[:, 1]]
        bm = (ba >= 0) & (bb >= 0) & (ba != bb)
        for i, j in zip(ba[bm].tolist(), bb[bm].tolist()):
            nbr[i][j] += BRIDGE_EPS
            nbr[j][i] += BRIDGE_EPS
    neighbors = {i: [(j, ww) for j, ww in nbr[i].items()] for i in range(len(parts))}
    sizes = np.array([len(p) for p in parts], dtype=float)
    return neighbors, sizes
