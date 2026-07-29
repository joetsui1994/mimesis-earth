# python/src/mimesis_earth/agglomerate.py
"""Bottom-up agglomeration: leaf districts -> provinces -> countries."""

import math
from collections import defaultdict

import numpy as np
from scipy.sparse.csgraph import connected_components

from mimesis_earth.partition import allocate_counts, partition_atoms, redistribute_counts
from mimesis_earth.spec import MIN_ATOMS_PER_LEAF

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


def leaf_partition(mesh, group_atoms, n_districts, roughness, size_variance,
                   atom_cost, rng):
    """Partition one landmass group's atoms into n_districts single-island
    districts. Physical islands (mesh components, no bridges) are the units;
    the smallest are clustered onto their nearest until there are <= n_districts
    units, each >= MIN_ATOMS_PER_LEAF. Districts are then allocated across units
    (clamped so none exceeds unit_atoms // MIN_ATOMS_PER_LEAF) and each unit is
    cut with partition_atoms."""
    group_atoms = np.asarray(group_atoms)
    sub = mesh.adjacency[group_atoms][:, group_atoms]
    ncomp, comp = connected_components(sub, directed=False)
    units = [group_atoms[comp == c] for c in range(ncomp)]
    cents = [mesh.points[u].mean(0) for u in units]

    def merge_smallest():
        i = int(np.argmin([len(u) for u in units]))
        ci = cents[i]
        j = min((k for k in range(len(units)) if k != i),
                key=lambda k: float(np.linalg.norm(cents[k] - ci)))
        units[j] = np.concatenate([units[j], units[i]])
        cents[j] = mesh.points[units[j]].mean(0)
        del units[i]
        del cents[i]

    while len(units) > 1 and (
        len(units) > n_districts or min(len(u) for u in units) < MIN_ATOMS_PER_LEAF
    ):
        merge_smallest()

    unit_sizes = np.array([len(u) for u in units], dtype=float)
    caps = np.maximum(1, (unit_sizes // MIN_ATOMS_PER_LEAF).astype(int))
    alloc = redistribute_counts(allocate_counts(n_districts, unit_sizes), caps)
    districts = []
    for u, k in zip(units, alloc.tolist()):
        if k <= 1:
            districts.append(u)
        else:
            districts.extend(
                partition_atoms(mesh, u, k, None, roughness, rng,
                                size_variance=size_variance, atom_cost=atom_cost)
            )
    return districts


def allocate_group_counts(group_sizes, levels):
    """Countries per landmass group (proportional to size, each >= 1) and the
    derived leaf-district count per group (C_g * prod(levels[1:])). Raises if a
    group is too small to host D_g * MIN_ATOMS_PER_LEAF atoms (review L)."""
    group_sizes = np.asarray(group_sizes, dtype=float)
    C = redistribute_counts(
        allocate_counts(levels[0], group_sizes), group_sizes.astype(int)
    )
    leaves_per_country = math.prod(levels[1:]) if len(levels) > 1 else 1
    D = C * leaves_per_country
    need = D * MIN_ATOMS_PER_LEAF
    if (group_sizes < need).any():
        g = int(np.flatnonzero(group_sizes < need)[0])
        raise ValueError(
            f"landmass group {g} is too small: has {int(group_sizes[g])} atoms, "
            f"needs >= {int(need[g])} for {int(D[g])} districts at "
            f"MIN_ATOMS_PER_LEAF={MIN_ATOMS_PER_LEAF}. Lower n_landmasses, raise "
            f"resolution or land_fraction, or lower spread."
        )
    return C, D


def _item_field(grow_field, parts):
    """Mean grow_field over each part's atoms."""
    return np.array([float(grow_field[p].mean()) for p in parts])


def _grow_targets(total_mass, k, size_variance, rng):
    if size_variance <= 0:
        return np.full(k, total_mass / k)
    w = rng.lognormal(0.0, size_variance, size=k)
    return total_mass * w / w.sum()


def partition_world(mesh, mask, spec, atom_cost, grow_field, rng):
    """Bottom-up partition. Returns level_nodes: list per level of
    dicts {atoms, parent, landmass}. Leaves are districts; parents set by
    field-biased region-grow. See design spec Components 1-4."""
    levels = spec.levels
    n_levels = len(levels)
    roughness = float(spec.border_roughness)
    lam = GROW_BIAS * roughness
    group_sizes = np.array(
        [(mask.group == g).sum() for g in range(spec.n_landmasses)], dtype=float
    )
    C, D = allocate_group_counts(group_sizes, levels)

    level_nodes = [[] for _ in range(n_levels)]

    for g in range(spec.n_landmasses):
        group_atoms = np.flatnonzero(mask.group == g)
        # per-level counts for this group (index 0 = countries ... last = leaves)
        cnt = [int(C[g])]
        for lvl in range(1, n_levels):
            cnt.append(cnt[-1] * levels[lvl])

        # --- leaves (finest level) ---
        leaves = leaf_partition(mesh, group_atoms, cnt[-1], roughness,
                                spec.size_variance, atom_cost, rng)

        # --- agglomerate upward: parts[level] = list of atom arrays;
        #     parent_of[level][i] = index into parts[level-1] within this group.
        parts = [None] * n_levels
        parent_of = [None] * n_levels
        parts[n_levels - 1] = leaves
        for lvl in range(n_levels - 2, -1, -1):
            child_parts = parts[lvl + 1]
            neighbors, sizes = build_item_graph(mesh, child_parts, bridges=mask.bridges)
            field = _item_field(grow_field, child_parts)
            k = cnt[lvl]
            cent = np.array([mesh.points[p].mean(0) for p in child_parts])
            cent /= np.linalg.norm(cent, axis=1, keepdims=True)
            seeds = _fps(cent, k)
            targets = _grow_targets(sizes.sum(), k, spec.size_variance, rng)
            assign = region_grow(neighbors, sizes, targets, seeds, rng,
                                 field=field, lam=lam)
            parts[lvl] = [
                np.concatenate([child_parts[i] for i in np.flatnonzero(assign == c)])
                for c in range(k)
            ]
            parent_of[lvl + 1] = assign  # child level -> its parent index (this level)

        # --- append to global level_nodes with per-group parent offsets ---
        base = [len(level_nodes[lvl]) for lvl in range(n_levels)]
        for lvl in range(n_levels):
            for i, atoms in enumerate(parts[lvl]):
                node = {"atoms": atoms, "landmass": g if lvl == 0 else None}
                if lvl == 0:
                    node["parent"] = None
                else:
                    node["parent"] = base[lvl - 1] + int(parent_of[lvl][i])
                level_nodes[lvl].append(node)

    return level_nodes


def _fps(points, k):
    """Farthest-point sampling: k well-spread indices into points."""
    chosen = [0]
    d = np.linalg.norm(points - points[0], axis=1)
    while len(chosen) < k:
        nxt = int(d.argmax())
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return chosen
