"""Competitive flood-fill partitioning of atoms over the adjacency graph."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from mimesis_earth.mesh import Mesh

BRIDGE_COST_FACTOR = 3.0

# Islands at or above this atom count get their own child quota in
# plan_islands; smaller fragments ("islets") attach to the nearest sizeable
# island instead of starving the quota. Numerically the same as
# spec.MIN_ATOMS_PER_LEAF, but the two constants mean different things (that
# one bounds land-fraction sizing so leaves stay resolvable; this one is the
# island-attachment threshold) and are not meant to be kept in lockstep.
ISLET_MAX_ATOMS = 8


def pick_seeds(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Farthest-point sampling: k well-spaced local indices into `points`."""
    first = int(rng.integers(len(points)))
    chosen = [first]
    d = np.linalg.norm(points - points[first], axis=1)
    while len(chosen) < k:
        nxt = int(d.argmax())
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen)


def _subgraph(
    mesh: Mesh,
    atom_idx: np.ndarray,
    extra_edges: Optional[np.ndarray],
    roughness: float,
    rng: np.random.Generator,
) -> csr_matrix:
    pos = -np.ones(len(mesh.points), dtype=int)
    pos[atom_idx] = np.arange(len(atom_idx))
    e = mesh.edges
    m = (pos[e[:, 0]] >= 0) & (pos[e[:, 1]] >= 0)
    local = np.column_stack([pos[e[m, 0]], pos[e[m, 1]]])
    w = np.arccos(
        np.clip(np.sum(mesh.points[e[m, 0]] * mesh.points[e[m, 1]], axis=1), -1, 1)
    )
    if extra_edges is not None and len(extra_edges) > 0:
        bm = (pos[extra_edges[:, 0]] >= 0) & (pos[extra_edges[:, 1]] >= 0)
        be = extra_edges[bm]
        if len(be) > 0:
            bw = BRIDGE_COST_FACTOR * np.arccos(
                np.clip(
                    np.sum(mesh.points[be[:, 0]] * mesh.points[be[:, 1]], axis=1),
                    -1,
                    1,
                )
            )
            local = np.vstack([local, np.column_stack([pos[be[:, 0]], pos[be[:, 1]]])])
            w = np.concatenate([w, bw])
    # symmetric per-edge noise makes borders wiggly; same draw for both directions.
    # roughness=0 draws nothing, so callers that need a plain (unperturbed)
    # subgraph -- e.g. component detection for seeding -- can pass the
    # caller's own rng here without advancing its stream.
    if roughness > 0:
        w = w * (1.0 + roughness * rng.uniform(0.0, 3.0, size=len(w)))
    n = len(atom_idx)
    return csr_matrix(
        (
            np.concatenate([w, w]),
            (
                np.concatenate([local[:, 0], local[:, 1]]),
                np.concatenate([local[:, 1], local[:, 0]]),
            ),
        ),
        shape=(n, n),
    )


def _assign_labels(adj, seeds, pts, weights):
    dist = np.asarray(dijkstra(adj, directed=False, indices=seeds))
    labels = (dist / weights[:, None]).argmin(axis=0)
    reachable = np.isfinite(dist.min(axis=0))
    if (~reachable).any():
        chord = np.linalg.norm(
            pts[~reachable][:, None, :] - pts[seeds][None, :, :], axis=2
        )
        labels[~reachable] = (chord / weights[None, :]).argmin(axis=1)
    return labels, reachable


def _repair_contiguity(adj, seeds, labels, reachable):
    """Weighted assignment can strand fragments of a part away from its seed
    (multiplicative Voronoi regions need not be connected). Reattach every
    stranded fragment to the adjacent part with the largest shared boundary.
    Genuinely unreachable atoms (disconnected inputs) keep their chord-based
    assignment untouched."""
    k = len(seeds)
    coo = adj.tocoo()
    while True:
        frag_ids = np.full(len(labels), -1)
        frags: list[np.ndarray] = []
        for i in range(k):
            members = np.flatnonzero((labels == i) & reachable)
            if len(members) < 2:
                continue
            sub = adj[members][:, members]
            n_comp, comp = connected_components(sub, directed=False)
            if n_comp <= 1:
                continue
            seed_pos = np.flatnonzero(members == seeds[i])
            seed_comp = (
                int(comp[seed_pos[0]])
                if len(seed_pos)
                else int(np.bincount(comp).argmax())
            )
            for c in range(n_comp):
                if c == seed_comp:
                    continue
                frag = members[comp == c]
                frag_ids[frag] = len(frags)
                frags.append(frag)
        if not frags:
            return labels
        moved = False
        boundary = (frag_ids[coo.row] >= 0) & (frag_ids[coo.col] < 0)
        votes = np.zeros((len(frags), k))
        np.add.at(
            votes, (frag_ids[coo.row][boundary], labels[coo.col][boundary]), 1.0
        )
        for fid, frag in enumerate(frags):
            if votes[fid].sum() > 0:
                labels[frag] = int(votes[fid].argmax())
                moved = True
        if not moved:
            return labels


def partition_atoms(
    mesh: Mesh,
    atom_idx: np.ndarray,
    k: int,
    extra_edges: Optional[np.ndarray],
    roughness: float,
    rng: np.random.Generator,
    size_variance: float = 0.0,
) -> list[np.ndarray]:
    """Split atom_idx into k non-empty contiguous parts. Returns global index arrays."""
    atom_idx = np.asarray(atom_idx)
    if not 1 <= k <= len(atom_idx):
        raise ValueError(f"cannot cut {len(atom_idx)} atoms into {k} parts")
    if k == 1:
        return [atom_idx]
    adj = _subgraph(mesh, atom_idx, extra_edges, roughness, rng)

    # seed only on substantial islands: an FPS seed trapped on a tiny islet
    # produces a starved part that Lloyd cannot rescue. roughness=0 means
    # _subgraph draws no noise here, so this reuses the caller's rng
    # without perturbing its stream (see the roughness>0 guard above).
    mesh_only = _subgraph(mesh, atom_idx, None, 0.0, rng)
    n_comp, comp = connected_components(mesh_only, directed=False)
    sizes = np.bincount(comp)
    substantial = sizes[comp] >= max(2, len(atom_idx) // (4 * k))
    candidates = np.flatnonzero(substantial)
    if len(candidates) < k:
        candidates = np.arange(len(atom_idx))
    seeds = candidates[pick_seeds(mesh.points[atom_idx[candidates]], k, rng)]

    weights = (
        rng.lognormal(0.0, size_variance, size=k)
        if size_variance > 0
        else np.ones(k)
    )

    pts = mesh.points[atom_idx]
    labels, reachable = _assign_labels(adj, seeds, pts, weights)
    if size_variance > 0:
        labels = _repair_contiguity(adj, seeds, labels, reachable)
    # Lloyd-style rebalancing: move each seed to its part's medoid and
    # reassign; evens out part sizes so deep hierarchy levels don't starve
    for _ in range(3):
        new_seeds = []
        for i in range(k):
            members = np.flatnonzero(labels == i)
            center = pts[members].mean(axis=0)
            norm = np.linalg.norm(center)
            if norm > 1e-12:
                center = center / norm
            j = members[int(np.argmin(np.linalg.norm(pts[members] - center, axis=1)))]
            new_seeds.append(int(j))
        # starved-part escape: a part far below average size (e.g. an FPS
        # seed stuck on a tiny islet) can't be rescued by its own medoid;
        # relocate its seed to the farthest atom of the largest part instead
        part_sizes = np.array([int((labels == i).sum()) for i in range(k)])
        expected = len(atom_idx) * weights / weights.sum()
        for i in range(k):
            if part_sizes[i] < max(2.0, expected[i] / 8.0):
                big = int(part_sizes.argmax())
                members = np.flatnonzero(labels == big)
                far = members[
                    int(np.argmax(np.linalg.norm(pts[members] - pts[new_seeds[big]], axis=1)))
                ]
                new_seeds[i] = int(far)
        # guard against two starved parts picking the identical atom: keep
        # the earlier occurrence's seed, revert any later duplicate to its
        # previous seed
        seen = set()
        for i, s in enumerate(new_seeds):
            if s in seen:
                new_seeds[i] = int(seeds[i])
            else:
                seen.add(s)
        if np.array_equal(np.array(new_seeds), seeds):
            break
        seeds = np.array(new_seeds)
        labels, reachable = _assign_labels(adj, seeds, pts, weights)
        if size_variance > 0:
            labels = _repair_contiguity(adj, seeds, labels, reachable)
    return [atom_idx[labels == i] for i in range(k)]


def allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    """Split `total` units among groups proportionally to weights, each >= 1."""
    weights = np.asarray(weights, dtype=float)
    if not ((weights >= 0).all() and weights.sum() > 0):
        raise ValueError("weights must be non-negative with positive sum")
    if not total >= len(weights):
        raise ValueError("total must be >= number of groups")
    share = weights / weights.sum()
    counts = np.maximum(1, np.floor(share * total)).astype(int)
    while counts.sum() > total:
        counts[counts.argmax()] -= 1
    remainder = share * total - counts
    while counts.sum() < total:
        i = int(remainder.argmax())
        counts[i] += 1
        remainder[i] -= 1.0
    return counts


def redistribute_counts(counts: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    """Clip per-parent child counts to capacities, handing the excess to
    parents with the most headroom. Preserves the total exactly."""
    counts = np.asarray(counts, dtype=int).copy()
    capacities = np.asarray(capacities, dtype=int)
    if not counts.sum() <= capacities.sum():
        raise ValueError("not enough total capacity")
    excess = int(np.clip(counts - capacities, 0, None).sum())
    counts = np.minimum(counts, capacities)
    while excess > 0:
        headroom = capacities - counts
        counts[int(headroom.argmax())] += 1
        excess -= 1
    return counts


def coupled_counts(
    total: int,
    sizes: np.ndarray,
    coupling: float,
    variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Split `total` children among parents. Weights follow each parent's
    territory share raised to `coupling` (0 = uniform, 1 = proportional),
    jittered log-normally by `variance`. Total exact, each parent >= 1."""
    weights = np.asarray(sizes, dtype=float) ** coupling
    if variance > 0:
        weights = weights * rng.lognormal(0.0, variance, size=len(weights))
    return allocate_counts(total, weights)


def honor_minimums(counts: np.ndarray, minimums: np.ndarray) -> np.ndarray:
    """Best-effort: raise counts toward per-parent minimums by borrowing from
    parents above their own minimum. Preserves the total exactly; never takes
    a donor below max(1, its minimum). Shortfalls remain when donors run out
    (callers degrade gracefully via island clustering)."""
    counts = np.asarray(counts, dtype=int).copy()
    minimums = np.asarray(minimums, dtype=int)
    while True:
        need = minimums - counts
        needy = np.flatnonzero(need > 0)
        surplus = counts - np.maximum(minimums, 1)
        donors = np.flatnonzero(surplus > 0)
        if len(needy) == 0 or len(donors) == 0:
            return counts
        counts[needy[int(np.argmax(need[needy]))]] += 1
        counts[donors[int(np.argmax(surplus[donors]))]] -= 1


def _island_analysis(mesh: Mesh, atom_idx: np.ndarray, rng: np.random.Generator):
    """Connected components of atom_idx over mesh edges only (no bridges).
    Returns (n_comp, comp_labels, comp_sizes, sizeable_component_ids)."""
    sub = _subgraph(mesh, atom_idx, None, 0.0, rng)  # roughness 0: no rng draws
    n_comp, comp = connected_components(sub, directed=False)
    sizes = np.bincount(comp)
    sizeable = np.flatnonzero(sizes >= ISLET_MAX_ATOMS)
    if len(sizeable) == 0:
        sizeable = np.array([int(sizes.argmax())])
    return n_comp, comp, sizes, sizeable


def count_sizeable_islands(
    mesh: Mesh, atom_idx: np.ndarray, rng: np.random.Generator
) -> int:
    _, _, _, sizeable = _island_analysis(mesh, np.asarray(atom_idx), rng)
    return len(sizeable)


def plan_islands(
    mesh: Mesh,
    atom_idx: np.ndarray,
    k: int,
    coupling: float,
    rng: np.random.Generator,
    analysis: Optional[tuple] = None,
) -> list[tuple[np.ndarray, int]]:
    """Split a parent's atoms into island groups with per-group child counts
    summing to k. Sizeable islands each host >= 1 child when quota allows;
    islets attach to whichever sizeable island holds their single closest
    atom (chord distance via cKDTree -- centroid distance can misjudge
    elongated or crescent islands whose centroid sits far from any
    coastline). With more sizeable islands than quota, the smaller islands
    are clustered onto the k largest by the same closest-atom rule (each
    cluster = 1 child). `analysis`, if given, must be a precomputed
    `_island_analysis(mesh, atom_idx, rng)` result -- callers that already
    ran the analysis for quota minimums can pass it through instead of
    recomputing it here."""
    atom_idx = np.asarray(atom_idx)
    if analysis is None:
        analysis = _island_analysis(mesh, atom_idx, rng)
    n_comp, comp, sizes, sizeable = analysis
    if n_comp == 1:
        return [(atom_idx, k)]
    pts = mesh.points[atom_idx]
    sizeable_set = set(sizeable.tolist())
    owner = np.array([c if c in sizeable_set else -1 for c in range(n_comp)])

    # attach every islet (non-sizeable component) to the sizeable island
    # holding its closest atom, by chord distance
    islet_atoms = ~np.isin(comp, sizeable)
    if islet_atoms.any():
        sizeable_atoms = ~islet_atoms
        tree = cKDTree(pts[sizeable_atoms])
        sizeable_atom_comp = comp[sizeable_atoms]
        dist, nn = tree.query(pts[islet_atoms])
        nearest_island = sizeable_atom_comp[nn]
        islet_comp = comp[islet_atoms]
        # resolve in ascending-distance order so each islet component picks
        # up the island reached by its single nearest atom
        resolved: dict[int, int] = {}
        for pos in np.argsort(dist):
            c = int(islet_comp[pos])
            if c not in resolved:
                resolved[c] = int(nearest_island[pos])
        for c, s in resolved.items():
            owner[c] = s

    m = len(sizeable)
    if m <= k:
        group_sizes = np.array(
            [float(sizes[owner == s].sum()) for s in sizeable]
        )
        alloc = allocate_counts(k, group_sizes**coupling)
        # uniform coupling (coupling=0) can hand a 1-atom island the same
        # quota as a continent; cap allocation at a soft per-island capacity
        # first, falling back to the hard atom-count capacity, so tiny
        # islands don't spawn spammy 1-atom children when a softer split
        # across the other islands is feasible.
        soft = np.maximum(1, (group_sizes // ISLET_MAX_ATOMS).astype(int))
        if soft.sum() >= k:
            alloc = redistribute_counts(alloc, soft)
        alloc = redistribute_counts(alloc, group_sizes.astype(int))
        groups = list(zip(sizeable.tolist(), alloc.tolist()))
    else:
        order = sizeable[np.argsort(-sizes[sizeable])]
        cluster_seeds = order[:k]
        cluster_of = {int(s): int(s) for s in cluster_seeds}
        seed_atoms = np.isin(comp, cluster_seeds)
        seed_tree = cKDTree(pts[seed_atoms])
        seed_atom_comp = comp[seed_atoms]
        for c in order[k:]:
            d, nn = seed_tree.query(pts[comp == c])
            nearest = int(np.argmin(d))
            cluster_of[int(c)] = int(seed_atom_comp[nn[nearest]])
        owner = np.array([cluster_of.get(int(o), int(o)) for o in owner])
        groups = [(int(s), 1) for s in cluster_seeds]
    out = []
    for s, count in groups:
        members = atom_idx[owner[comp] == s]
        out.append((members, int(count)))
    return out
