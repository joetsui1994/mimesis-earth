"""Competitive flood-fill partitioning of atoms over the adjacency graph."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from mimesis_earth.mesh import Mesh

BRIDGE_COST_FACTOR = 3.0


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


def _assign_labels(adj, seeds, pts):
    dist = np.asarray(dijkstra(adj, directed=False, indices=seeds))
    labels = dist.argmin(axis=0)
    unreachable = ~np.isfinite(dist.min(axis=0))
    if unreachable.any():
        chord = np.linalg.norm(
            pts[unreachable][:, None, :] - pts[seeds][None, :, :], axis=2
        )
        labels[unreachable] = chord.argmin(axis=1)
    return labels


def partition_atoms(
    mesh: Mesh,
    atom_idx: np.ndarray,
    k: int,
    extra_edges: Optional[np.ndarray],
    roughness: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Split atom_idx into k non-empty contiguous parts. Returns global index arrays."""
    atom_idx = np.asarray(atom_idx)
    assert 1 <= k <= len(atom_idx), f"cannot cut {len(atom_idx)} atoms into {k} parts"
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

    pts = mesh.points[atom_idx]
    labels = _assign_labels(adj, seeds, pts)
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
        mean_size = part_sizes.mean()
        for i in range(k):
            if part_sizes[i] < mean_size / 8.0:
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
        labels = _assign_labels(adj, seeds, pts)
    return [atom_idx[labels == i] for i in range(k)]


def child_counts(
    mean: int, n_parents: int, variance: float, rng: np.random.Generator
) -> np.ndarray:
    """How many children each parent gets. variance=0 -> exactly `mean` each."""
    if variance <= 0:
        return np.full(n_parents, mean, dtype=int)
    counts = np.round(rng.normal(mean, variance * mean, n_parents)).astype(int)
    return np.clip(counts, 1, None)


def allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    """Split `total` units among groups proportionally to weights, each >= 1."""
    weights = np.asarray(weights, dtype=float)
    assert (weights >= 0).all() and weights.sum() > 0, "weights must be non-negative with positive sum"
    assert total >= len(weights)
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
    assert counts.sum() <= capacities.sum(), "not enough total capacity"
    excess = int(np.clip(counts - capacities, 0, None).sum())
    counts = np.minimum(counts, capacities)
    while excess > 0:
        headroom = capacities - counts
        counts[int(headroom.argmax())] += 1
        excess -= 1
    return counts
