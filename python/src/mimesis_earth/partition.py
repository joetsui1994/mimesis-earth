"""Competitive flood-fill partitioning of atoms over the adjacency graph."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

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
    # symmetric per-edge noise makes borders wiggly; same draw for both directions
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
    seeds = pick_seeds(mesh.points[atom_idx], k, rng)
    dist = dijkstra(adj, directed=False, indices=seeds)
    labels = np.asarray(dist).argmin(axis=0)
    # atoms unreachable from every seed (disconnected slivers with no bridge):
    # attach to the nearest seed by straight-line distance
    unreachable = ~np.isfinite(np.asarray(dist).min(axis=0))
    if unreachable.any():
        pts = mesh.points[atom_idx]
        chord = np.linalg.norm(
            pts[unreachable][:, None, :] - pts[seeds][None, :, :], axis=2
        )
        labels[unreachable] = chord.argmin(axis=1)
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
