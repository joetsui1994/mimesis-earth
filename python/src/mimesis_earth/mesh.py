"""The atom mesh: jittered Fibonacci points and their spherical Voronoi cells."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull, SphericalVoronoi


@dataclass
class Mesh:
    points: np.ndarray  # (n, 3) atom centers on the unit sphere
    vertices: np.ndarray  # (m, 3) Voronoi vertices on the unit sphere
    regions: list  # per atom: ordered Voronoi vertex indices (closed ring)
    edges: np.ndarray  # (e, 2) unique undirected atom adjacencies, col0 < col1
    adjacency: csr_matrix  # symmetric (n, n), weights = geodesic edge length
    areas: np.ndarray  # (n,) spherical cell areas on the unit sphere


def fibonacci_points(n: int, rng: np.random.Generator, jitter: float = 0.35) -> np.ndarray:
    i = np.arange(n)
    golden = (1 + 5**0.5) / 2
    theta = 2 * np.pi * i / golden
    z = 1 - (2 * i + 1) / n
    r = np.sqrt(np.clip(1 - z**2, 0, None))
    pts = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    # tangential jitter scaled to typical point spacing, then renormalize
    spacing = np.sqrt(4 * np.pi / n)
    noise = rng.normal(scale=jitter * spacing, size=(n, 3))
    noise -= pts * np.sum(noise * pts, axis=1, keepdims=True)
    pts = pts + noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def build_mesh(n: int, rng: np.random.Generator) -> Mesh:
    pts = fibonacci_points(n, rng)
    sv = SphericalVoronoi(pts, radius=1.0)
    sv.sort_vertices_of_regions()
    hull = ConvexHull(pts)  # Delaunay triangulation on the sphere
    tri = hull.simplices
    e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    w = np.arccos(np.clip(np.sum(pts[e[:, 0]] * pts[e[:, 1]], axis=1), -1.0, 1.0))
    adjacency = csr_matrix(
        (
            np.concatenate([w, w]),
            (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])),
        ),
        shape=(n, n),
    )
    return Mesh(
        points=pts,
        vertices=sv.vertices,
        regions=sv.regions,
        edges=e,
        adjacency=adjacency,
        areas=sv.calculate_areas(),
    )
