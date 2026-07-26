import numpy as np
from scipy.sparse.csgraph import connected_components

from mimesis_earth.mesh import Mesh, build_mesh, fibonacci_points


def test_fibonacci_points_on_unit_sphere():
    rng = np.random.default_rng(1)
    pts = fibonacci_points(500, rng)
    assert pts.shape == (500, 3)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-12)


def test_fibonacci_points_deterministic():
    a = fibonacci_points(200, np.random.default_rng(7))
    b = fibonacci_points(200, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_build_mesh_structure():
    mesh = build_mesh(500, np.random.default_rng(3))
    assert isinstance(mesh, Mesh)
    assert mesh.points.shape == (500, 3)
    assert len(mesh.regions) == 500
    # every region references valid voronoi vertices
    for region in mesh.regions:
        assert len(region) >= 3
        assert max(region) < len(mesh.vertices)
    # areas tile the sphere
    np.testing.assert_allclose(mesh.areas.sum(), 4 * np.pi, rtol=1e-6)
    assert (mesh.areas > 0).all()
    # adjacency is symmetric, no self-loops, connected-ish degree
    assert (mesh.adjacency != mesh.adjacency.T).nnz == 0
    assert mesh.adjacency.diagonal().sum() == 0
    degrees = np.diff(mesh.adjacency.indptr)
    assert degrees.min() >= 3
    n_comp, _ = connected_components(mesh.adjacency, directed=False)
    assert n_comp == 1


def test_edges_are_unique_and_undirected():
    mesh = build_mesh(300, np.random.default_rng(5))
    e = mesh.edges
    assert (e[:, 0] < e[:, 1]).all()
    assert len(np.unique(e, axis=0)) == len(e)


def test_adjacency_weights_are_geodesic():
    mesh = build_mesh(300, np.random.default_rng(9))
    for a, b in mesh.edges[:50]:
        expected = np.arccos(np.clip(mesh.points[a] @ mesh.points[b], -1, 1))
        assert abs(mesh.adjacency[a, b] - expected) < 1e-12


def test_build_mesh_deterministic():
    m1 = build_mesh(300, np.random.default_rng(11))
    m2 = build_mesh(300, np.random.default_rng(11))
    np.testing.assert_array_equal(m1.edges, m2.edges)
    np.testing.assert_array_equal(m1.areas, m2.areas)
    assert (m1.adjacency != m2.adjacency).nnz == 0
    for r1, r2 in zip(m1.regions, m2.regions):
        assert r1 == r2
