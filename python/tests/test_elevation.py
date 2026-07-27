import numpy as np

from mimesis_earth.elevation import build_elevation, ridged_noise
from mimesis_earth.mesh import build_mesh
from mimesis_earth.noise import unit_vectors
from mimesis_earth.spec import WorldSpec


def test_ridged_noise_deterministic_and_crisp():
    pts = unit_vectors(4000, np.random.default_rng(1))
    a = ridged_noise(pts, np.random.default_rng(2))
    b = ridged_noise(pts, np.random.default_rng(2))
    np.testing.assert_array_equal(a, b)
    assert abs(a.mean()) < 1e-9 and abs(a.std() - 1.0) < 1e-9
    # ridged fields are asymmetric: sharp crests, broad valleys
    assert abs(np.median(a) - a.mean()) > 0.05


def test_build_elevation_shape_and_determinism():
    mesh = build_mesh(4000, np.random.default_rng(3))
    seeds = unit_vectors(3, np.random.default_rng(4))
    spec = WorldSpec(levels=[3, 3], n_landmasses=3, resolution=4000)
    e1 = build_elevation(mesh, seeds, spec, np.random.default_rng(5))
    e2 = build_elevation(mesh, seeds, spec, np.random.default_rng(5))
    np.testing.assert_array_equal(e1, e2)
    assert e1.shape == (4000,)
    assert np.isfinite(e1).all()


def test_elevation_peaks_near_seeds():
    mesh = build_mesh(4000, np.random.default_rng(6))
    seeds = unit_vectors(2, np.random.default_rng(7))
    spec = WorldSpec(levels=[2, 3], n_landmasses=2, coast_ruggedness=0.3,
                     resolution=4000)
    elev = build_elevation(mesh, seeds, spec, np.random.default_rng(8))
    angle = np.arccos(np.clip(mesh.points @ seeds.T, -1, 1)).min(axis=1)
    near = elev[angle < 0.5].mean()
    far = elev[angle > 1.5].mean()
    assert near > far + 0.5


def test_ruggedness_scales_relief():
    mesh = build_mesh(4000, np.random.default_rng(9))
    seeds = unit_vectors(3, np.random.default_rng(10))
    smooth = build_elevation(
        mesh, seeds,
        WorldSpec(levels=[3, 3], n_landmasses=3, coast_ruggedness=0.0,
                  resolution=4000),
        np.random.default_rng(11),
    )
    rough = build_elevation(
        mesh, seeds,
        WorldSpec(levels=[3, 3], n_landmasses=3, coast_ruggedness=1.0,
                  resolution=4000),
        np.random.default_rng(11),
    )
    # relief = residual variance after removing the continent base trend;
    # proxy: local roughness via edge-difference std
    # local roughness across true mesh neighbors (array-index neighbors are
    # ~137 degrees apart on a Fibonacci sphere and measure nothing local)
    def edge_std(e):
        return np.abs(e[mesh.edges[:, 0]] - e[mesh.edges[:, 1]]).std()

    assert edge_std(rough) > 4.0 * edge_std(smooth)
