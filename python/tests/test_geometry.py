import numpy as np
from shapely.geometry import Point

from mimesis_earth.geometry import (
    R_EARTH_KM,
    atoms_polygon,
    cell_polygon,
    xyz_to_lonlat,
)
from mimesis_earth.mesh import build_mesh


def lonlat_to_xyz(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def test_xyz_to_lonlat_roundtrip():
    lon, lat = xyz_to_lonlat(np.array([lonlat_to_xyz(45.0, 30.0)]))
    np.testing.assert_allclose([lon[0], lat[0]], [45.0, 30.0], atol=1e-9)


def test_all_cells_valid_and_in_range():
    mesh = build_mesh(1000, np.random.default_rng(30))
    for i in range(len(mesh.points)):
        poly = cell_polygon(mesh.vertices[mesh.regions[i]])
        assert poly.is_valid, f"cell {i} invalid"
        assert not poly.is_empty
        minx, miny, maxx, maxy = poly.bounds
        assert -180.0001 <= minx <= maxx <= 180.0001
        assert -90.0001 <= miny <= maxy <= 90.0001


def test_pole_cell_contains_pole():
    # hexagon of vertices at lat 85 -> cell encloses the north pole
    lons = np.array([0.0, 60.0, 120.0, 180.0, -120.0, -60.0])
    verts = np.stack([lonlat_to_xyz(lo, 85.0) for lo in lons])
    poly = cell_polygon(verts)
    assert poly.is_valid
    assert poly.contains(Point(10.0, 89.5))


def test_antimeridian_cell_split():
    # small square straddling lon=180
    corners = [(179.0, 10.0), (-179.0, 10.0), (-179.0, 12.0), (179.0, 12.0)]
    verts = np.stack([lonlat_to_xyz(lo, la) for lo, la in corners])
    poly = cell_polygon(verts)
    assert poly.is_valid
    assert poly.geom_type == "MultiPolygon"
    assert poly.bounds[0] >= -180.0001 and poly.bounds[2] <= 180.0001


def test_atoms_polygon_union():
    mesh = build_mesh(1000, np.random.default_rng(31))
    ids = np.arange(40)
    merged = atoms_polygon(mesh, ids)
    assert merged.is_valid
    # union area equals sum of parts (no double-counting, no gaps)
    parts = sum(
        cell_polygon(mesh.vertices[mesh.regions[i]]).area for i in ids
    )
    np.testing.assert_allclose(merged.area, parts, rtol=1e-6)


def test_atoms_polygon_cache_reuse():
    mesh = build_mesh(500, np.random.default_rng(32))
    cache: dict = {}
    a = atoms_polygon(mesh, np.arange(30), cache)
    assert len(cache) == 30
    # overlapping request reuses cached polygons and adds only the new ones
    b = atoms_polygon(mesh, np.arange(10, 50), cache)
    assert len(cache) == 50
    # cached result identical to uncached computation
    fresh = atoms_polygon(mesh, np.arange(10, 50))
    assert b.equals(fresh)
    assert a.is_valid and b.is_valid


def test_earth_radius_constant():
    assert R_EARTH_KM == 6371.0


def test_snap_union_matches_and_snaps_inputs():
    """snap_union general-behavior smoke test. Note: the specific defect this
    guards against (Pyodide/WASM GEOS 3.12 fatally erroring on near-pole ring
    unions) cannot be reproduced on native GEOS; that guarantee rests on the
    manual cross-GEOS byte-identical spike check recorded in the addendum."""
    from mimesis_earth.geometry import PRECISION_GRID, snap_union
    import shapely
    from shapely.geometry import box

    a = box(0, 0, 1, 1)
    b = box(1, 0, 2, 1)  # shares the x=1 edge
    merged = snap_union([a, b])
    assert merged.is_valid
    # union dissolves the shared edge into one 2x1 rectangle
    assert abs(merged.area - 2.0) < 1e-9
    # inputs with sub-grid offsets get snapped, so a near-degenerate sliver
    # collapses rather than producing an invalid noding
    tiny = box(0, 0, 1, 1e-12)  # thinner than the grid
    out = snap_union([tiny])
    assert out.is_valid
