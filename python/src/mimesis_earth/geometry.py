"""Convert spherical Voronoi cells to WGS84 lon/lat shapely polygons.

Handles the two classic sphere-to-plane traps:
- antimeridian: cells straddling lon=+-180 are split into a MultiPolygon
- poles: cells enclosing a pole get an explicit closure ring over the pole
"""

import numpy as np
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

R_EARTH_KM = 6371.0


def xyz_to_lonlat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points)
    lon = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(points[:, 2], -1.0, 1.0)))
    return lon, lat


def _unwrapped_ring(verts3d: np.ndarray) -> list[tuple[float, float]]:
    """Ring in (lon, lat) with longitudes unwrapped into a continuous sequence.
    Cells that enclose a pole get extra points closing the ring over the pole."""
    lon, lat = xyz_to_lonlat(verts3d)
    deltas = (np.diff(lon) + 180.0) % 360.0 - 180.0
    ulon = np.concatenate([[lon[0]], lon[0] + np.cumsum(deltas)])
    closing = ((lon[0] - ulon[-1]) + 180.0) % 360.0 - 180.0
    winding = (ulon[-1] + closing) - ulon[0]
    ring = list(zip(ulon.tolist(), lat.tolist()))
    if abs(winding) > 180.0:  # ring wraps fully around a pole
        sign = 1.0 if winding > 0 else -1.0
        pole_lat = 90.0 if float(np.mean(lat)) > 0 else -90.0
        start_lon, start_lat = ring[0]
        ring = ring + [
            (start_lon + 360.0 * sign, start_lat),
            (start_lon + 360.0 * sign, pole_lat),
            (start_lon, pole_lat),
        ]
    deduped = [ring[0]]
    for pt in ring[1:]:
        prev = deduped[-1]
        if abs(pt[0] - prev[0]) > 1e-12 or abs(pt[1] - prev[1]) > 1e-12:
            deduped.append(pt)
    return deduped


def _normalize_lon(poly):
    """Clip an unwrapped-longitude polygon into [-180, 180], splitting across
    the antimeridian if needed."""
    minx, _, maxx, _ = poly.bounds
    k0 = int(np.floor((minx + 180.0) / 360.0))
    k1 = int(np.floor((maxx + 180.0) / 360.0))
    if k0 == 0 and k1 == 0:
        return poly
    parts = []
    for k in range(k0, k1 + 1):
        piece = poly.intersection(
            box(k * 360.0 - 180.0, -90.0, k * 360.0 + 180.0, 90.0)
        )
        if not piece.is_empty:
            parts.append(translate(piece, xoff=-360.0 * k))
    return unary_union(parts)


def _polygons_only(geom):
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        return unary_union(polys)
    return geom


def cell_polygon(verts3d: np.ndarray):
    """Lon/lat polygon (or MultiPolygon if split) for one Voronoi cell."""
    ring = _unwrapped_ring(verts3d)
    if len(ring) < 3:  # degenerate cell (never seen in practice; stay safe)
        return Polygon()
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = _polygons_only(make_valid(poly))
    return _polygons_only(_normalize_lon(poly))


def atoms_polygon(mesh, atom_ids, cell_cache: dict | None = None):
    """Union of the given atoms' cell polygons. Optional cache: atom id -> polygon."""
    geoms = []
    for i in atom_ids:
        i = int(i)
        if cell_cache is not None and i in cell_cache:
            geoms.append(cell_cache[i])
            continue
        g = cell_polygon(mesh.vertices[mesh.regions[i]])
        if cell_cache is not None:
            cell_cache[i] = g
        geoms.append(g)
    return _polygons_only(unary_union(geoms))
