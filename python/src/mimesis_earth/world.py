"""World and Unit: the generated product and its export methods."""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from shapely.geometry import MultiPolygon, mapping
from shapely.geometry.polygon import orient

from mimesis_earth.spec import WorldSpec

COORD_DECIMALS = 6  # ~0.1 m; applied identically to shared borders


def _round_coords(obj):
    if isinstance(obj, float):
        return round(obj, COORD_DECIMALS)
    if isinstance(obj, (list, tuple)):
        return [_round_coords(x) for x in obj]
    return obj


def _rfc7946(geom):
    """CCW exteriors, CW holes, per RFC 7946."""
    if geom.geom_type == "Polygon":
        return orient(geom, 1.0)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([orient(p, 1.0) for p in geom.geoms])
    return geom


@dataclass
class Unit:
    id: str
    level: int
    level_name: str
    parent_id: Optional[str]
    name: str
    population: int
    area_km2: float
    centroid_lon: float
    centroid_lat: float
    geometry: object  # shapely Polygon or MultiPolygon, WGS84 lon/lat
    landmass: Optional[int] = None  # set for level-0 units
    elevation_m: int = 0  # area-weighted mean elevation, clamped to >= -100


@dataclass
class World:
    spec: WorldSpec
    units: list[Unit] = field(default_factory=list)

    def units_at(self, level: int) -> list[Unit]:
        return [u for u in self.units if u.level == level]

    def _feature(self, u: Unit) -> dict:
        geom = mapping(u.geometry)
        geom["coordinates"] = _round_coords(geom["coordinates"])
        return {
            "type": "Feature",
            "properties": {
                "id": u.id,
                "name": u.name,
                "level": u.level,
                "level_name": u.level_name,
                "parent_id": u.parent_id,
                "population": u.population,
                "area_km2": round(u.area_km2, 3),
                "centroid_lon": round(u.centroid_lon, COORD_DECIMALS),
                "centroid_lat": round(u.centroid_lat, COORD_DECIMALS),
                "landmass": u.landmass,
                "elevation_m": u.elevation_m,
            },
            "geometry": geom,
        }

    def geojson_dict(self, level: int) -> dict:
        feats = [self._feature(u) for u in self.units_at(level)]
        feats.sort(key=lambda f: f["properties"]["id"])
        return {"type": "FeatureCollection", "features": feats}

    def to_geojson(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for level, name in enumerate(self.spec.level_names):
            path = directory / f"level{level}_{name}.geojson"
            path.write_text(
                json.dumps(
                    self.geojson_dict(level), sort_keys=True, separators=(",", ":")
                )
            )
        (directory / "spec.json").write_text(self.spec.model_dump_json(indent=2))

    def to_csv(self, path) -> None:
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "id", "level", "level_name", "parent_id", "name",
                    "population", "area_km2", "centroid_lon", "centroid_lat",
                    "landmass", "elevation_m",
                ]
            )
            for u in sorted(self.units, key=lambda u: (u.level, u.id)):
                writer.writerow(
                    [
                        u.id, u.level, u.level_name, u.parent_id or "", u.name,
                        u.population, round(u.area_km2, 3),
                        round(u.centroid_lon, COORD_DECIMALS),
                        round(u.centroid_lat, COORD_DECIMALS),
                        u.landmass if u.landmass is not None else "",
                        u.elevation_m,
                    ]
                )

    def gdf(self, level: Optional[int] = None):
        import geopandas  # optional dependency, imported lazily

        units = self.units if level is None else self.units_at(level)
        units = sorted(units, key=lambda u: (u.level, u.id))
        records = [
            {
                "id": u.id,
                "level": u.level,
                "level_name": u.level_name,
                "parent_id": u.parent_id,
                "name": u.name,
                "population": u.population,
                "area_km2": u.area_km2,
                "centroid_lon": u.centroid_lon,
                "centroid_lat": u.centroid_lat,
                "landmass": u.landmass,
                "elevation_m": u.elevation_m,
            }
            for u in units
        ]
        return geopandas.GeoDataFrame(
            records, geometry=[u.geometry for u in units], crs="EPSG:4326"
        )
