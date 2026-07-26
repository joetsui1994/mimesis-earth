# mimesis-earth

Rapidly generate synthetic world geographies: strictly nested administrative
units (countries → provinces → districts) with organic coastlines, real WGS84
coordinates, and consistent synthetic demographics. Deterministic: a spec +
seed always reproduces the same world.

## Install & run

```bash
python -m venv .venv && .venv/bin/pip install -e './python[dev]'
./scripts/build_web.sh            # embeds the web UI (needs Node >= 18; run once, and after frontend changes)
.venv/bin/mimesis-earth serve     # open http://localhost:8000
```

Press **space** for a new world. Drag to rotate, scroll to zoom, click a
polygon to inspect. The top-left panel sets parameters for the next world.

## Python API

```python
from mimesis_earth import WorldSpec, generate

world = generate(WorldSpec(levels=[8, 6, 9], n_landmasses=4, seed=42))
world.gdf(level=2)          # geopandas GeoDataFrame of districts
world.to_geojson("out/")    # one FeatureCollection per level + spec.json
world.to_csv("out/units.csv")
```

## Development

- Python package: `python/` — `cd python && ../.venv/bin/pytest`
- Frontend: `web/` — `cd web && npm install && npm run dev` (proxies /api to :8000)
- Node >= 18 required for frontend development
- Embed frontend in the package: `./scripts/build_web.sh`

Design docs: `docs/superpowers/specs/`.
