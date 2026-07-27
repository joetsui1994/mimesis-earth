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

world = generate(WorldSpec(
    levels=[8, 6, 9], n_landmasses=4, size_variance=0.4, count_coupling=0.7,
    border_meander=0.5, seed=42,
))
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

## Static / GitHub Pages deploy

The app also runs with **no backend** as a static site: worlds are generated
client-side by the same Python core compiled to WebAssembly (Pyodide). A
visitor sees a pre-baked globe instantly; the live generator loads in the
background and, once ready, spacebar and the sliders produce unlimited worlds
in the browser.

- Build the static bundle: `./scripts/build_static.sh` → `web/dist`
  (bundles the frontend, the `mimesis_earth` wheel, and pre-baked gallery
  worlds; Pyodide + scientific wheels load from the jsdelivr CDN at runtime).
- Deploy: pushing to `main` publishes `web/dist` to GitHub Pages via
  `.github/workflows/deploy.yml`.
- The local `mimesis-earth serve` path is unaffected — it uses the fast
  server API and never loads Pyodide.
