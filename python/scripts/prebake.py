"""Generate a curated set of worlds as static JSON for the gallery.

Each file matches the shape the API returns: {"spec": {...}, "levels": [...]}.
Coordinates keep the generator's native 6-dp rounding (already compact)."""

import json
import sys
from pathlib import Path

from mimesis_earth import WorldSpec, generate

# Curated specs spanning the parameter range, showing off variety.
CURATED = [
    {"levels": [6, 5, 6], "n_landmasses": 5, "spread": 0.7, "coast_ruggedness": 0.5,
     "border_meander": 0.8, "seed": 101},
    {"levels": [5, 4, 4], "n_landmasses": 3, "spread": 0.4, "coast_ruggedness": 0.9,
     "size_variance": 0.8, "border_meander": 1.0, "seed": 202},
    {"levels": [8, 4], "n_landmasses": 6, "spread": 1.0, "coast_ruggedness": 1.0,
     "seed": 303},
    {"levels": [4, 4, 3], "n_landmasses": 2, "spread": 0.2, "coast_ruggedness": 0.3,
     "border_meander": 0.4, "seed": 404},
    {"levels": [7, 5, 5], "n_landmasses": 4, "spread": 0.8, "coast_ruggedness": 0.7,
     "size_variance": 0.4, "seed": 505},
    {"levels": [8, 6], "n_landmasses": 8, "spread": 0.9, "coast_ruggedness": 1.0,
     "seed": 606},
    {"levels": [5, 5, 5], "n_landmasses": 3, "spread": 0.5, "coast_ruggedness": 0.6,
     "border_meander": 0.6, "seed": 707},
    {"levels": [10, 4], "n_landmasses": 5, "spread": 0.6, "coast_ruggedness": 0.8,
     "seed": 808},
    {"levels": [4, 3, 3], "n_landmasses": 2, "spread": 0.3, "coast_ruggedness": 0.4,
     "seed": 909},
    {"levels": [6, 5, 4], "n_landmasses": 6, "spread": 1.0, "coast_ruggedness": 0.95,
     "size_variance": 1.2, "border_meander": 0.9, "seed": 111},
    {"levels": [5, 4], "n_landmasses": 4, "spread": 0.55, "coast_ruggedness": 0.5,
     "seed": 222},
    {"levels": [7, 4, 4], "n_landmasses": 3, "spread": 0.65, "coast_ruggedness": 0.75,
     "border_meander": 0.85, "seed": 333},
]


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, kwargs in enumerate(CURATED):
        spec = WorldSpec(**kwargs)
        world = generate(spec)
        payload = {
            "spec": spec.model_dump(),
            "levels": [
                {"level": lvl, "name": name, "geojson": world.geojson_dict(lvl)}
                for lvl, name in enumerate(spec.level_names)
            ],
        }
        fname = f"world-{i:02d}.json"
        (out / fname).write_text(
            json.dumps(payload, separators=(",", ":"))
        )
        top = world.units_at(0)
        manifest.append({
            "file": fname,
            "seed": spec.seed,
            "countries": len(top),
            "landmasses": spec.n_landmasses,
        })
    (out / "index.json").write_text(json.dumps(manifest, separators=(",", ":")))
    print(f"pre-baked {len(CURATED)} worlds to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "web/public/worlds")
