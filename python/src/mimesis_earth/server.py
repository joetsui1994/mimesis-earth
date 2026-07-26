"""Stateless HTTP wrapper: POST a WorldSpec, get a world back as GeoJSON."""

from importlib import resources

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mimesis_earth.generate import generate
from mimesis_earth.spec import WorldSpec

app = FastAPI(title="mimesis-earth", docs_url=None, redoc_url=None)


@app.post("/api/generate")
def generate_world(spec: WorldSpec) -> dict:
    world = generate(spec)
    return {
        "spec": spec.model_dump(),
        "levels": [
            {
                "level": level,
                "name": name,
                "geojson": world.geojson_dict(level),
            }
            for level, name in enumerate(spec.level_names)
        ],
    }


def _mount_frontend() -> None:
    webapp = resources.files("mimesis_earth") / "webapp"
    if webapp.is_dir():
        app.mount("/", StaticFiles(directory=str(webapp), html=True), name="webapp")


_mount_frontend()
