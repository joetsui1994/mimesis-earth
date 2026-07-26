from fastapi.testclient import TestClient

from mimesis_earth.server import app

client = TestClient(app)

SPEC = {
    "levels": [3, 3],
    "n_landmasses": 2,
    "resolution": 4000,
    "seed": 1,
}


def test_generate_endpoint():
    resp = client.post("/api/generate", json=SPEC)
    assert resp.status_code == 200
    data = resp.json()
    assert data["spec"]["levels"] == [3, 3]
    assert data["spec"]["seed"] == 1
    assert len(data["levels"]) == 2
    assert data["levels"][0]["name"] == "country"
    fc = data["levels"][0]["geojson"]
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 3
    assert len(data["levels"][1]["geojson"]["features"]) == 9


def test_invalid_spec_names_field():
    resp = client.post("/api/generate", json={**SPEC, "spread": 3.0})
    assert resp.status_code == 422
    assert "spread" in resp.text


def test_impossible_spec_is_422_not_500():
    resp = client.post(
        "/api/generate", json={"levels": [30, 30, 30], "resolution": 2000}
    )
    assert resp.status_code == 422
    assert "resolution" in resp.text
