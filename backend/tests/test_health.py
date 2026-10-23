from fastapi.testclient import TestClient

from app.main import APP_VERSION, app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": APP_VERSION}


def test_health_rejects_post():
    response = client.post("/health")
    assert response.status_code == 405


def test_unknown_route_returns_404():
    response = client.get("/does-not-exist")
    assert response.status_code == 404
