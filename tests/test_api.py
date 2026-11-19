from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict(monkeypatch):
    monkeypatch.setattr("src.api.predict_cpu", lambda history: 45.5)

    response = client.post("/predict", json={
        "vm_id": "vm-101",
        "cpu_history": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    })

    assert response.status_code == 200
    assert response.json()["predicted_cpu"] == 45.5
    assert response.json()["vm_id"] == "vm-101"


def test_optimize_low_cpu(monkeypatch):
    monkeypatch.setattr("src.api.predict_cpu", lambda history: 10.0)

    response = client.post("/optimize", json={
        "vm_id": "vm-low",
        "cpu_history": [5] * 12
    })

    assert response.status_code == 200
    assert response.json()["status"] == "UNDERUTILIZED"


def test_optimize_high_cpu(monkeypatch):
    monkeypatch.setattr("src.api.predict_cpu", lambda history: 92.0)

    response = client.post("/optimize", json={
        "vm_id": "vm-high",
        "cpu_history": [90] * 12
    })

    assert response.status_code == 200
    assert response.json()["status"] == "CRITICAL"


def test_invalid_history():
    response = client.post("/predict", json={
        "vm_id": "vm-invalid",
        "cpu_history": [10, 20, 30]
    })

    assert response.status_code == 422


def test_invalid_cpu_values():
    response = client.post("/predict", json={
        "vm_id": "vm-invalid",
        "cpu_history": [10, 20, 30, 40, 50, 60,
                        70, 80, 90, 100, 120, 130]
    })

    assert response.status_code == 422