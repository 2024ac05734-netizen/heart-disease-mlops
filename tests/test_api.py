"""Integration tests for the FastAPI application using TestClient."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from api.main import app  # noqa: E402

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["label"] in ("Heart Disease", "No Heart Disease")


def test_predict_batch(sample_patient):
    r = client.post("/predict/batch", json={"records": [sample_patient, sample_patient]})
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_predict_validation_error():
    r = client.post("/predict", json={"age": 63})  # missing fields
    assert r.status_code == 422


def test_metrics_endpoint():
    client.post(
        "/predict",
        json={
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        },
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "heart_predictions_total" in r.text
