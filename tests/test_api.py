from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app

client = TestClient(app)

_data = load_breast_cancer()
# First malignant example and first benign example from the raw dataset,
# used as realistic fixtures for the /predict tests below.
_malignant_example = next(
    row.tolist() for row, target in zip(_data.data, _data.target) if target == 0
)
_benign_example = next(
    row.tolist() for row, target in zip(_data.data, _data.target) if target == 1
)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_malignant_example():
    response = client.post("/predict", json={"features": _malignant_example})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "malignant"
    assert body["malignant_probability"] > 0.5


def test_predict_benign_example():
    response = client.post("/predict", json={"features": _benign_example})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "benign"
    assert body["malignant_probability"] < 0.5


def test_predict_rejects_wrong_feature_count():
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 422
