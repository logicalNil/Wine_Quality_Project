import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_predict_red_wine():
    sample_data = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
        "type": "red"
    }

    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    data = response.json()
    assert "quality_class" in data
    assert "probabilities" in data
    assert "class_labels" in data
    assert data["quality_class"] in [0, 1, 2, 3]


def test_predict_white_wine():
    sample_data = {
        "fixed_acidity": 6.6,
        "volatile_acidity": 0.16,
        "citric_acid": 0.4,
        "residual_sugar": 1.5,
        "chlorides": 0.044,
        "free_sulfur_dioxide": 48.0,
        "total_sulfur_dioxide": 143.0,
        "density": 0.9912,
        "pH": 3.54,
        "sulphates": 0.52,
        "alcohol": 12.4,
        "type": "white"
    }

    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    data = response.json()
    assert "quality_class" in data
    assert "probabilities" in data
    assert "class_labels" in data
    assert data["quality_class"] in [0, 1, 2, 3]