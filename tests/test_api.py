"""Tests for the FastAPI app."""
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_get_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "message": "Welcome to the Census Income Prediction API!"
    }


def test_post_predicts_low_income(client):
    person = {
        "age": 38,
        "workclass": "Private",
        "fnlgt": 215646,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Divorced",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=person)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_predicts_high_income(client):
    person = {
        "age": 52,
        "workclass": "Private",
        "fnlgt": 99185,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 7298,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    r = client.post("/predict", json=person)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
