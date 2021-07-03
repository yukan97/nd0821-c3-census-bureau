import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture(scope='session')
def example_greater_than_fifty():
    example = {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlgt": 287927,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 15024,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    return example


@pytest.fixture(scope='session')
def example_less_than_fifty():
    example = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    return example


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ["Greetings!"]


def test_post_more_than_fifty(example_greater_than_fifty):
    r = client.post("/inference/", json=example_greater_than_fifty)
    assert r.status_code == 200
    assert r.json() == ">50K"


def test_post_less_than_fifty(example_less_than_fifty):

    r = client.post("/inference/", json=example_less_than_fifty)
    assert r.status_code == 200
    assert r.json() == "<=50K"
