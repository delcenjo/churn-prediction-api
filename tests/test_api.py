from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

HIGH_RISK = {
    "tenure": 1,
    "monthly_charges": 100.0,
    "total_charges": 100.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
}
LOW_RISK = {
    "tenure": 60,
    "monthly_charges": 30.0,
    "total_charges": 1800.0,
    "contract": "Two year",
    "internet_service": "No",
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_expected_shape():
    body = client.post("/predict", json=HIGH_RISK).json()
    assert set(body) == {"churn_probability", "churn", "threshold"}
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_rejects_invalid_input():
    invalid = {**HIGH_RISK, "tenure": -5}
    assert client.post("/predict", json=invalid).status_code == 422


def test_high_risk_scores_above_low_risk():
    high = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    low = client.post("/predict", json=LOW_RISK).json()["churn_probability"]
    assert high > low
