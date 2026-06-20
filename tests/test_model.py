from app.model import ensure_model, predict_churn


def test_predict_returns_valid_probability():
    ensure_model()
    result = predict_churn(
        {
            "tenure": 5,
            "monthly_charges": 70.0,
            "total_charges": 350.0,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
        }
    )
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert isinstance(result["churn"], bool)
