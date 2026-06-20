import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_PATH, MODELS_DIR

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
NUMERIC = ["tenure", "monthly_charges", "total_charges"]
CATEGORICAL = ["contract", "internet_service"]


def make_synthetic_dataset(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    monthly_charges = rng.uniform(20, 120, n).round(2)
    total_charges = (monthly_charges * np.maximum(tenure, 1) * rng.uniform(0.8, 1.1, n)).round(2)
    contract = rng.choice(CONTRACTS, n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(INTERNET, n, p=[0.35, 0.45, 0.20])

    logit = (
        -1.0
        - 0.05 * tenure
        + 0.015 * monthly_charges
        + np.where(contract == "Month-to-month", 1.1, 0.0)
        + np.where(contract == "Two year", -1.0, 0.0)
        + np.where(internet == "Fiber optic", 0.6, 0.0)
    )
    churn = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract,
            "internet_service": internet,
            "churn": churn,
        }
    )


def build_pipeline():
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))])


def main():
    data = make_synthetic_dataset()
    features = data.drop(columns=["churn"])
    target = data["churn"]
    pipeline = build_pipeline().fit(features, target)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Trained model -> {MODEL_PATH} (base churn rate {target.mean():.2f})")


if __name__ == "__main__":
    main()
