from functools import lru_cache

import joblib
import pandas as pd

from .config import MODEL_PATH, THRESHOLD
from .train import main as train_model


def ensure_model():
    if not MODEL_PATH.exists():
        train_model()


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def predict_churn(features, threshold=THRESHOLD):
    probability = float(load_model().predict_proba(pd.DataFrame([features]))[0, 1])
    return {
        "churn_probability": round(probability, 4),
        "churn": probability >= threshold,
        "threshold": threshold,
    }
