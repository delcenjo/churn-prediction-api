from contextlib import asynccontextmanager

from fastapi import FastAPI

from .model import ensure_model, load_model, predict_churn
from .schemas import CustomerFeatures, PredictionResponse


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_model()
    load_model()
    yield


app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def root():
    return {"name": "Churn Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    return predict_churn(features.model_dump())
