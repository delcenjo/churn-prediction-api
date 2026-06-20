# Churn Prediction API

A production-style REST API that serves customer churn predictions. It wraps a
scikit-learn model behind FastAPI with typed request validation, is containerised
with Docker and tested in CI.

It is the deployment counterpart to the [customer churn](https://github.com/delcenjo)
modelling project: the same kind of model, exposed as a service.

## Endpoints

| Method | Path       | Description                          |
| ------ | ---------- | ------------------------------------ |
| GET    | `/health`  | liveness check                       |
| POST   | `/predict` | churn probability for one customer   |
| GET    | `/docs`    | interactive OpenAPI documentation    |

### Example

```bash
curl -X POST http://localhost:8000/predict -H "content-type: application/json" -d '{
  "tenure": 1, "monthly_charges": 100, "total_charges": 100,
  "contract": "Month-to-month", "internet_service": "Fiber optic"
}'
```

```json
{ "churn_probability": 0.8936, "churn": true, "threshold": 0.5 }
```

A long-tenure customer on a two-year contract scores `0.0093` instead. Invalid
input (e.g. a negative tenure) is rejected with HTTP 422 thanks to Pydantic
validation.

## How the model is produced

`app/train.py` builds a scikit-learn `Pipeline` (scaling + one-hot encoding +
logistic regression) on a deterministic synthetic dataset that mimics the churn
signal, and serialises it to `models/model.joblib`. In a real deployment this
artifact would be the model trained in the churn project; the serving code is
unchanged.

## Project structure

```
app/
  config.py     paths and threshold
  schemas.py    Pydantic request/response models
  train.py      build and persist the model
  model.py      load the model and score a customer
  main.py       FastAPI application and endpoints
tests/          model and API tests (FastAPI TestClient)
Dockerfile      container image (bakes in a trained model)
.github/workflows/ci.yml   install, train, test on every push
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m app.train                      # produce models/model.joblib
uvicorn app.main:app --reload            # serve on http://localhost:8000
pytest
```

## Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Possible improvements

- Load the model artifact from object storage or a model registry.
- Add request logging, Prometheus metrics and structured error handling.
- Batch prediction endpoint and a configurable decision threshold.
