# Churn Prediction API

This is a small FastAPI service that takes a handful of facts about a customer
and returns how likely that customer is to churn. Under the hood it loads a
scikit-learn pipeline (scaling, one-hot encoding, logistic regression) and
exposes it over HTTP with typed request validation. It runs the same locally,
in tests, and inside a container.

## Asking for a prediction

Send a customer to `POST /predict` as JSON:

```bash
curl -X POST http://localhost:8000/predict \
  -H "content-type: application/json" \
  -d '{
    "tenure": 1,
    "monthly_charges": 100,
    "total_charges": 100,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic"
  }'
```

You get back the probability, a boolean decision, and the threshold that was
applied to reach it:

```json
{ "churn_probability": 0.8936, "churn": true, "threshold": 0.5 }
```

The same request for a settled customer (long tenure, two-year contract, low
charges) comes back near the bottom of the range and `"churn": false`. The
decision is just `churn_probability >= threshold`, and the threshold is fixed at
0.5 (see `app/config.py`).

The five fields are validated by Pydantic before they ever reach the model:

- `tenure`: integer months, 0 to 120
- `monthly_charges`: number, 0 or more
- `total_charges`: number, 0 or more
- `contract`: one of `Month-to-month`, `One year`, `Two year`
- `internet_service`: one of `DSL`, `Fiber optic`, `No`

Anything outside those bounds (a negative tenure, an unknown contract value) is
rejected with a 422 and a description of what went wrong, so the model only ever
sees input it can score.

## The rest of the surface

There are two other endpoints worth knowing about:

- `GET /health` returns `{"status": "ok"}` and is handy for liveness checks.
- `GET /docs` is the interactive OpenAPI page FastAPI generates, where you can
  try `/predict` from the browser.

There is also a `GET /` that just points you at `/docs`.

## Running it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

That serves on http://localhost:8000. You do not have to train a model first:
on startup the app calls `ensure_model()`, which trains and saves
`models/model.joblib` if it is not already there. If you would rather build the
artifact up front, run `python -m app.train` yourself, which also prints the base
churn rate of the generated data.

## Running it in Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

The build trains the model during the image build (`RUN python -m app.train`),
so the model file is baked in and the container is ready to answer requests the
moment it starts. The image is based on `python:3.12-slim` and runs uvicorn on
port 8000.

## Where the model comes from

`app/train.py` generates a deterministic synthetic dataset (seeded, so it is
reproducible) whose churn signal is driven by tenure, monthly charges, contract
type, and internet service. It fits a `Pipeline` of `StandardScaler` plus
`OneHotEncoder` feeding a `LogisticRegression`, then dumps the whole pipeline to
`models/model.joblib` with joblib. At request time `app/model.py` loads that
pipeline once (cached with `lru_cache`), wraps the incoming features in a single
-row DataFrame, and reads off `predict_proba`.

The synthetic data is a stand-in. Swap `model.joblib` for a pipeline trained on
real data with the same five input columns and nothing else has to change.

## Layout

```
app/
  config.py     paths and the decision threshold
  schemas.py    Pydantic request/response models and the allowed enum values
  train.py      build the synthetic dataset and persist the pipeline
  model.py      ensure/load the model and score one customer
  main.py       FastAPI app, lifespan startup, and the endpoints
tests/          API and model tests
Dockerfile      builds the image and bakes in a trained model
.github/workflows/ci.yml   install, train, test on push and pull requests
```

## Tests and CI

The tests use FastAPI's `TestClient`, with a session fixture
(`tests/conftest.py`) that makes sure a model exists before anything runs.
`tests/test_api.py` checks the health endpoint, the shape of a prediction
response, that invalid input gives a 422, and that an obviously high-risk
customer scores above an obviously low-risk one. `tests/test_model.py` exercises
the scoring function directly. Run them with:

```bash
pytest
```

GitHub Actions does the same thing on every push to `main` and on pull requests:
set up Python 3.12, install with `pip install -e ".[dev]"`, train the model, and
run `pytest -q`.
