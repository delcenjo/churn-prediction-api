import pytest

from app.model import ensure_model


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    ensure_model()
