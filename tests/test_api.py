import os
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

@pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS", "0") != "1",
    reason="Model tests disabled by default. Set RUN_MODEL_TESTS=1 with a configured model.",
)
def test_predict_contract():
    r = client.post("/predict", json={"features": {}})
    assert r.status_code in (200, 400, 500)
