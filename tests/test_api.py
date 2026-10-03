import io

from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def image_bytes(color: tuple[int, int, int] = (240, 240, 240)) -> bytes:
    image = Image.new("RGB", (32, 32), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_measured_fields():
    response = client.post("/predict", files={"file": ("sample.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]
    assert 0 <= body["confidence"] <= 1
    assert body["latency_ms"] >= 0
    assert body["timestamp"]


def test_invalid_upload_is_rejected():
    response = client.post("/predict", files={"file": ("broken.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


def test_empty_upload_is_rejected():
    response = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_missing_upload_is_rejected():
    response = client.post("/predict")
    assert response.status_code == 422


def test_metrics_include_prediction():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "ml_prediction_total" in response.text
