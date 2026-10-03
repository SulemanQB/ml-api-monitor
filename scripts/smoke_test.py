"""Run a small end-to-end check against a running API."""
import io
import os
import sys

import requests
from PIL import Image

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def image_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), (220, 220, 220))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def check(condition: bool, label: str) -> None:
    print(f"{label:<28} {'PASS' if condition else 'FAIL'}")
    if not condition:
        raise SystemExit(1)


def main() -> None:
    print("====================================")
    print("ML API MONITOR - SMOKE TEST")
    print("====================================")
    health = requests.get(f"{API_URL}/health", timeout=10).json()
    check(health.get("status") in {"ok", "degraded"}, "Health endpoint")
    prediction = requests.post(f"{API_URL}/predict", files={"file": ("smoke.png", image_bytes(), "image/png")}, timeout=30)
    body = prediction.json()
    check(prediction.ok and bool(body.get("prediction")), "Prediction endpoint")
    check(body.get("latency_ms", -1) >= 0, "Latency logging")
    check(0 <= body.get("confidence", -1) <= 1, "Confidence logging")
    metrics = requests.get(f"{API_URL}/metrics", timeout=10).text
    check("ml_prediction_total" in metrics, "Metrics endpoint")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"Smoke test could not reach {API_URL}: {exc}", file=sys.stderr)
        raise SystemExit(1)
