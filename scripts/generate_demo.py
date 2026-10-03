"""Send generated normal and synthetic contrast images through the running API."""
import io
import os
import statistics

import requests
from PIL import Image, ImageDraw

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def make_image(kind: str, index: int) -> bytes:
    image = Image.new("RGB", (96, 96), (235, 235, 235) if kind == "normal" else (15, 20, 35))
    draw = ImageDraw.Draw(image)
    if kind == "normal":
        draw.rectangle((16, 16, 80, 80), fill=(220, 220, 220))
    else:
        draw.ellipse((10 + index, 18, 85, 86), fill=(20, 100 + index * 10, 220))
        draw.line((0, index * 3, 95, 95 - index * 3), fill=(255, 220, 40), width=5)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def send(kind: str, count: int) -> list[dict]:
    results = []
    for index in range(count):
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (f"{kind}-{index}.png", make_image(kind, index), "image/png")},
            timeout=30,
        )
        response.raise_for_status()
        results.append(response.json())
    return results


def main() -> None:
    normal = send("normal", 10)
    ood = send("ood", 10)
    monitoring = requests.get(f"{API_URL}/monitoring", timeout=10).json()
    latencies = [item["latency_ms"] for item in normal + ood]
    print("ML API MONITOR - DEMO")
    print(f"normal confidence mean: {statistics.mean(item['confidence'] for item in normal):.3f}")
    print(f"ood confidence mean: {statistics.mean(item['confidence'] for item in ood):.3f}")
    print(f"inference latency mean: {statistics.mean(latencies):.3f} ms")
    print(f"inference latency p95: {sorted(latencies)[round(0.95 * (len(latencies) - 1))]:.3f} ms")
    print(f"drift status: {monitoring['drift']['status']}")
    print(f"metrics: {API_URL}/metrics")


if __name__ == "__main__":
    main()
