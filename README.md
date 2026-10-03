# ML API Monitor

**ML API Monitor is a single-process FastAPI demo that exposes image inference and measures latency, confidence, errors, prediction behavior, and a simple confidence-distribution drift signal.**

The service has a validated HTTP boundary, measured inference, structured process logs, operational metrics, statistical monitoring, tests, and a reproducible demo. It runs locally on CPU without a dataset or external service.

## Why This Project Exists

Deploying a model is only one part of the ML lifecycle. After deployment, an engineer needs to know whether requests succeed, inference is getting slower, confidence is changing, and unusual traffic may need investigation. This project turns those concerns into a runnable service rather than a notebook-only experiment.

## What I Built

- A FastAPI `POST /predict` endpoint for image uploads.
- A deterministic `tiny-color` brightness and color heuristic for repeatable demos. It is not a trained model.
- An optional `mobilenet_v3_small` path using torchvision pretrained weights.
- Measured inference-only latency and confidence in every successful response.
- JSON prediction logs without raw image contents.
- Thread-safe in-memory monitoring with request, success, error, latency, confidence, and prediction counters.
- Prometheus-compatible `GET /metrics` output and a Streamlit dashboard.
- A two-sample KS test comparing baseline and recent confidence windows.
- Automated tests plus live smoke and synthetic stress-input demo scripts.


## Architecture

```mermaid
flowchart TD
    A[Client image] --> B[FastAPI /predict]
    B --> C[Upload limits and Pillow decode]
    C --> D[Image preprocessing]
    D --> E[Default brightness/color heuristic or optional MobileNet]
    E --> F[Prediction, confidence, inference latency]
    F --> G[JSON prediction log]
    F --> H[In-memory monitor]
    H --> I[Prometheus-compatible /metrics]
    H --> J[Baseline and recent confidence windows]
    J --> K[KS drift detector]
    K --> L[/monitoring JSON and Streamlit dashboard]
    M[/health] --> B
```

The log stream is structured process output. Drift is calculated from the monitor's bounded confidence windows, not by parsing logs.

## Technologies

| Area | Choice | Why |
| --- | --- | --- |
| API | FastAPI, Uvicorn | Typed HTTP API with generated OpenAPI docs |
| Validation | Pydantic, Pillow | Structured responses and safe image decoding |
| Inference | NumPy default; optional PyTorch/torchvision | The default path is a brightness/color heuristic and needs no checkpoint; optional MobileNet uses pretrained torchvision weights |
| Monitoring | In-memory monitor, Prometheus text format | Minimal local deployment with standard metric names |
| Drift | SciPy two-sample KS test | Explainable comparison of confidence distributions |
| UI | Streamlit | Small monitoring view for demonstrations |
| Testing | Pytest, FastAPI TestClient | Deterministic unit and API coverage |

## Project Structure

```text
ml-api-monitor/
├── app/
│   ├── config.py          # validated environment settings
│   ├── drift.py           # KS confidence drift detector
│   ├── logging_config.py  # JSON logging formatter
│   ├── main.py            # FastAPI application and endpoints
│   ├── model.py           # default and optional model adapters
│   ├── monitoring.py      # thread-safe in-memory metrics
│   └── schemas.py         # API response contracts
├── dashboard/app.py       # Streamlit monitoring dashboard
├── data/README.md         # dataset guidance; no dataset committed
├── diagrams/architecture.mmd
├── scripts/
│   ├── generate_demo.py   # synthetic stress-input traffic generator
│   └── smoke_test.py      # live endpoint check
├── tests/                 # API, monitoring, drift, and config tests
├── .env.example
├── requirements-dashboard.txt
├── requirements-mobilenet.txt
├── pyproject.toml         # pytest discovery configuration
├── requirements.txt
└── README.md
```

## Setup

Requirements:


```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The default `tiny-color` path is a brightness and color heuristic, not a trained classifier. It requires no checkpoint download and does not require Torch. The optional MobileNet path requires a compatible Torch/torchvision installation and may download pretrained weights on first use:

```powershell
python -m pip install -r requirements-mobilenet.txt
$env:MODEL_NAME = "mobilenet_v3_small"
```

## Run the Service

Start the API in one terminal:

```powershell
uvicorn app.main:app --reload
```

Then open the interactive API documentation at http://127.0.0.1:8000/docs.

### API Examples

Health:

```powershell
curl.exe http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_loaded":true}
```

Prediction:

```powershell
curl.exe -X POST -F "file=@path\to\image.png" http://127.0.0.1:8000/predict
```

```json
{
  "prediction": "bright-scene",
  "confidence": 0.998,
  "latency_ms": 0.42,
  "timestamp": "2026-09-29T12:00:00+00:00"
}
```

Monitoring:

```powershell
curl.exe http://127.0.0.1:8000/metrics
curl.exe http://127.0.0.1:8000/monitoring
```

Invalid, empty, oversized, corrupt, or over-large-pixel uploads return a client error without exposing an internal traceback. The default limits are 5 MB and 4 million pixels.

## Demonstration Workflow

With the API running, use the reproducible synthetic traffic generator:

```powershell
python scripts/generate_demo.py
```

It submits ten normal-looking and ten synthetic contrast images and prints measured confidence means, latency mean, latency p95, and drift status. These are stress inputs for demonstrating monitoring mechanics, not a benchmark dataset or a validated OOD evaluation.

Run the dashboard in another terminal:

```powershell
python -m pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

For a quick end-to-end check:

```powershell
python scripts/smoke_test.py
```

## Monitoring and Drift

The monitor records:

- request, success, and error counts
- a bounded sample of inference latency and confidence, plus running sums
- prediction counts and a short list of recent events
- separate baseline and recent confidence windows used by the drift check

The first `BASELINE_WINDOW` successful predictions become the reference window. Later successful predictions fill the recent window. The detector is a two-sample Kolmogorov-Smirnov test. If either window has fewer than two values, the status is `INSUFFICIENT_DATA` and no p-value is reported. Once both windows have at least two values, a p-value below `DRIFT_THRESHOLD` reports `DRIFT`; otherwise it reports `NO_DRIFT`.

Confidence drift is only a monitoring proxy. It does not prove that the input distribution changed, and an OOD image can still receive a high-confidence incorrect prediction.

## Configuration

Settings are read from environment variables and validated at startup:

| Variable | Default | Description |
| --- | ---: | --- |
| `MODEL_NAME` | `tiny-color` | `tiny-color` or `mobilenet_v3_small` |
| `DRIFT_METHOD` | `ks` | The only implemented detector currently supported |
| `DRIFT_THRESHOLD` | `0.05` | KS p-value threshold, exclusive range 0 to 1 |
| `BASELINE_WINDOW` | `20` | Minimum 2 reference confidence values |
| `RECENT_WINDOW` | `20` | Minimum 2 recent confidence values |
| `MAX_UPLOAD_BYTES` | `5000000` | Maximum uploaded image size |
| `MAX_IMAGE_PIXELS` | `4000000` | Maximum decoded image area |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |

## Testing and Results

Run the full suite with:

```powershell
python -m pytest -q
```

The tests cover health behavior, valid and invalid uploads, monitoring counters, Prometheus output, configuration validation, and deterministic KS cases. A local validation run produced 16 passing tests and a successful smoke test. One local demo run measured:

| Metric | Observed |
| --- | ---: |
| Requests | 21, including one smoke request and 20 demo requests |
| Normal confidence mean | 0.997 |
| Synthetic contrast confidence mean | 0.914 |
| Inference latency | 0.66-2.66 ms across that run |
| Drift status | `INSUFFICIENT_DATA` because only one recent value remained after the baseline filled |

Measurements are environment- and sample-dependent. The repository does not claim accuracy for the default path: `tiny-color` is a brightness and color heuristic, not a model trained on a labeled dataset.

## Limitations and Future Improvements

This is a single-process demonstration. Metrics and drift state reset on restart and are not shared across Uvicorn workers. The in-memory monitor is bounded and is not a replacement for persistent Prometheus/Grafana storage. The default path is a deterministic brightness and color heuristic, not a trained or validated production image model, and the confidence signal is not a reliable OOD detector.

Production extensions would include persistent metrics, centralized logs, OpenTelemetry traces, model versioning, input/data-quality monitoring, calibration, a validated dataset, alerting, authentication, containerization, and horizontal-scaling tests.

## Troubleshooting

**`/health` reports `degraded`**: check the startup logs and `MODEL_NAME`. The optional MobileNet path needs compatible Torch/torchvision packages and may need network access for weights.

**The demo cannot connect**: start `uvicorn app.main:app --reload` first, or set `API_URL` to the running service URL.

**The dashboard cannot start**: install `requirements-dashboard.txt` after the core requirements; the dashboard also needs the running API.

**Drift remains `INSUFFICIENT_DATA`**: increase traffic after the baseline fills, or lower the window sizes to values of at least 2 for a controlled demonstration.

## How to Walk Through the Demo

1. Start with [app/main.py](app/main.py) and explain the `/predict` contract and error boundary.
2. Show [app/model.py](app/model.py) and distinguish the deterministic offline model from the optional MobileNet adapter.
3. Run `uvicorn app.main:app --reload`, then open `/docs` and submit an image.
4. Run `python scripts/generate_demo.py` and show the measured confidence and latency output.
5. Open `/metrics` and the Streamlit dashboard to explain operational visibility.
6. Walk through [app/monitoring.py](app/monitoring.py) and [app/drift.py](app/drift.py), especially the baseline/recent windows and small-sample guard.
7. Discuss why confidence drift is only a proxy, why in-memory state does not scale horizontally, and what would be added in production.

## Project Information

This project covers model serving, API design, observability, statistical monitoring, testing, reproducibility, and explicit limitation tracking.