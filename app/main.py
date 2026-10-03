import io
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .config import Settings, load_settings
from .logging_config import configure_logging
from .model import ImageClassifier
from .monitoring import Monitor
from .schemas import HealthResponse, MonitoringSnapshot, PredictionResponse

settings: Settings = load_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML API Monitor", version="1.0.0", description="Image inference with monitoring metrics and confidence drift analysis.")
model = ImageClassifier(settings.model_name)
monitor = Monitor(
    baseline_window=settings.baseline_window,
    recent_window=settings.recent_window,
    drift_threshold=settings.drift_threshold,
)


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in bounded chunks so oversized requests fail safely."""
    chunks: list[bytes] = []
    bytes_read = 0
    while chunk := await file.read(min(64 * 1024, max_bytes - bytes_read + 1)):
        chunks.append(chunk)
        bytes_read += len(chunk)
        if bytes_read > max_bytes:
            raise ValueError("image exceeds the configured upload limit")
    return b"".join(chunks)


def decode_image(contents: bytes, max_pixels: int) -> Image.Image:
    """Decode an image and enforce its maximum decoded area."""
    image = Image.open(io.BytesIO(contents))
    if image.width * image.height > max_pixels:
        raise ValueError("image exceeds the configured pixel limit")
    image.load()
    return image


@app.middleware("http")
async def monitor_prediction_requests(request, call_next):
    is_prediction = request.method == "POST" and request.url.path == "/predict"
    if is_prediction:
        monitor.record_request()
    response = await call_next(request)
    if is_prediction and response.status_code >= 400:
        monitor.record_error()
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    response = HealthResponse(status="ok" if model.loaded else "degraded", model_loaded=model.loaded)
    if not model.loaded:
        return JSONResponse(status_code=503, content=response.model_dump())
    return response


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="an image file is required")
    try:
        contents = await read_upload(file, settings.max_upload_bytes)
        if not contents:
            raise ValueError("empty upload")
        image = await run_in_threadpool(decode_image, contents, settings.max_image_pixels)
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("invalid_image", extra={"event_data": {"filename": file.filename, "error": str(exc)}})
        raise HTTPException(status_code=400, detail="uploaded file is not a valid image") from exc

    try:
        result = await run_in_threadpool(model.predict, image)
    except Exception as exc:
        logger.exception("inference_failed", extra={"event_data": {"error": str(exc)}})
        raise HTTPException(status_code=503, detail="inference is unavailable") from exc

    timestamp = datetime.now(timezone.utc)
    monitor.record_prediction(result.label, result.confidence, result.latency_ms, timestamp.isoformat())
    logger.info("prediction", extra={"event_data": {"event": "prediction", "prediction": result.label, "confidence": result.confidence, "latency_ms": result.latency_ms}})
    return PredictionResponse(prediction=result.label, confidence=result.confidence, latency_ms=result.latency_ms, timestamp=timestamp)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return monitor.prometheus()


@app.get("/monitoring", response_model=MonitoringSnapshot)
def monitoring() -> dict:
    return monitor.snapshot()
