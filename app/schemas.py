from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    timestamp: datetime


class DriftResponse(BaseModel):
    status: str
    method: str
    p_value: float | None = None
    baseline_count: int
    recent_count: int


class MonitoringSnapshot(BaseModel):
    total_requests: int
    successful_predictions: int
    failed_predictions: int
    average_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    average_confidence: float | None
    prediction_distribution: dict[str, int]
    recent_events: list[dict]
    drift: DriftResponse
