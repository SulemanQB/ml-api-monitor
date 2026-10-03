from collections import Counter, deque
from statistics import mean, median
from threading import RLock

from .drift import DriftResult, detect_confidence_drift


class Monitor:
    def __init__(self, baseline_window: int = 20, recent_window: int = 20, drift_threshold: float = 0.05):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.drift_threshold = drift_threshold
        self._lock = RLock()
        self.total_requests = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.latencies: deque[float] = deque(maxlen=500)
        self.confidences: deque[float] = deque(maxlen=500)
        self.latency_count = 0
        self.latency_sum_ms = 0.0
        self.confidence_count = 0
        self.confidence_sum = 0.0
        self.baseline_confidences: deque[float] = deque(maxlen=baseline_window)
        self.recent_confidences: deque[float] = deque(maxlen=recent_window)
        self.predictions: Counter[str] = Counter()
        self.events: deque[dict] = deque(maxlen=25)

    def record_request(self) -> None:
        with self._lock:
            self.total_requests += 1

    def record_error(self) -> None:
        with self._lock:
            self.failed_predictions += 1

    def record_prediction(self, prediction: str, confidence: float, latency_ms: float, timestamp: str) -> None:
        event = {"prediction": prediction, "confidence": confidence, "latency_ms": latency_ms, "timestamp": timestamp}
        with self._lock:
            self.successful_predictions += 1
            self.latencies.append(latency_ms)
            self.confidences.append(confidence)
            self.latency_count += 1
            self.latency_sum_ms += latency_ms
            self.confidence_count += 1
            self.confidence_sum += confidence
            self.predictions[prediction] += 1
            self.events.appendleft(event)
            if len(self.baseline_confidences) < self.baseline_window:
                self.baseline_confidences.append(confidence)
            else:
                self.recent_confidences.append(confidence)

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
        return ordered[index]

    def drift(self) -> DriftResult:
        with self._lock:
            return detect_confidence_drift(list(self.baseline_confidences), list(self.recent_confidences), self.drift_threshold)

    def snapshot(self) -> dict:
        with self._lock:
            latencies = list(self.latencies)
            confidences = list(self.confidences)
            return {
                "total_requests": self.total_requests,
                "successful_predictions": self.successful_predictions,
                "failed_predictions": self.failed_predictions,
                "average_latency_ms": mean(latencies) if latencies else None,
                "p50_latency_ms": median(latencies) if len(latencies) >= 2 else None,
                "p95_latency_ms": self._percentile(latencies, 95) if len(latencies) >= 2 else None,
                "average_confidence": mean(confidences) if confidences else None,
                "prediction_distribution": dict(self.predictions),
                "recent_events": list(self.events),
                "drift": self.drift().__dict__,
            }

    def prometheus(self) -> str:
        with self._lock:
            lines = [
                "# HELP ml_requests_total Total API requests.",
                "# TYPE ml_requests_total counter",
                f"ml_requests_total {self.total_requests}",
                "# HELP ml_prediction_total Successful predictions.",
                "# TYPE ml_prediction_total counter",
                f"ml_prediction_total {self.successful_predictions}",
                "# HELP ml_prediction_errors_total Failed prediction requests.",
                "# TYPE ml_prediction_errors_total counter",
                f"ml_prediction_errors_total {self.failed_predictions}",
            ]
            if self.latencies:
                lines.extend([
                    "# TYPE ml_inference_latency_seconds summary",
                    f"ml_inference_latency_seconds_count {self.latency_count}",
                    f"ml_inference_latency_seconds_sum {self.latency_sum_ms / 1000:.6f}",
                ])
            if self.confidences:
                lines.extend([
                    "# TYPE ml_prediction_confidence summary",
                    f"ml_prediction_confidence_count {self.confidence_count}",
                    f"ml_prediction_confidence_sum {self.confidence_sum:.6f}",
                ])
            return "\n".join(lines) + "\n"
