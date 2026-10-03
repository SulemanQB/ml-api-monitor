from dataclasses import dataclass
import os


class ConfigurationError(ValueError):
    """Raised when an environment setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    model_name: str = "tiny-color"
    drift_method: str = "ks"
    drift_threshold: float = 0.05
    baseline_window: int = 20
    recent_window: int = 20
    max_upload_bytes: int = 5_000_000
    max_image_pixels: int = 4_000_000
    log_level: str = "INFO"


def _read_int(name: str, default: int, minimum: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _read_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not minimum < value < maximum:
        raise ConfigurationError(f"{name} must be greater than {minimum} and less than {maximum}")
    return value


def load_settings() -> Settings:
    """Load and validate service configuration from environment variables."""
    model_name = os.getenv("MODEL_NAME", "tiny-color").strip().lower()
    if model_name not in {"tiny-color", "mobilenet_v3_small"}:
        raise ConfigurationError("MODEL_NAME must be tiny-color or mobilenet_v3_small")

    drift_method = os.getenv("DRIFT_METHOD", "ks").strip().lower()
    if drift_method != "ks":
        raise ConfigurationError("DRIFT_METHOD must be ks; no other detector is implemented")

    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError("LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")

    return Settings(
        model_name=model_name,
        drift_method=drift_method,
        drift_threshold=_read_float("DRIFT_THRESHOLD", 0.05, 0.0, 1.0),
        baseline_window=_read_int("BASELINE_WINDOW", 20, 2),
        recent_window=_read_int("RECENT_WINDOW", 20, 2),
        max_upload_bytes=_read_int("MAX_UPLOAD_BYTES", 5_000_000, 1),
        max_image_pixels=_read_int("MAX_IMAGE_PIXELS", 4_000_000, 1),
        log_level=log_level,
    )
