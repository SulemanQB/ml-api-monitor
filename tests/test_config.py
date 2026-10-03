import pytest

from app.config import ConfigurationError, load_settings


def test_load_settings_uses_safe_defaults(monkeypatch):
    for name in (
        "MODEL_NAME",
        "DRIFT_METHOD",
        "DRIFT_THRESHOLD",
        "BASELINE_WINDOW",
        "RECENT_WINDOW",
        "MAX_UPLOAD_BYTES",
        "MAX_IMAGE_PIXELS",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()
    assert settings.model_name == "tiny-color"
    assert settings.drift_method == "ks"
    assert settings.baseline_window == 20


@pytest.mark.parametrize(
    ("name", "value"),
    [("DRIFT_THRESHOLD", "1.0"), ("BASELINE_WINDOW", "1"), ("LOG_LEVEL", "verbose"), ("MODEL_NAME", "unknown")],
)
def test_load_settings_rejects_invalid_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError):
        load_settings()
