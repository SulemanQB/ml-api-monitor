from app.drift import detect_confidence_drift


def test_identical_distributions_have_no_drift():
    values = [0.8, 0.82, 0.81, 0.79, 0.83]
    result = detect_confidence_drift(values, values)
    assert result.status == "NO_DRIFT"
    assert result.p_value == 1.0


def test_shifted_distributions_have_drift():
    baseline = [0.8, 0.82, 0.81, 0.79, 0.83, 0.84]
    recent = [0.1, 0.12, 0.11, 0.09, 0.13, 0.14]
    result = detect_confidence_drift(baseline, recent)
    assert result.status == "DRIFT"
    assert result.p_value is not None


def test_small_samples_are_not_overinterpreted():
    result = detect_confidence_drift([0.8], [0.2])
    assert result.status == "INSUFFICIENT_DATA"
