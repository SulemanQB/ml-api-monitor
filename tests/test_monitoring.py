from app.monitoring import Monitor


def test_monitor_records_prediction_and_error():
    monitor = Monitor(baseline_window=2, recent_window=2)
    monitor.record_request()
    monitor.record_prediction("bright-scene", 0.9, 4.0, "2026-09-29T00:00:00+00:00")
    monitor.record_request()
    monitor.record_error()
    snapshot = monitor.snapshot()
    assert snapshot["total_requests"] == 2
    assert snapshot["successful_predictions"] == 1
    assert snapshot["failed_predictions"] == 1
    assert snapshot["average_latency_ms"] == 4.0
    assert snapshot["average_confidence"] == 0.9
    assert "ml_prediction_total 1" in monitor.prometheus()

def test_prometheus_summaries_remain_cumulative_after_bounded_history_rolls():
    monitor = Monitor()
    for index in range(501):
        monitor.record_prediction("bright-scene", 0.9, 4.0, str(index))

    metrics = monitor.prometheus()
    assert "ml_inference_latency_seconds_count 501" in metrics
    assert "ml_inference_latency_seconds_sum 2.004000" in metrics
    assert "ml_prediction_confidence_count 501" in metrics
