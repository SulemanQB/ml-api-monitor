from dataclasses import dataclass

from scipy.stats import ks_2samp


@dataclass(frozen=True)
class DriftResult:
    status: str
    method: str
    p_value: float | None
    baseline_count: int
    recent_count: int


def detect_confidence_drift(
    baseline: list[float], recent: list[float], threshold: float = 0.05
) -> DriftResult:
    """Compare confidence samples with a two-sample KS test."""
    if len(baseline) < 2 or len(recent) < 2:
        return DriftResult("INSUFFICIENT_DATA", "ks", None, len(baseline), len(recent))
    statistic = ks_2samp(baseline, recent)
    status = "DRIFT" if statistic.pvalue < threshold else "NO_DRIFT"
    return DriftResult(status, "ks", float(statistic.pvalue), len(baseline), len(recent))
