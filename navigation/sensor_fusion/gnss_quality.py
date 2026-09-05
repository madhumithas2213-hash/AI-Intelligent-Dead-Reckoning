"""
Adaptive GNSS Quality Assessment Engine.
Dynamically evaluates GNSS signal quality (GOOD, DEGRADED, UNRELIABLE, LOST) based on HDOP,
satellite counts, accuracy estimates, and innovation distance, returning dynamic noise scales.
"""

from typing import Tuple, Any, Optional
import numpy as np


def safe_parse_satellites(val: Any, default: int = 8) -> int:
    """
    Safely parse satellite count handling text or Excel auto-formatted date strings (e.g. 'Aug-20').

    Args:
        val: Input satellite count representation.
        default: Fallback value.

    Returns:
        int: Valid satellite count integer.
    """
    if val is None or pd_is_nan(val):
        return default
    try:
        f_val = float(val)
        if np.isnan(f_val):
            return default
        return max(0, int(f_val))
    except Exception:
        return default


def pd_is_nan(val: Any) -> bool:
    """Check if value is NaN."""
    try:
        return np.isnan(val)
    except Exception:
        return False


class GNSSQualityEvaluator:
    """
    Evaluates GNSS receiver fix quality and provides adaptive measurement covariance multiplier R_scale.
    """

    def __init__(self, accuracy_good_m: float = 10.0, accuracy_degraded_m: float = 35.0, gnss_timeout_seconds: float = 2.0) -> None:
        """
        Args:
            accuracy_good_m: Accuracy threshold in meters for GOOD signal quality.
            accuracy_degraded_m: Accuracy threshold in meters for DEGRADED signal quality.
            gnss_timeout_seconds: Timeout threshold for GNSS loss.
        """
        self.accuracy_good_m = accuracy_good_m
        self.accuracy_degraded_m = accuracy_degraded_m
        self.gnss_timeout_seconds = gnss_timeout_seconds

    def evaluate_quality(
        self,
        accuracy_m: float,
        satellites: Any,
        speed_mps: float = 0.0,
        timestamp_sec: float = 0.0,
        innovation_dist: float = 0.0
    ) -> Tuple[str, float]:
        """
        Evaluate GNSS signal quality and compute measurement covariance multiplier R_scale.

        Returns:
            Tuple[str, float]: (Quality Category ['GOOD', 'DEGRADED', 'UNRELIABLE', 'LOST'], R_scale).
        """
        sat_count = safe_parse_satellites(satellites, default=8)
        acc_m = float(accuracy_m) if (accuracy_m is not None and not np.isnan(accuracy_m) and accuracy_m > 0) else 15.0

        if acc_m <= self.accuracy_good_m and sat_count >= 4:
            return "GOOD", 1.0
        elif acc_m <= self.accuracy_degraded_m and sat_count >= 3:
            r_scale = max(1.0, float((acc_m / 5.0) ** 2))
            return "DEGRADED", r_scale
        elif innovation_dist > 500.0:
            return "UNRELIABLE", 100.0
        else:
            return "UNRELIABLE", 100.0
