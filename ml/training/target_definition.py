"""
Target Definition Module for Vehicle Forward Velocity Estimation.
Defines the ground-truth target, documents physical SI units (m/s),
and provides conversion and target extraction helpers.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from ml.training.config import TARGET_COLUMN, TARGET_UNIT, KMH_TO_MPS, MPS_TO_KMH


def extract_velocity_target(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the ground-truth vehicle forward velocity target from sequence DataFrame.

    Target selection priority:
    1. 'gps_speed_mps' (m/s) if present.
    2. 'gps_speed_kmh' converted via / 3.6 if 'gps_speed_mps' absent.
    3. 'calc_haversine_speed_mps' fallback if GPS speed unavailable.

    Args:
        df: Processed sequence DataFrame.

    Returns:
        np.ndarray: 1D array of velocity values in m/s (dtype float32).
    """
    cols = df.columns.tolist()

    if TARGET_COLUMN in cols:
        target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    elif "gps_speed_kmh" in cols:
        kmh = pd.to_numeric(df["gps_speed_kmh"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        target = kmh * KMH_TO_MPS
    elif "calc_haversine_speed_mps" in cols:
        target = pd.to_numeric(df["calc_haversine_speed_mps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        raise ValueError("Sequence DataFrame contains no valid target velocity column.")

    # Ensure non-negative speeds (velocity magnitude / forward speed)
    target = np.maximum(target, 0.0)
    return target


def mps_to_kmh(speed_mps: np.ndarray) -> np.ndarray:
    """Convert speed from m/s to km/h."""
    return speed_mps * MPS_TO_KMH


def kmh_to_mps(speed_kmh: np.ndarray) -> np.ndarray:
    """Convert speed from km/h to m/s."""
    return speed_kmh * KMH_TO_MPS


def validate_target_series(target: np.ndarray) -> Tuple[bool, str]:
    """
    Validate target series for NaN, Inf, and physically plausible range (0 to 60 m/s = ~216 km/h).

    Returns:
        Tuple[bool, str]: (is_valid, status_message)
    """
    if len(target) == 0:
        return False, "Target array is empty."
    if np.isnan(target).any():
        return False, "Target contains NaN values."
    if np.isinf(target).any():
        return False, "Target contains Inf values."
    if np.min(target) < 0.0:
        return False, f"Target contains negative values (min: {np.min(target):.2f})."
    if np.max(target) > 70.0:
        return False, f"Target contains unphysically high speed (>70 m/s: max {np.max(target):.2f})."

    return True, f"Target valid. Samples: {len(target)}, Min: {np.min(target):.2f} m/s, Max: {np.max(target):.2f} m/s, Mean: {np.mean(target):.2f} m/s"
