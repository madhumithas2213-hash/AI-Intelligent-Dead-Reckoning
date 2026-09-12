"""
Sensor Cleaner Module.
Handles NaN and missing value imputation across high-frequency IMU and low-frequency GPS streams.
Employs linear interpolation for short IMU missing sequences (<=3 samples) and forward-fill
for GPS, while refusing to interpolate GPS across large temporal gaps (>5 seconds).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class CleaningStatistics:
    """Tracks missing value statistics per sensor column before and after cleaning."""
    total_rows: int
    missing_before: Dict[str, int]
    missing_after: Dict[str, int]
    pct_missing_before: Dict[str, float]


class SensorCleaner:
    """
    Cleans missing values while keeping raw sensor columns intact.
    """

    def __init__(self, max_imu_interpolate_gap: int = 3, max_gps_gap_sec: float = 5.0) -> None:
        """
        Args:
            max_imu_interpolate_gap: Maximum consecutive missing IMU samples to linearly interpolate.
            max_gps_gap_sec: Gap threshold in seconds beyond which GPS is not interpolated.
        """
        self.max_imu_interpolate_gap = max_imu_interpolate_gap
        self.max_gps_gap_sec = max_gps_gap_sec

    def clean_sequence_sensors(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningStatistics]:
        """
        Clean missing values across IMU, Magnetometer, Orientation, and GPS channels.

        Args:
            df: Processed DataFrame containing canonical sensor channels.

        Returns:
            Tuple[pd.DataFrame, CleaningStatistics]: (DataFrame with clean columns, cleaning stats).
        """
        df_clean = df.copy()

        # Target sensor channels to inspect
        sensor_cols = [col for col in df_clean.columns if not col.startswith("_") and col != "date_timestamp_str"]

        total_rows = len(df_clean)
        missing_before = {col: int(df_clean[col].isna().sum()) for col in sensor_cols if col in df_clean.columns}
        pct_missing_before = {col: (count / total_rows) * 100.0 for col, count in missing_before.items()}

        # Identify IMU vs GPS columns
        imu_cols = [c for c in sensor_cols if "accel" in c or "gyro" in c or "mag" in c or "gravity" in c or "orientation" in c]
        gps_cols = [c for c in sensor_cols if "gps" in c]

        # 1. Clean IMU columns using linear interpolation bounded by max gap limit
        for col in imu_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                if df_clean[col].isna().sum() > 0:
                    df_clean[col] = df_clean[col].interpolate(method="linear", limit=self.max_imu_interpolate_gap)
                    df_clean[col] = df_clean[col].bfill().ffill()

        # 2. Clean GPS columns using forward fill, but respect large temporal gaps
        if "dt_ms" in df_clean.columns and "timestamp_sec" in df_clean.columns:
            large_gaps = df_clean["dt_ms"] > (self.max_gps_gap_sec * 1000.0)
        else:
            large_gaps = pd.Series(False, index=df_clean.index)

        for col in gps_cols:
            if col in df_clean.columns and df_clean[col].isna().sum() > 0:
                # Forward fill GPS observations
                df_clean[col] = df_clean[col].ffill().bfill()
                # Nullify GPS during large temporal outage gaps
                if large_gaps.any():
                    df_clean.loc[large_gaps, col] = np.nan

        missing_after = {col: int(df_clean[col].isna().sum()) for col in sensor_cols if col in df_clean.columns}

        stats = CleaningStatistics(
            total_rows=total_rows,
            missing_before=missing_before,
            missing_after=missing_after,
            pct_missing_before=pct_missing_before,
        )

        return df_clean, stats
