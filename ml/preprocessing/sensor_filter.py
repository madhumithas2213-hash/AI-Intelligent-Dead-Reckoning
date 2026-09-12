"""
Sensor Filter Module.
Applies zero-phase Butterworth low-pass filtering to high-frequency IMU channels
while preserving raw sensor streams and static gravity vectors.

Filter Specification & Justification:
- Filter Type: 4th-order Zero-Phase Butterworth Low-Pass Filter (via scipy.signal.filtfilt).
- Cutoff Frequency: 3.0 Hz (for 10 Hz nominal rate, or 15.0 Hz for 100 Hz streams).
- Rationale: Vehicle dynamic maneuvers (longitudinal acceleration, braking, cornering yaw rates)
  are physically bounded below 2.5 Hz. Frequencies above 3.0 Hz represent road surface roughness,
  chassis vibrations, and engine noise. Zero-phase forward-backward filtering avoids temporal phase lag.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


@dataclass
class FilterConfiguration:
    """Configuration parameters and documentation for signal filtering."""
    filter_type: str = "Butterworth Low-Pass (Zero-Phase)"
    order: int = 4
    cutoff_hz: float = 3.0
    sampling_freq_hz: float = 10.0
    justification: str = (
        "Vehicle translational acceleration and cornering maneuvers occur below 2.5 Hz. "
        "Frequencies >3.0 Hz reflect engine vibration and road surface noise. Zero-phase "
        "filtfilt eliminates temporal phase distortion."
    )


class SensorFilter:
    """
    Applies signal filtering to accelerometer, gyroscope, and magnetometer signals.
    """

    def __init__(self, config: Optional[FilterConfiguration] = None) -> None:
        """
        Args:
            config: Optional filter configuration instance.
        """
        self.config = config if config is not None else FilterConfiguration()

    def filter_sequence(self, df: pd.DataFrame, sample_rate_hz: float = 10.0) -> pd.DataFrame:
        """
        Apply zero-phase Butterworth low-pass filter to raw IMU columns while keeping raw values intact.

        Args:
            df: DataFrame containing canonical raw IMU columns.
            sample_rate_hz: Effective sampling rate of the sequence.

        Returns:
            pd.DataFrame: DataFrame augmented with filtered columns (accel_filtered_*, gyro_filtered_*).
        """
        df_filtered = df.copy()

        nyquist = 0.5 * sample_rate_hz
        cutoff = self.config.cutoff_hz

        # Ensure cutoff is below Nyquist frequency
        if cutoff >= nyquist:
            cutoff = max(0.1, nyquist * 0.6)

        b, a = butter(self.config.order, cutoff / nyquist, btype="low", analog=False)

        # Target IMU columns to filter
        accel_raw_cols = ["accel_raw_x_ms2", "accel_raw_y_ms2", "accel_raw_z_ms2"]
        gyro_raw_cols = ["gyro_raw_yaw_rads", "gyro_raw_pitch_rads", "gyro_raw_roll_rads"]

        # Filter Accelerometer
        for col in accel_raw_cols:
            if col in df_filtered.columns:
                out_col = col.replace("accel_raw_", "accel_filtered_")
                signal = pd.to_numeric(df_filtered[col], errors="coerce").to_numpy(dtype=np.float64)
                signal = np.nan_to_num(signal, nan=0.0)
                if len(signal) > 3 * self.config.order:
                    df_filtered[out_col] = filtfilt(b, a, signal)
                else:
                    df_filtered[out_col] = signal

        # Filter Gyroscope
        for col in gyro_raw_cols:
            if col in df_filtered.columns:
                out_col = col.replace("gyro_raw_", "gyro_filtered_")
                signal = pd.to_numeric(df_filtered[col], errors="coerce").to_numpy(dtype=np.float64)
                signal = np.nan_to_num(signal, nan=0.0)
                if len(signal) > 3 * self.config.order:
                    df_filtered[out_col] = filtfilt(b, a, signal)
                else:
                    df_filtered[out_col] = signal

        return df_filtered
