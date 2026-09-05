"""
Sensor Data Cleaner Module.
Implements Butterworth zero-phase filtering, zero-velocity (ZUPT) detection,
gravity/linear acceleration separation, Z-score scaling, and sliding window generation.
"""

from typing import Tuple, Optional, Dict, Union
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


class SensorDataCleaner:
    """
    Cleans, filters, normalizes, and segments multi-modal sensor telemetry streams.
    """

    def __init__(self, sample_rate_hz: float = 100.0) -> None:
        """
        Args:
            sample_rate_hz: Uniform data acquisition rate in Hz (default: 100.0 Hz).
        """
        self.sample_rate_hz = sample_rate_hz
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def apply_butterworth_filter(
        self,
        data: np.ndarray,
        cutoff_hz: float = 15.0,
        filter_type: str = "low",
        order: int = 4
    ) -> np.ndarray:
        """
        Apply zero-phase Butterworth filter to a 1D or 2D sensor signal array.

        Args:
            data: Input raw sensor array [N] or [N, channels].
            cutoff_hz: Cutoff frequency in Hz.
            filter_type: 'low' or 'high'.
            order: Filter order.

        Returns:
            np.ndarray: Denoised sensor signal array.
        """
        nyquist = 0.5 * self.sample_rate_hz
        if cutoff_hz >= nyquist:
            cutoff_hz = nyquist - 0.1

        normal_cutoff = cutoff_hz / nyquist
        b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)

        if data.ndim == 1:
            if len(data) <= 3 * order:
                return data.copy()
            return filtfilt(b, a, data)
        else:
            if len(data) <= 3 * order:
                return data.copy()
            filtered = np.zeros_like(data, dtype=np.float64)
            for col in range(data.shape[1]):
                filtered[:, col] = filtfilt(b, a, data[:, col])
            return filtered

    def detect_zero_velocity(
        self,
        accel_data: np.ndarray,
        gyro_data: np.ndarray,
        window_size: int = 20,
        accel_var_threshold: float = 0.05,
        gyro_var_threshold: float = 0.01
    ) -> np.ndarray:
        """
        Detect Zero-Velocity (ZUPT) stationary conditions using sliding energy variance.

        Args:
            accel_data: 3D acceleration array [N, 3].
            gyro_data: 3D angular velocity array [N, 3].
            window_size: Moving variance window size in timesteps.
            accel_var_threshold: Acceleration variance threshold (m/s^2)^2.
            gyro_var_threshold: Gyroscope variance threshold (rad/s)^2.

        Returns:
            np.ndarray: Boolean array [N] where True indicates zero-velocity / stationary.
        """
        n_samples = len(accel_data)
        is_stationary = np.zeros(n_samples, dtype=bool)

        accel_mag = np.linalg.norm(accel_data, axis=1)
        gyro_mag = np.linalg.norm(gyro_data, axis=1)

        half_w = window_size // 2
        for i in range(n_samples):
            start = max(0, i - half_w)
            end = min(n_samples, i + half_w + 1)

            accel_var = np.var(accel_mag[start:end])
            gyro_var = np.var(gyro_mag[start:end])

            if accel_var < accel_var_threshold and gyro_var < gyro_var_threshold:
                is_stationary[i] = True

        return is_stationary

    def extract_linear_acceleration(
        self,
        accel_data: np.ndarray,
        gravity_cutoff_hz: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate total acceleration into gravity component g and dynamic linear acceleration a_lin.

        Args:
            accel_data: 3D raw acceleration array [N, 3] in m/s^2.
            gravity_cutoff_hz: Low-pass cutoff frequency for gravity component extraction.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (gravity_vector [N, 3], linear_acceleration [N, 3]).
        """
        gravity = self.apply_butterworth_filter(accel_data, cutoff_hz=gravity_cutoff_hz, filter_type="low")
        linear_accel = accel_data - gravity
        return gravity, linear_accel

    def fit_transform_scaler(self, data: np.ndarray) -> np.ndarray:
        """
        Compute Z-score normalization parameters (mean and std) and scale data.

        Args:
            data: Signal array [N, channels].

        Returns:
            np.ndarray: Normalized array with zero mean and unit variance.
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0

        return (data - self.mean) / self.std

    def transform_scaler(self, data: np.ndarray) -> np.ndarray:
        """
        Apply fitted Z-score normalization scaler to new data.

        Args:
            data: Signal array [N, channels].

        Returns:
            np.ndarray: Scaled signal array.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_transform_scaler first.")

        return (data - self.mean) / self.std

    def create_sliding_windows(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        window_size: int = 200,
        step_size: int = 10
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Segment continuous time-series into overlapping windows for ML inference.

        Args:
            data: Feature array of shape [N, channels].
            labels: Optional ground-truth target label array of shape [N] or [N, label_dim].
            window_size: Number of timesteps per window (e.g. 200 = 2 seconds at 100Hz).
            step_size: Window stride timesteps (e.g. 10 = 0.1s hop).

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Windowed feature array [num_windows, window_size, channels]
            and optional label array [num_windows].
        """
        num_samples = len(data)
        if num_samples < window_size:
            raise ValueError(f"Data sample length ({num_samples}) is shorter than window size ({window_size}).")

        num_windows = (num_samples - window_size) // step_size + 1
        num_channels = data.shape[1] if data.ndim > 1 else 1

        X = np.zeros((num_windows, window_size, num_channels), dtype=np.float32)

        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            if data.ndim == 1:
                X[i, :, 0] = data[start:end]
            else:
                X[i] = data[start:end, :]

        if labels is None:
            return X

        # Target label for window is the value at the final timestep of the window
        if labels.ndim == 1:
            y = np.array([labels[i * step_size + window_size - 1] for i in range(num_windows)], dtype=np.float32)
        else:
            y = np.array([labels[i * step_size + window_size - 1, :] for i in range(num_windows)], dtype=np.float32)

        return X, y
