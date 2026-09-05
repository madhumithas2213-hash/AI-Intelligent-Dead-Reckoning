"""
AI/ML Model Correction Interface.
Provides a clean, plug-in interface for current (Step 4 GRU velocity regressor)
and future ML models (acceleration correction, bias estimation, dynamic Q/R noise scaling).
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path

# Attempt import of Step 4 VehicleVelocityPredictor
try:
    from ml.models.infer_velocity import VehicleVelocityPredictor
    from ml.training.config import BEST_MODEL_PATH, SCALER_PATH
    HAS_STEP4_PREDICTOR = True
except ImportError:
    HAS_STEP4_PREDICTOR = False


class MLCorrectionInterface:
    """
    Modular interface for future ML correction models.
    Can predict velocity corrections, acceleration bias corrections, and adaptive noise scales.
    """

    def __init__(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None) -> None:
        """
        Args:
            model_path: Optional path to trained ML model checkpoint.
            scaler_path: Optional path to feature scaler pickle.
        """
        self.is_model_loaded: bool = False
        self.predictor: Optional[Any] = None

        if HAS_STEP4_PREDICTOR:
            m_path = model_path or BEST_MODEL_PATH
            s_path = scaler_path or SCALER_PATH
            if Path(m_path).exists() and Path(s_path).exists():
                try:
                    self.predictor = VehicleVelocityPredictor(model_path=m_path, scaler_path=s_path)
                    self.is_model_loaded = True
                    print("[ML Interface] Step 4 VehicleVelocityPredictor successfully attached.")
                except Exception as e:
                    print(f"[ML Interface] Failed to load Step 4 predictor: {e}. Operating in fallback mode.")

    def predict_velocity_correction(self, sensor_window: np.ndarray) -> Tuple[Optional[float], float]:
        """
        Predict vehicle forward velocity from IMU sensor feature window using attached ML model.

        Args:
            sensor_window: 2D array [seq_len, num_features] of sensor measurements.

        Returns:
            Tuple[Optional[float], float]: (predicted_speed_mps or None, prediction_variance).
        """
        if not self.is_model_loaded or self.predictor is None:
            # Model plug-in not active — do not return fake predictions
            return None, 1.0

        try:
            vel_mps, vel_kmh, _ = self.predictor.predict_window(sensor_window)
            # Estimate nominal model variance (e.g. 0.25 (m/s)^2)
            return float(vel_mps), 0.25
        except Exception as e:
            print(f"[ML Interface] Inference error: {e}")
            return None, 1.0

    def predict_bias_correction(self, sensor_window: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interface for future Deep Bias Estimation network.
        Predicts [accel_bias_x, accel_bias_y] and [gyro_bias_z].

        Returns:
            Tuple[np.ndarray, np.ndarray]: (accel_bias [2], gyro_bias [1]). Defaults to 0.
        """
        # Marked as future plugin
        return np.zeros(2), np.zeros(1)

    def predict_noise_scaling(self, sensor_window: np.ndarray) -> float:
        """
        Interface for future Context Noise Scaling MLP.
        Predicts dynamic process noise scaling Q_scale (e.g. higher in rough terrain).

        Returns:
            float: Q_scale multiplier. Default: 1.0.
        """
        # Marked as future plugin
        return 1.0
