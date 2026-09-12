"""
Standalone Inference Module for Vehicle Velocity Estimation (Step 4).
Loads trained velocity model and scaler to perform fast CPU/Edge inference on IMU sensor windows.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import pickle
from typing import Dict, Union, Tuple
import numpy as np
import torch

from ml.training.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    INPUT_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    MPS_TO_KMH,
)
from ml.models.gru_velocity_model import ForwardVelocityGRU


class VehicleVelocityPredictor:
    """
    Production-ready vehicle forward velocity predictor for mobile/edge deployment.
    """

    def __init__(self, model_path: Union[str, Path] = BEST_MODEL_PATH, scaler_path: Union[str, Path] = SCALER_PATH) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)

        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Velocity scaler not found at: {self.scaler_path}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Velocity model checkpoint not found at: {self.model_path}")

        # Load Scaler
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load PyTorch Model
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.model_path, map_location="cpu")
        self.model = ForwardVelocityGRU(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_window(self, window_features: np.ndarray) -> Tuple[float, float, float]:
        """
        Predict vehicle forward velocity from a single IMU sensor window.

        Args:
            window_features: 2D numpy array [seq_len, 14] of unscaled sensor measurements.

        Returns:
            Tuple[float, float, float]: (velocity_mps, velocity_kmh, latency_ms)
        """
        t0 = time.perf_counter()

        # Input shape validation
        if window_features.ndim != 2 or window_features.shape[1] != INPUT_DIM:
            raise ValueError(f"Input window must have shape [seq_len, {INPUT_DIM}], got {window_features.shape}")

        # Apply feature scaler
        scaled_win = self.scaler.transform(window_features)
        input_tensor = torch.from_numpy(scaled_win).unsqueeze(0).float()  # [1, seq_len, 14]

        # Model forward pass
        with torch.no_grad():
            pred_tensor = self.model(input_tensor)
            vel_mps = float(pred_tensor.numpy()[0, 0])

        latency_ms = (time.perf_counter() - t0) * 1000.0
        vel_kmh = vel_mps * MPS_TO_KMH

        return vel_mps, vel_kmh, latency_ms

    def predict_batch(self, batch_windows: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict velocity array for a batch of windows.

        Args:
            batch_windows: 3D numpy array [batch_size, seq_len, 14].

        Returns:
            Dict[str, np.ndarray]: Dict containing 'velocity_mps' and 'velocity_kmh'.
        """
        batch_size, seq_len, n_feats = batch_windows.shape
        if n_feats != INPUT_DIM:
            raise ValueError(f"Feature dimension must be {INPUT_DIM}, got {n_feats}")

        # Reshape for scaling
        reshaped = batch_windows.reshape(-1, n_feats)
        scaled_reshaped = self.scaler.transform(reshaped)
        scaled_batch = scaled_reshaped.reshape(batch_size, seq_len, n_feats)

        input_tensor = torch.from_numpy(scaled_batch).float()

        with torch.no_grad():
            preds = self.model(input_tensor).numpy()

        vel_mps = np.maximum(preds, 0.0)
        vel_kmh = vel_mps * MPS_TO_KMH

        return {
            "velocity_mps": vel_mps,
            "velocity_kmh": vel_kmh,
        }


if __name__ == "__main__":
    predictor = VehicleVelocityPredictor()
    dummy_win = np.random.randn(20, INPUT_DIM).astype(np.float32)
    v_mps, v_kmh, lat = predictor.predict_window(dummy_win)
    print(f"[Inference PASS] Velocity: {v_mps:.2f} m/s ({v_kmh:.2f} km/h) | Latency: {lat:.3f} ms")
