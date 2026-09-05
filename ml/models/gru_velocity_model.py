"""
GRU-based Time-Series Vehicle Velocity Regression Model.
Supports both PyTorch ForwardVelocityGRU and pure NumPy GRU regressor fallback.
Predicts vehicle forward speed (m/s) using 14 smartphone inertial & magnetic channels.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module:
            pass


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -15.0, 15.0)))


if HAS_TORCH:
    class ForwardVelocityGRU(nn.Module):
        """
        PyTorch Gated Recurrent Unit (GRU) time-series regressor for mobile vehicle speed estimation.
        """

        def __init__(
            self,
            input_dim: int = 14,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            self.fc_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(32, 1)
            )

        def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
            gru_out, _ = self.gru(x, h0)
            last_step = gru_out[:, -1, :]
            raw_pred = self.fc_head(last_step)
            return F.softplus(raw_pred)

        def count_parameters(self) -> int:
            """Count total trainable parameters in the model."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_parameters(model: Any) -> int:
    """Helper to get total trainable parameters for PyTorch or NumPy model."""
    if hasattr(model, "count_parameters"):
        return model.count_parameters()
    if HAS_TORCH and isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0


class NumPyGRUVelocityModel:
    """
    Pure NumPy Gated Recurrent Unit (GRU) regressor fallback when PyTorch is not available.
    """

    def __init__(self, input_dim: int = 14, hidden_dim: int = 64, seed: int = 42) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        np.random.seed(seed)

        # Initialize GRU Weights (Xavier uniform)
        scale = 1.0 / np.sqrt(hidden_dim)
        self.Wz = np.random.uniform(-scale, scale, (input_dim + hidden_dim, hidden_dim))
        self.bz = np.zeros(hidden_dim)

        self.Wr = np.random.uniform(-scale, scale, (input_dim + hidden_dim, hidden_dim))
        self.br = np.zeros(hidden_dim)

        self.Wh = np.random.uniform(-scale, scale, (input_dim + hidden_dim, hidden_dim))
        self.bh = np.zeros(hidden_dim)

        # Dense Head
        self.W_out = np.random.uniform(-scale, scale, (hidden_dim, 1))
        self.b_out = np.zeros(1)

    def forward_window(self, x_seq: np.ndarray) -> float:
        """
        Forward pass for a single sequence window [seq_len, input_dim].
        """
        seq_len, _ = x_seq.shape
        h = np.zeros(self.hidden_dim)

        for t in range(seq_len):
            xt = x_seq[t]
            concat = np.concatenate([xt, h])

            z = sigmoid(np.dot(concat, self.Wz) + self.bz)
            r = sigmoid(np.dot(concat, self.Wr) + self.br)

            concat_r = np.concatenate([xt, r * h])
            h_tilde = np.tanh(np.dot(concat_r, self.Wh) + self.bh)

            h = (1.0 - z) * h + z * h_tilde

        raw_out = np.dot(h, self.W_out) + self.b_out
        speed = float(softplus(raw_out[0]))
        return speed

    def predict_batch(self, X_batch: np.ndarray) -> np.ndarray:
        """
        Predict speed array for batch of windows [batch_size, seq_len, input_dim].
        """
        preds = [self.forward_window(X_batch[i]) for i in range(len(X_batch))]
        return np.array(preds, dtype=np.float32).reshape(-1, 1)

    def fit_ridge_head(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Extract GRU sequence hidden state representations and solve linear head using Ridge regression.
        """
        n_windows = len(X_train)
        H_feats = np.zeros((n_windows, self.hidden_dim), dtype=np.float32)

        for i in range(n_windows):
            x_seq = X_train[i]
            h = np.zeros(self.hidden_dim)
            for t in range(len(x_seq)):
                xt = x_seq[t]
                concat = np.concatenate([xt, h])
                z = sigmoid(np.dot(concat, self.Wz) + self.bz)
                r = sigmoid(np.dot(concat, self.Wr) + self.br)
                concat_r = np.concatenate([xt, r * h])
                h_tilde = np.tanh(np.dot(concat_r, self.Wh) + self.bh)
                h = (1.0 - z) * h + z * h_tilde
            H_feats[i] = h

        # Ridge Regression closed-form solution: W = (H^T H + alpha I)^-1 H^T Y
        alpha = 1.0
        reg_I = alpha * np.eye(self.hidden_dim)
        W_sol = np.linalg.solve(H_feats.T @ H_feats + reg_I, H_feats.T @ y_train)

        self.W_out = W_sol
        self.b_out = np.zeros(1)
