"""
GRU Based Vehicle Forward Velocity Estimator.
Regresses 1D forward vehicle velocity directly from high-frequency IMU sequence windows.
"""

import torch
import torch.nn as nn


class ForwardVelocityGRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) network for instantaneous vehicle forward speed estimation.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ) -> None:
        """
        Args:
            input_dim: Number of IMU features per timestep (accel, gyro).
            hidden_dim: GRU hidden state dimensionality.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability.
        """
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

        self.fc_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1)  # Predicts 1D forward speed (m/s)
        )

    def forward(self, x: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: IMU temporal window tensor of shape [batch_size, seq_len, input_dim].
            h0: Optional initial hidden state [num_layers, batch_size, hidden_dim].

        Returns:
            torch.Tensor: Predicted speed array of shape [batch_size, 1] (m/s).
        """
        out, h_n = self.gru(x, h0)
        # Take hidden state of the final timestep
        last_timestep_out = out[:, -1, :]
        speed_pred = self.fc_regressor(last_timestep_out)
        return speed_pred
