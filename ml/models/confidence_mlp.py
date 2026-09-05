"""
Context/Confidence MLP Module.
Estimates real-time measurement noise variances (R) and process noise scales (Q)
for the adaptive EKF/UKF based on multi-sensor signal quality metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicConfidenceMLP(nn.Module):
    """
    Multi-Layer Perceptron that maps sensor confidence indicators (GNSS DOP,
    classifier output probabilities, innovation residual history) to adaptive
    EKF noise variances.
    """

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32) -> None:
        """
        Args:
            input_dim: Number of input signal health features.
            hidden_dim: Hidden layer width.
        """
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Outputs [log(sigma_gnss^2), log(sigma_imu^2)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Health metrics tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Positive variance estimates [batch_size, 2] via exp activation.
        """
        log_variances = self.net(features)
        # Use Softplus or Exp to ensure strictly positive variance bounds
        variances = F.softplus(log_variances) + 1e-4
        return variances
