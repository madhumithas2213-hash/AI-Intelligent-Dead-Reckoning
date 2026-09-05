"""
1D CNN Based Motion and Road Surface Disturbance Classifier.
Classifies phone placement dynamic state (e.g. fixed mount, hand-held, seat)
and road roughness/disturbances from 6-DOF IMU streams.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionDisturbanceCNN(nn.Module):
    """
    1D Convolutional Neural Network for classifying dynamic vehicle dynamic contexts
    and IMU disturbance modes.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 4,
        seq_len: int = 200
    ) -> None:
        """
        Args:
            in_channels: Number of input signal channels (e.g. Accel X,Y,Z + Gyro X,Y,Z).
            num_classes: Target dynamic state categories.
            seq_len: Window length in timesteps.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.seq_len = seq_len

        # Feature Extraction Layers
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)

        # Calculate flattened dimension after 3 max-pooling operations
        conv_out_len = seq_len // 8
        self.fc1 = nn.Linear(128 * conv_out_len, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, in_channels, seq_len]

        Returns:
            torch.Tensor: Logits of shape [batch_size, num_classes]
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Dense Head
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        logits = self.fc_out(x)
        return logits
