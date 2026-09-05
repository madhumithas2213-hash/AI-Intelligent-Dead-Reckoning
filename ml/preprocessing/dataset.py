"""
PyTorch IMU Dataset Module.
Wraps windowed sensor feature arrays and ground truth targets into PyTorch Datasets
for training and evaluating deep learning models (1D CNN, GRU, Confidence MLP).
"""

from typing import Optional, Tuple, Union
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        """Fallback Dataset base class when PyTorch is not installed."""
        pass


class IMUWindowDataset(Dataset):
    """
    Dataset wrapper for overlapping IMU feature windows and target labels.
    Compatible with PyTorch DataLoader when torch is available.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        transform: Optional[callable] = None
    ) -> None:
        """
        Args:
            features: Feature window array of shape [num_windows, window_size, num_channels].
            labels: Target label array of shape [num_windows] or [num_windows, label_dim].
            transform: Optional feature transform callable.
        """
        if features.ndim != 3:
            raise ValueError(f"Features must be 3D array [num_windows, seq_len, channels]. Got shape {features.shape}")

        if HAS_TORCH:
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        else:
            self.features = features.astype(np.float32)
            self.labels = labels.astype(np.float32) if labels is not None else None

        self.transform = transform
        self.transform = transform

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get single window item.

        Args:
            idx: Window index.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Single input feature window tensor
            [seq_len, channels] or (features, label).
        """
        x = self.features[idx]
        if self.transform is not None:
            x = self.transform(x)

        if self.labels is None:
            return x

        y = self.labels[idx]
        return x, y
