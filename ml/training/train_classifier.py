"""
Training script placeholder for 1D CNN Motion/Disturbance Classifier.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.models.cnn_classifier import MotionDisturbanceCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Train 1D CNN Motion Classifier")
    parser.add_argument("--data_dir", type=str, default="dataset/processed", help="Path to processed dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="ml/outputs/models", help="Artifact export path")
    return parser.parse_args()


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device):
    """Placeholder for single epoch training loop."""
    model.train()
    total_loss = 0.0
    # Implementation will iterate over loader once dataset is available
    return total_loss


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Initializing MotionDisturbanceCNN training on device: {device}")

    model = MotionDisturbanceCNN(in_channels=6, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("[Training] Model architecture initialized. Awaiting dataset loading pipeline...")


if __name__ == "__main__":
    main()
