"""
Training script placeholder for Context/Confidence MLP.
"""

import argparse
import torch
import torch.nn as nn

from ml.models.confidence_mlp import DynamicConfidenceMLP


def parse_args():
    parser = argparse.ArgumentParser(description="Train Context/Confidence MLP")
    parser.add_argument("--data_dir", type=str, default="dataset/processed")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Initializing DynamicConfidenceMLP training on device: {device}")

    model = DynamicConfidenceMLP(input_dim=8, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("[Training] Confidence MLP Model initialized.")


if __name__ == "__main__":
    main()
