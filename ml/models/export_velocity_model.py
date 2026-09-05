"""
Model Export Script for Smartphone / Edge Deployment (Step 4).
Exports trained PyTorch ForwardVelocityGRU model to TorchScript (.pt) and ONNX (.onnx).
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Any
import torch

from ml.training.config import (
    BEST_MODEL_PATH,
    EXPORT_TORCHSCRIPT_PATH,
    EXPORT_ONNX_PATH,
    INPUT_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    WINDOW_SIZE,
)
from ml.models.gru_velocity_model import ForwardVelocityGRU, count_parameters


def export_velocity_model() -> Dict[str, Any]:
    """
    Export PyTorch velocity GRU model to TorchScript and ONNX formats.
    """
    print("================================================================================")
    print("        STEP 4 — MODEL EXPORT FOR SMARTPHONE / EDGE DEPLOYMENT")
    print("================================================================================")

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model checkpoint missing at: {BEST_MODEL_PATH}")

    # 1. Load PyTorch Checkpoint
    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu")
    model = ForwardVelocityGRU(input_dim=INPUT_DIM, hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    num_params = count_parameters(model)
    print(f"[Model] Loaded model with {num_params:,} parameters.")

    dummy_input = torch.zeros(1, WINDOW_SIZE, INPUT_DIM, dtype=torch.float32)

    # 2. Export TorchScript Model (.pt)
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(EXPORT_TORCHSCRIPT_PATH)
        torchscript_size_kb = EXPORT_TORCHSCRIPT_PATH.stat().st_size / 1024.0
        print(f"[Export PASS] TorchScript model saved: {EXPORT_TORCHSCRIPT_PATH} ({torchscript_size_kb:.1f} KB)")
    except Exception as e:
        print(f"[Export WARN] TorchScript export failed: {e}")
        torchscript_size_kb = 0.0

    # 3. Export ONNX Model (.onnx)
    onnx_size_kb = 0.0
    try:
        torch.onnx.export(
            model,
            dummy_input,
            EXPORT_ONNX_PATH,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["sensor_features"],
            output_names=["predicted_velocity_mps"],
            dynamic_axes={
                "sensor_features": {0: "batch_size", 1: "sequence_length"},
                "predicted_velocity_mps": {0: "batch_size"},
            },
        )
        onnx_size_kb = EXPORT_ONNX_PATH.stat().st_size / 1024.0
        print(f"[Export PASS] ONNX model saved: {EXPORT_ONNX_PATH} ({onnx_size_kb:.1f} KB)")
    except Exception as e:
        print(f"[Export WARN] ONNX export failed: {e}")

    return {
        "num_params": num_params,
        "torchscript_path": str(EXPORT_TORCHSCRIPT_PATH),
        "torchscript_size_kb": torchscript_size_kb,
        "onnx_path": str(EXPORT_ONNX_PATH),
        "onnx_size_kb": onnx_size_kb,
    }


if __name__ == "__main__":
    export_velocity_model()
