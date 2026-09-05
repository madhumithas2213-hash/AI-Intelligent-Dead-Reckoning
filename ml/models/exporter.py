"""
Model Exporter Module.
Converts PyTorch checkpoint weights to ONNX format for deployment on mobile edge devices.
"""

from pathlib import Path
from typing import Union, Tuple
import torch
import torch.nn as nn


class ModelExporter:
    """
    Utility for exporting PyTorch model weights to ONNX format for Android ONNX Runtime execution.
    """

    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        export_path: Union[str, Path],
        input_names: list[str] = ["input"],
        output_names: list[str] = ["output"],
        dynamic_axes: dict = None
    ) -> Path:
        """
        Export a PyTorch nn.Module to ONNX format.

        Args:
            model: PyTorch model instance in eval mode.
            dummy_input: Sample tensor with correct shape and dtype.
            export_path: Path to write out .onnx file.
            input_names: Name strings for model inputs.
            output_names: Name strings for model outputs.
            dynamic_axes: Dynamic batch/sequence dimensions dict.

        Returns:
            Path: Absolute path to the exported ONNX model artifact.
        """
        out_path = Path(export_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        
        if dynamic_axes is None:
            dynamic_axes = {
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"}
            }

        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

        print(f"[ONNX Export Success] Model saved to: {out_path.resolve()}")
        return out_path
