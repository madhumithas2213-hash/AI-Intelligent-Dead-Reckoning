"""
Trajectory Evaluator & Plotting Runner.
Loads estimated trajectories vs ground truth and outputs ATE/RPE reports and trajectory plots.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from ml.evaluation.metrics import calculate_ate, calculate_rpe, calculate_speed_rmse


class TrajectoryEvaluator:
    """
    Evaluation manager for trajectory visualization and report generation.
    """

    def __init__(self, output_dir: Union[str, Path] = "ml/outputs/plots") -> None:
        """
        Args:
            output_dir: Directory where evaluation plots will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_and_plot(
        self,
        estimated_traj: np.ndarray,
        ground_truth_traj: np.ndarray,
        title: str = "Trajectory Comparison",
        save_filename: Optional[str] = "trajectory_eval.png"
    ) -> dict[str, float]:
        """
        Compute error metrics and generate 2D trajectory comparison plot.

        Args:
            estimated_traj: Array of shape [N, 2] (X, Y in meters).
            ground_truth_traj: Array of shape [N, 2] (X, Y in meters).
            title: Plot header string.
            save_filename: Image file basename.

        Returns:
            dict: Summary metrics dictionary containing ATE and RPE values.
        """
        ate = calculate_ate(estimated_traj, ground_truth_traj)
        rpe = calculate_rpe(estimated_traj, ground_truth_traj)

        metrics = {"ATE_m": ate, "RPE_m": rpe}

        # Plot 2D Trajectory
        plt.figure(figsize=(10, 6))
        plt.plot(ground_truth_traj[:, 0], ground_truth_traj[:, 1], 'k--', label="Ground Truth (GNSS)", linewidth=2)
        plt.plot(estimated_traj[:, 0], estimated_traj[:, 1], 'r-', label="IDR Estimated Path", linewidth=1.5)
        plt.title(f"{title}\nATE: {ate:.2f} m | RPE: {rpe:.2f} m")
        plt.xlabel("East Offset (m)")
        plt.ylabel("North Offset (m)")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="best")
        plt.axis("equal")

        if save_filename:
            save_path = self.output_dir / save_filename
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[Evaluator] Plot saved to: {save_path.resolve()}")

        plt.close()
        return metrics
