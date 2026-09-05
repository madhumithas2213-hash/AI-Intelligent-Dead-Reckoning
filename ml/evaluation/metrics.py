"""
Trajectory Evaluation Metrics Module.
Implements Absolute Trajectory Error (ATE), Relative Pose Error (RPE), and Speed RMSE functions.
"""

import numpy as np


def calculate_ate(estimated_pos: np.ndarray, ground_truth_pos: np.ndarray) -> float:
    """
    Compute Absolute Trajectory Error (ATE) Root Mean Square Error (RMSE) in meters.

    Args:
        estimated_pos: Estimated (X, Y) or (X, Y, Z) trajectory array of shape [N, D].
        ground_truth_pos: Ground truth trajectory array of shape [N, D].

    Returns:
        float: ATE RMSE in meters.
    """
    if estimated_pos.shape != ground_truth_pos.shape:
        raise ValueError(f"Shape mismatch: {estimated_pos.shape} vs {ground_truth_pos.shape}")

    error_vectors = estimated_pos - ground_truth_pos
    squared_errors = np.sum(error_vectors ** 2, axis=1)
    rmse = float(np.sqrt(np.mean(squared_errors)))
    return rmse


def calculate_rpe(
    estimated_pos: np.ndarray,
    ground_truth_pos: np.ndarray,
    step_delta: int = 100
) -> float:
    """
    Compute Relative Pose Error (RPE) RMSE over a fixed step interval.

    Args:
        estimated_pos: Estimated trajectory [N, D].
        ground_truth_pos: Ground truth trajectory [N, D].
        step_delta: Step offset interval (e.g., 100 steps = 1 second at 100Hz).

    Returns:
        float: RPE RMSE in meters.
    """
    num_pts = len(estimated_pos)
    if num_pts <= step_delta:
        return 0.0

    rpe_errors = []
    for i in range(num_pts - step_delta):
        est_disp = estimated_pos[i + step_delta] - estimated_pos[i]
        gt_disp = ground_truth_pos[i + step_delta] - ground_truth_pos[i]
        rpe_errors.append(np.linalg.norm(est_disp - gt_disp))

    return float(np.sqrt(np.mean(np.square(rpe_errors))))


def calculate_speed_rmse(predicted_speed: np.ndarray, true_speed: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE) between predicted 1D forward speed and true speed.

    Args:
        predicted_speed: Array of shape [N].
        true_speed: Array of shape [N].

    Returns:
        float: Speed RMSE in m/s.
    """
    return float(np.sqrt(np.mean((predicted_speed - true_speed) ** 2)))
