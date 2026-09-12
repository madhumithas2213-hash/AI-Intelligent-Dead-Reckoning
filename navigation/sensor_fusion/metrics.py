"""
Sensor Fusion Performance Metrics Module.
Calculates Position RMSE, Mean Error, Max Error, Final Position Error, Velocity RMSE,
and Heading Error metrics.
"""

from typing import Dict, Any, Tuple
import numpy as np


class SensorFusionMetrics:
    """
    Computes standard quantitative navigation evaluation metrics.
    """

    @staticmethod
    def calculate_position_errors(
        est_x: np.ndarray,
        est_y: np.ndarray,
        ref_x: np.ndarray,
        ref_y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute position accuracy metrics in meters.

        Args:
            est_x: Estimated East coordinates array.
            est_y: Estimated North coordinates array.
            ref_x: Reference ground truth East coordinates array.
            ref_y: Reference ground truth North coordinates array.

        Returns:
            Dict[str, float]: Dictionary containing rmse, mean, max, final position error.
        """
        errors = np.sqrt((est_x - ref_x)**2 + (est_y - ref_y)**2)
        
        rmse = float(np.sqrt(np.mean(errors**2)))
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        final_err = float(errors[-1]) if len(errors) > 0 else 0.0

        return {
            "position_rmse_m": rmse,
            "mean_position_error_m": mean_err,
            "max_position_error_m": max_err,
            "final_position_error_m": final_err,
        }

    @staticmethod
    def calculate_velocity_rmse(est_speed: np.ndarray, ref_speed: np.ndarray) -> float:
        """
        Compute Velocity Root Mean Squared Error (m/s).

        Args:
            est_speed: Estimated speed array (m/s).
            ref_speed: Reference ground truth speed array (m/s).

        Returns:
            float: Velocity RMSE in m/s.
        """
        return float(np.sqrt(np.mean((est_speed - ref_speed)**2)))

    @staticmethod
    def calculate_heading_rmse_deg(est_heading_rad: np.ndarray, ref_heading_rad: np.ndarray) -> float:
        """
        Compute Heading Root Mean Squared Error (degrees), accounting for circular wrapping [-pi, pi].

        Args:
            est_heading_rad: Estimated heading array in radians.
            ref_heading_rad: Reference heading array in radians.

        Returns:
            float: Heading RMSE in degrees.
        """
        # Circular angular difference
        diff_rad = (est_heading_rad - ref_heading_rad + np.pi) % (2 * np.pi) - np.pi
        diff_deg = np.degrees(diff_rad)
        return float(np.sqrt(np.mean(diff_deg**2)))

    @classmethod
    def evaluate_trajectory(
        cls,
        est_x: np.ndarray,
        est_y: np.ndarray,
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        est_speed: np.ndarray,
        ref_speed: np.ndarray,
        est_heading: Optional[np.ndarray] = None,
        ref_heading: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Comprehensive trajectory evaluation pass.
        """
        res = cls.calculate_position_errors(est_x, est_y, ref_x, ref_y)
        res["velocity_rmse_mps"] = cls.calculate_velocity_rmse(est_speed, ref_speed)

        if est_heading is not None and ref_heading is not None:
            res["heading_rmse_deg"] = cls.calculate_heading_rmse_deg(est_heading, ref_heading)
        else:
            res["heading_rmse_deg"] = 0.0

        return res
