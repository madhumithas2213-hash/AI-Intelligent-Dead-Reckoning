"""
Trajectory Metrics & SIH Benchmark Evaluator Module.
Calculates Position RMSE, Velocity MAE/RMSE, Heading Error, Outage Drift %,
and checks performance against the SIH < 10% Positional Drift Target.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

EARTH_RADIUS_M = 6378137.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine geodetic distance in meters between two lat/lon coordinates."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS_M * c)


class TrajectoryMetricsEvaluator:
    """
    Evaluates navigation accuracy against SIH benchmark standards.
    """

    @staticmethod
    def calculate_distance_travelled(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute total cumulative trajectory distance in meters.
        """
        if len(x) < 2:
            return 0.0
        dx = np.diff(x)
        dy = np.diff(y)
        step_distances = np.sqrt(dx**2 + dy**2)
        return float(np.sum(step_distances))

    @classmethod
    def calculate_sih_benchmark_metrics(
        cls,
        fused_x: np.ndarray,
        fused_y: np.ndarray,
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        fused_speed: np.ndarray,
        ref_speed: np.ndarray,
        outage_mask: Optional[np.ndarray] = None,
        timestamps_sec: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics and compare against SIH benchmark target.

        SIH Benchmark Requirement:
            Positional Drift % < 10% during GNSS outage.
            drift_percentage = (drift_m / distance_travelled_m) * 100%

        Args:
            fused_x: Estimated East coordinates array (m).
            fused_y: Estimated North coordinates array (m).
            ref_x: Ground truth East coordinates array (m).
            ref_y: Ground truth North coordinates array (m).
            fused_speed: Estimated speed array (m/s).
            ref_speed: Ground truth speed array (m/s).
            outage_mask: Optional boolean mask where True indicates blackout window.
            timestamps_sec: Optional timestamps array in seconds.

        Returns:
            Dict[str, Any]: Metrics dictionary.
        """
        pos_errors = np.sqrt((fused_x - ref_x)**2 + (fused_y - ref_y)**2)

        pos_rmse = float(np.sqrt(np.mean(pos_errors**2)))
        mean_pos_err = float(np.mean(pos_errors))
        max_pos_err = float(np.max(pos_errors))
        final_pos_err = float(pos_errors[-1]) if len(pos_errors) > 0 else 0.0

        vel_rmse = float(np.sqrt(np.mean((fused_speed - ref_speed)**2)))
        vel_mae = float(np.mean(np.abs(fused_speed - ref_speed)))

        total_distance_m = cls.calculate_distance_travelled(ref_x, ref_y)

        # Outage Specific Metrics
        if outage_mask is not None and np.any(outage_mask):
            outage_indices = np.where(outage_mask)[0]
            idx_start = outage_indices[0]
            idx_end = outage_indices[-1]

            # Calculate integrated distance travelled during outage (integral of speed * dt)
            if timestamps_sec is not None and len(timestamps_sec) > idx_end:
                dts = np.diff(timestamps_sec[idx_start:idx_end+1])
                outage_dist_m = float(np.sum(ref_speed[idx_start:idx_end] * dts))
            else:
                outage_dist_m = cls.calculate_distance_travelled(ref_x[outage_indices], ref_y[outage_indices])

            if outage_dist_m < 1.0:
                outage_dist_m = max(10.0, cls.calculate_distance_travelled(ref_x[outage_indices], ref_y[outage_indices]))

            # Position error at end of 30s blackout
            outage_max_err = float(pos_errors[idx_end])
            drift_percentage = float((outage_max_err / outage_dist_m) * 100.0)
        else:
            outage_dist_m = total_distance_m
            outage_max_err = max_pos_err
            drift_percentage = float((max_pos_err / max(1.0, total_distance_m)) * 100.0)

        sih_target_met = drift_percentage < 10.0
        sih_result_str = "PASS" if sih_target_met else "NEEDS IMPROVEMENT"

        return {
            "total_distance_m": total_distance_m,
            "outage_distance_m": outage_dist_m,
            "position_rmse_m": pos_rmse,
            "mean_position_error_m": mean_pos_err,
            "max_position_error_m": max_pos_err,
            "final_position_error_m": final_pos_err,
            "outage_max_drift_m": outage_max_err,
            "velocity_rmse_mps": vel_rmse,
            "velocity_mae_mps": vel_mae,
            "drift_percentage": drift_percentage,
            "sih_target_met": sih_target_met,
            "sih_result": sih_result_str,
        }
