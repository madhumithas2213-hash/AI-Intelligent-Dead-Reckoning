"""
Navigation State Vector Data Class for Adaptive EKF Sensor Fusion.
Maintains continuous state [pos_x, pos_y, vel_x, vel_y, heading_rad, bias_ax, bias_ay, bias_gz],
geodetic (WGS84 Lat/Lon) representations, navigation mode, covariance, and dynamic sensor confidence score.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class NavigationState:
    """
    Data structure representing vehicle navigation state at a specific timestamp.
    """
    timestamp_sec: float = 0.0
    pos_x: float = 0.0  # Easting in meters (local ENU)
    pos_y: float = 0.0  # Northing in meters (local ENU)
    vel_x: float = 0.0  # East velocity in m/s
    vel_y: float = 0.0  # North velocity in m/s
    heading_rad: float = 0.0  # Yaw angle in radians (0 = North, CW)
    bias_ax: float = 0.0  # Accelerometer X bias
    bias_ay: float = 0.0  # Accelerometer Y bias
    bias_gz: float = 0.0  # Gyroscope Z bias

    latitude: float = 0.0
    longitude: float = 0.0
    speed_mps: float = 0.0
    speed_kmh: float = 0.0

    mode: str = "GNSS+INS"  # GNSS+INS | DEAD_RECKONING
    gnss_quality: str = "GOOD"  # GOOD | DEGRADED | UNRELIABLE | LOST
    confidence_score: float = 95.0  # 0 to 100%
    position_uncertainty_m: float = 3.0  # 1-sigma position error in meters

    covariance: Optional[np.ndarray] = None  # 8x8 P matrix

    def compute_dynamic_confidence(
        self,
        dt: float,
        accel_mag: float,
        gyro_mag: float,
        is_imu_valid: bool = True,
        is_gyro_valid: bool = True
    ) -> float:
        """
        Compute dynamic sensor confidence score (0-100%) from component qualities:
        - IMU Accelerometer validity (25%)
        - Gyroscope validity (25%)
        - Timestamp dt quality (25%)
        - GNSS quality (25%)

        Returns:
            float: Dynamic confidence score in percent (10% to 100%).
        """
        score = 0.0

        # 1. Accelerometer Quality (25%)
        if is_imu_valid and not np.isnan(accel_mag) and (0.0 <= accel_mag <= 30.0):
            score += 25.0

        # 2. Gyroscope Quality (25%)
        if is_gyro_valid and not np.isnan(gyro_mag) and (0.0 <= gyro_mag <= 5.0):
            score += 25.0

        # 3. Timestamp dt Quality (25%)
        if not np.isnan(dt) and (0.001 <= dt <= 1.5):
            score += 25.0

        # 4. GNSS Quality (25%)
        if self.mode == "GNSS+INS":
            if self.gnss_quality == "GOOD":
                score += 25.0
            elif self.gnss_quality == "DEGRADED":
                score += 15.0
            else:
                score += 10.0
        else:
            # DEAD_RECKONING: confidence decays gracefully with time
            score += max(5.0, 15.0 - self.position_uncertainty_m * 0.1)

        self.confidence_score = float(np.clip(score, 10.0, 100.0))
        return self.confidence_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert state instance to serializable dictionary."""
        return {
            "timestamp_sec": float(self.timestamp_sec),
            "pos_x": float(self.pos_x),
            "pos_y": float(self.pos_y),
            "vel_x": float(self.vel_x),
            "vel_y": float(self.vel_y),
            "heading_rad": float(self.heading_rad),
            "heading_deg": float(np.degrees(self.heading_rad) % 360.0),
            "bias_ax": float(self.bias_ax),
            "bias_ay": float(self.bias_ay),
            "bias_gz": float(self.bias_gz),
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "speed_mps": float(self.speed_mps),
            "speed_kmh": float(self.speed_kmh),
            "mode": str(self.mode),
            "gnss_quality": str(self.gnss_quality),
            "confidence_score": float(self.confidence_score),
            "position_uncertainty_m": float(self.position_uncertainty_m),
        }
