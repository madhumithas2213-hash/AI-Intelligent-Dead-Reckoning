"""
Automatic Phone-to-Vehicle Frame Alignment Module.
Estimates rotation matrix R_p2v that transforms smartphone body frame readings
to the vehicle body frame (Forward-Right-Down / Forward-Left-Up).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class PhoneVehicleAligner:
    """
    Estimates phone-to-vehicle relative attitude using gravity vector decomposition
    during stationary phases and vehicle acceleration direction during motion.
    """

    def __init__(self) -> None:
        self.is_calibrated: bool = False
        self.R_phone_to_vehicle: np.ndarray = np.eye(3)

    def estimate_pitch_roll_from_gravity(self, accel_mean: np.ndarray) -> Tuple[float, float]:
        """
        Estimate pitch and roll angles from mean stationary accelerometer readings.

        Args:
            accel_mean: 3D acceleration vector [ax, ay, az] in m/s^2.

        Returns:
            Tuple[float, float]: (pitch_rad, roll_rad).
        """
        ax, ay, az = accel_mean
        norm = np.linalg.norm(accel_mean)
        if norm < 1e-3:
            return 0.0, 0.0

        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        return float(pitch), float(roll)

    def estimate_yaw_from_acceleration(self, accel_window: np.ndarray, speed_diff: float) -> float:
        """
        Estimate yaw misalignment angle between phone forward axis and vehicle longitudinal axis
        during vehicle acceleration / deceleration phases.

        Args:
            accel_window: 2D horizontal phone acceleration array [N, 2].
            speed_diff: Vehicle forward speed delta over the window.

        Returns:
            float: Yaw misalignment angle in radians.
        """
        if abs(speed_diff) < 0.5:
            return 0.0  # Insufficient acceleration signal

        mean_horiz_accel = np.mean(accel_window, axis=0)
        yaw_offset = np.arctan2(mean_horiz_accel[1], mean_horiz_accel[0])
        return float(yaw_offset)

    def compute_alignment_matrix(
        self,
        stationary_accel: np.ndarray,
        dynamic_accel_window: np.ndarray,
        speed_delta: float
    ) -> np.ndarray:
        """
        Compute full 3D rotation matrix transforming phone coordinates to vehicle frame.

        Args:
            stationary_accel: Mean accelerometer vector when stationary [3].
            dynamic_accel_window: Horizontal phone acceleration during linear motion [N, 2].
            speed_delta: Forward speed change in m/s.

        Returns:
            np.ndarray: 3x3 Orthogonal rotation matrix R_p2v.
        """
        pitch, roll = self.estimate_pitch_roll_from_gravity(stationary_accel)
        yaw = self.estimate_yaw_from_acceleration(dynamic_accel_window, speed_delta)

        rotation = R.from_euler('zyx', [yaw, pitch, roll])
        self.R_phone_to_vehicle = rotation.as_matrix()
        self.is_calibrated = True
        return self.R_phone_to_vehicle

    def transform_to_vehicle_frame(self, phone_sensor_data: np.ndarray) -> np.ndarray:
        """
        Rotate 3D sensor vectors from phone frame into vehicle body frame.

        Args:
            phone_sensor_data: Array of shape [N, 3] or [3].

        Returns:
            np.ndarray: Rotated sensor data in vehicle body frame [N, 3].
        """
        if not self.is_calibrated:
            print("[Warning] PhoneVehicleAligner not calibrated. Returning unaligned data.")
            return phone_sensor_data

        if phone_sensor_data.ndim == 1:
            return self.R_phone_to_vehicle @ phone_sensor_data
        else:
            return (self.R_phone_to_vehicle @ phone_sensor_data.T).T
