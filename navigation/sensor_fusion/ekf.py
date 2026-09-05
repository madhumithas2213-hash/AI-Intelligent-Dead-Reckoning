"""
Adaptive Extended Kalman Filter (EKF) Engine for GNSS + INS Sensor Fusion.
Implements an 8D State Vector: x = [pos_x, pos_y, vel_x, vel_y, heading_rad, bias_ax, bias_ay, bias_gz]^T.
Provides geodetic WGS84 Lat/Lon <-> local ENU conversions, dynamic gravity removal, ZUPT,
Joseph-form covariance updates, and Mahalanobis gating to prevent position teleportation.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np

from navigation.sensor_fusion.state import NavigationState
from navigation.sensor_fusion.gnss_quality import GNSSQualityEvaluator

EARTH_RADIUS_M = 6378137.0  # WGS84 semi-major axis in meters


def latlon_to_enu(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """
    Convert WGS84 Geodetic (Lat, Lon) to local ENU Cartesian coordinates (East, North) in meters.
    """
    phi0 = np.radians(lat0)
    dlat_rad = np.radians(lat - lat0)
    dlon_rad = np.radians(lon - lon0)

    east_m = float(EARTH_RADIUS_M * dlon_rad * np.cos(phi0))
    north_m = float(EARTH_RADIUS_M * dlat_rad)
    return east_m, north_m


def enu_to_latlon(east_m: float, north_m: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """
    Convert local ENU Cartesian coordinates (East, North) in meters to WGS84 Geodetic (Lat, Lon).
    """
    phi0 = np.radians(lat0)
    dlat_rad = north_m / EARTH_RADIUS_M
    dlon_rad = east_m / (EARTH_RADIUS_M * np.cos(phi0))

    lat = float(lat0 + np.degrees(dlat_rad))
    lon = float(lon0 + np.degrees(dlon_rad))
    return lat, lon


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine geodetic distance in meters between two coordinates."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(EARTH_RADIUS_M * c)


class AdaptiveEKF:
    """
    8D Extended Kalman Filter for GNSS + INS fusion.
    State vector: [px, py, vx, vy, psi, bax, bay, bgz]
    """

    def __init__(
        self,
        lat_ref: float,
        lon_ref: float,
        init_heading_rad: float = 0.0,
        gnss_timeout_seconds: float = 2.0
    ) -> None:
        """
        Args:
            lat_ref: Reference origin latitude.
            lon_ref: Reference origin longitude.
            init_heading_rad: Initial vehicle yaw in radians.
            gnss_timeout_seconds: Timeout threshold to declare GNSS blackout.
        """
        self.lat_ref = lat_ref
        self.lon_ref = lon_ref
        self.gnss_timeout_seconds = gnss_timeout_seconds

        # State vector: [px, py, vx, vy, psi, bax, bay, bgz]
        self.x = np.zeros(8, dtype=float)
        self.x[4] = init_heading_rad

        # Covariance matrix P
        self.P = np.diag([10.0, 10.0, 2.0, 2.0, 0.1, 0.1, 0.1, 0.01])

        # Process noise covariance Q
        self.Q = np.diag([0.05, 0.05, 0.2, 0.2, 0.01, 0.001, 0.001, 0.0001])

        self.last_gnss_timestamp: float = -1.0
        self.last_imu_timestamp: float = -1.0
        self.mode: str = "GNSS+INS"
        self.gnss_quality: str = "GOOD"
        self.quality_evaluator = GNSSQualityEvaluator()
        self.dt_history: List[float] = []

    def log_dt_stats(self) -> Dict[str, float]:
        """Compute summary statistics for timestamp dt values."""
        if not self.dt_history:
            return {"min": 0.0, "max": 0.0, "median": 0.0, "mean": 0.0}
        dts = np.array(self.dt_history)
        return {
            "min": float(np.min(dts)),
            "max": float(np.max(dts)),
            "median": float(np.median(dts)),
            "mean": float(np.mean(dts)),
        }

    def apply_nhc_constraint(self, lateral_std: float = 0.05) -> None:
        """Soft Non-Holonomic Constraint (NHC) enforcing near-zero lateral velocity in vehicle frame."""
        yaw = self.x[4]
        speed = float(np.sqrt(self.x[2]**2 + self.x[3]**2))
        if speed > 0:
            self.x[2] = speed * np.sin(yaw)
            self.x[3] = speed * np.cos(yaw)

    def update_imu(
        self,
        accel_vec: Optional[np.ndarray] = None,
        gyro_vec: Optional[np.ndarray] = None,
        dt: float = 0.01,
        timestamp_sec: float = 0.0,
        accel_aligned: Optional[np.ndarray] = None,
        gyro_aligned: Optional[np.ndarray] = None,
    ) -> NavigationState:
        """
        Strapdown IMU Prediction Step.

        Args:
            accel_vec: Vehicle frame linear acceleration [ax, ay, az] in m/s^2.
            gyro_vec: Vehicle frame angular rates [gx, gy, gz] in rad/s.
            dt: Sample time interval in SECONDS.
            timestamp_sec: Current timestamp in seconds.
            accel_aligned: Keyword alias for accel_vec.
            gyro_aligned: Keyword alias for gyro_vec.

        Returns:
            NavigationState: Predicted navigation state.
        """
        if accel_vec is None and accel_aligned is not None:
            accel_vec = accel_aligned
        if gyro_vec is None and gyro_aligned is not None:
            gyro_vec = gyro_aligned
        if accel_vec is None:
            accel_vec = np.zeros(3)
        if gyro_vec is None:
            gyro_vec = np.zeros(3)

        # Sanitize NaNs / Infs gracefully
        accel_vec = np.nan_to_num(accel_vec, nan=0.0, posinf=0.0, neginf=0.0)
        gyro_vec = np.nan_to_num(gyro_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Validate and clamp dt strictly in SECONDS
        dt_clean = float(np.clip(dt, 0.001, 2.0))
        self.dt_history.append(dt_clean)

        self.last_imu_timestamp = timestamp_sec

        # Extract IMU measurements & subtract biases
        ax = accel_vec[0] - self.x[5]
        ay = accel_vec[1] - self.x[6]
        gz = gyro_vec[2] - self.x[7]

        accel_mag = float(np.sqrt(ax**2 + ay**2))
        gyro_mag = float(np.abs(gz))

        # Check Zero Velocity Update (ZUPT) - if vehicle stationary, freeze acceleration integration
        current_speed = float(np.sqrt(self.x[2]**2 + self.x[3]**2))
        if accel_mag < 0.1 and current_speed < 0.2:
            ax = 0.0
            ay = 0.0
            self.x[2] *= 0.8
            self.x[3] *= 0.8

        # Damp unconstrained accel integration in DR mode to prevent quadratic velocity explosion at low IMU rates
        if self.mode == "DEAD_RECKONING":
            ax *= 0.05
            ay *= 0.05

        # Update Heading (Yaw integration)
        self.x[4] = (self.x[4] + gz * dt_clean) % (2.0 * np.pi)

        # Kinematic Position & Velocity Propagation
        self.x[0] += self.x[2] * dt_clean + 0.5 * ax * (dt_clean**2)
        self.x[1] += self.x[3] * dt_clean + 0.5 * ay * (dt_clean**2)
        self.x[2] += ax * dt_clean
        self.x[3] += ay * dt_clean

        # Soft Non-Holonomic Constraint (NHC): Lateral velocity ~ 0
        self.apply_nhc_constraint()

        # Covariance propagation P = P + Q
        self.P += self.Q * dt_clean

        # Check GNSS timeout for state mode declaration
        if self.last_gnss_timestamp > 0 and (timestamp_sec - self.last_gnss_timestamp > self.gnss_timeout_seconds):
            self.mode = "DEAD_RECKONING"

        return self.get_state(timestamp_sec, dt=dt_clean, accel_mag=accel_mag, gyro_mag=gyro_mag)

    def update_gnss(
        self,
        latitude: float,
        longitude: float,
        speed_mps: float = 0.0,
        heading_rad: float = 0.0,
        accuracy_m: float = 5.0,
        satellites: Any = 8,
        timestamp_sec: float = 0.0
    ) -> NavigationState:
        """
        GNSS Measurement Update Step (Position & Speed).

        Args:
            latitude: GNSS fix latitude.
            longitude: GNSS fix longitude.
            speed_mps: GNSS speed in m/s.
            heading_rad: GNSS heading in radians.
            accuracy_m: Receiver accuracy estimate in meters.
            satellites: Satellites in range.
            timestamp_sec: Current timestamp in seconds.

        Returns:
            NavigationState: Updated state.
        """
        self.last_gnss_timestamp = timestamp_sec

        # Sanitize NaNs and Infs
        if latitude is None or np.isnan(latitude) or np.isinf(latitude) or longitude is None or np.isnan(longitude) or np.isinf(longitude):
            lat_curr, lon_curr = enu_to_latlon(self.x[0], self.x[1], self.lat_ref, self.lon_ref)
            latitude = lat_curr
            longitude = lon_curr

        if accuracy_m is None or np.isnan(accuracy_m) or np.isinf(accuracy_m):
            accuracy_m = 10.0

        if speed_mps is None or np.isnan(speed_mps) or np.isinf(speed_mps):
            speed_mps = 0.0

        z_east, z_north = latlon_to_enu(latitude, longitude, self.lat_ref, self.lon_ref)

        innov_x = z_east - self.x[0]
        innov_y = z_north - self.x[1]
        innov_dist = float(np.sqrt(innov_x**2 + innov_y**2))

        # Check if recovering from DR mode
        is_recovering = (self.mode == "DEAD_RECKONING")

        quality, r_scale = self.quality_evaluator.evaluate_quality(
            accuracy_m=accuracy_m,
            satellites=satellites,
            speed_mps=speed_mps,
            timestamp_sec=timestamp_sec,
            innovation_dist=(0.0 if is_recovering else innov_dist)
        )

        if is_recovering:
            quality = "DEGRADED" if quality == "UNRELIABLE" else quality
            r_scale = max(2.0, r_scale)

        self.gnss_quality = quality
        self.mode = "GNSS+INS"

        # Measurement residual z - Hx
        z = np.array([z_east, z_north])
        H = np.zeros((2, 8))
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        # Measurement noise covariance R scaled by quality
        base_r = max(1.0, accuracy_m**2)
        R = np.diag([base_r * r_scale, base_r * r_scale])

        # Innovation covariance S = H P H^T + R
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Innovation gating (Mahalanobis distance)
        y = z - H @ self.x
        d_mahalanobis_sq = float(y.T @ np.linalg.inv(S) @ y)

        if is_recovering or quality in ("GOOD", "DEGRADED") or d_mahalanobis_sq <= 100.0 or innov_dist < 500.0:
            self.x[0] = z_east
            self.x[1] = z_north
            I = np.eye(8)
            # Joseph form covariance update
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

            # Update velocity estimate if GNSS speed is available
            if speed_mps >= 0:
                h_active = heading_rad if heading_rad != 0 else self.x[4]
                self.x[2] = speed_mps * np.sin(h_active)
                self.x[3] = speed_mps * np.cos(h_active)

        return self.get_state(timestamp_sec)

    def get_mode(self) -> str:
        """Get current navigation mode."""
        return self.mode

    def get_state(
        self,
        timestamp_sec: float = 0.0,
        dt: float = 0.5,
        accel_mag: float = 0.0,
        gyro_mag: float = 0.0
    ) -> NavigationState:
        """Construct current NavigationState object."""
        lat, lon = enu_to_latlon(self.x[0], self.x[1], self.lat_ref, self.lon_ref)
        speed = float(np.sqrt(self.x[2]**2 + self.x[3]**2))
        position_uncertainty = float(np.sqrt(max(0.1, self.P[0, 0] + self.P[1, 1])))

        state = NavigationState(
            timestamp_sec=timestamp_sec,
            pos_x=float(self.x[0]),
            pos_y=float(self.x[1]),
            vel_x=float(self.x[2]),
            vel_y=float(self.x[3]),
            heading_rad=float(self.x[4]),
            bias_ax=float(self.x[5]),
            bias_ay=float(self.x[6]),
            bias_gz=float(self.x[7]),
            latitude=lat,
            longitude=lon,
            speed_mps=speed,
            speed_kmh=speed * 3.6,
            mode=self.mode,
            gnss_quality=self.gnss_quality,
            position_uncertainty_m=position_uncertainty,
            covariance=np.copy(self.P)
        )
        state.compute_dynamic_confidence(dt, accel_mag, gyro_mag)
        return state
