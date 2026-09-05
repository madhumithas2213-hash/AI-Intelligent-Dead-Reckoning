"""
Feature Engineering Module.
Derives physical magnitudes, unit conversions, jerk, angular acceleration,
and GPS horizontal displacement features from preprocessed sensor streams.
"""

from typing import Dict, List
import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Computes derived physical features for machine learning models and dead reckoning.
    """

    @staticmethod
    def haversine_distance_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """
        Calculate Great-Circle Haversine distance between coordinate pairs in meters.
        """
        R = 6371000.0  # Earth radius in meters
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return R * c

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived physical magnitude and kinematic features.

        Args:
            df: DataFrame containing clean and filtered sensor channels.

        Returns:
            pd.DataFrame: DataFrame augmented with derived features.
        """
        df_feat = df.copy()

        # 1. Accelerometer Magnitudes
        if "accel_raw_x_ms2" in df_feat.columns:
            ax = pd.to_numeric(df_feat["accel_raw_x_ms2"], errors="coerce").fillna(0.0).to_numpy()
            ay = pd.to_numeric(df_feat["accel_raw_y_ms2"], errors="coerce").fillna(0.0).to_numpy()
            az = pd.to_numeric(df_feat["accel_raw_z_ms2"], errors="coerce").fillna(0.0).to_numpy()
            df_feat["accel_raw_mag_ms2"] = np.sqrt(ax**2 + ay**2 + az**2)

        if "accel_filtered_x_ms2" in df_feat.columns:
            afx = pd.to_numeric(df_feat["accel_filtered_x_ms2"], errors="coerce").fillna(0.0).to_numpy()
            afy = pd.to_numeric(df_feat["accel_filtered_y_ms2"], errors="coerce").fillna(0.0).to_numpy()
            afz = pd.to_numeric(df_feat["accel_filtered_z_ms2"], errors="coerce").fillna(0.0).to_numpy()
            df_feat["accel_filtered_mag_ms2"] = np.sqrt(afx**2 + afy**2 + afz**2)

        if "gravity_x_ms2" in df_feat.columns:
            gx = pd.to_numeric(df_feat["gravity_x_ms2"], errors="coerce").fillna(0.0).to_numpy()
            gy = pd.to_numeric(df_feat["gravity_y_ms2"], errors="coerce").fillna(0.0).to_numpy()
            gz = pd.to_numeric(df_feat["gravity_z_ms2"], errors="coerce").fillna(0.0).to_numpy()
            df_feat["gravity_mag_ms2"] = np.sqrt(gx**2 + gy**2 + gz**2)

        # 2. Gyroscope Magnitude
        if "gyro_raw_yaw_rads" in df_feat.columns:
            gyaw = pd.to_numeric(df_feat["gyro_raw_yaw_rads"], errors="coerce").fillna(0.0).to_numpy()
            gpitch = pd.to_numeric(df_feat["gyro_raw_pitch_rads"], errors="coerce").fillna(0.0).to_numpy()
            groll = pd.to_numeric(df_feat["gyro_raw_roll_rads"], errors="coerce").fillna(0.0).to_numpy()
            df_feat["gyro_raw_mag_rads"] = np.sqrt(gyaw**2 + gpitch**2 + groll**2)

        if "gyro_filtered_yaw_rads" in df_feat.columns:
            gfyaw = pd.to_numeric(df_feat["gyro_filtered_yaw_rads"], errors="coerce").fillna(0.0).to_numpy()
            gfpitch = pd.to_numeric(df_feat["gyro_filtered_pitch_rads"], errors="coerce").fillna(0.0).to_numpy()
            gfroll = pd.to_numeric(df_feat["gyro_filtered_roll_rads"], errors="coerce").fillna(0.0).to_numpy()
            df_feat["gyro_filtered_mag_rads"] = np.sqrt(gfyaw**2 + gfpitch**2 + gfroll**2)

        # 3. Magnetometer Magnitude
        if "mag_x_ut" in df_feat.columns:
            mx = pd.to_numeric(df_feat["mag_x_ut"], errors="coerce").fillna(0.0).to_numpy()
            my = pd.to_numeric(df_feat["mag_y_ut"], errors="coerce").fillna(0.0).to_numpy()
            mz = pd.to_numeric(df_feat["mag_z_ut"], errors="coerce").fillna(0.0).to_numpy()
            df_feat["mag_mag_ut"] = np.sqrt(mx**2 + my**2 + mz**2)

        # 4. GPS Unit Conversions (km/h -> m/s)
        if "gps_speed_kmh" in df_feat.columns:
            sp_kmh = pd.to_numeric(df_feat["gps_speed_kmh"], errors="coerce").fillna(0.0)
            df_feat["gps_speed_mps"] = sp_kmh / 3.6

        # 5. Calculate Jerk & Angular Acceleration
        dt_s = df_feat["dt_ms"].to_numpy() / 1000.0
        dt_s[dt_s < 1e-4] = 0.01  # Prevent division by zero

        if "accel_filtered_mag_ms2" in df_feat.columns:
            a_mag = df_feat["accel_filtered_mag_ms2"].to_numpy()
            jerk = np.zeros_like(a_mag)
            jerk[1:] = np.diff(a_mag) / dt_s[1:]
            df_feat["jerk_ms3"] = jerk

        if "gyro_filtered_mag_rads" in df_feat.columns:
            g_mag = df_feat["gyro_filtered_mag_rads"].to_numpy()
            ang_accel = np.zeros_like(g_mag)
            ang_accel[1:] = np.diff(g_mag) / dt_s[1:]
            df_feat["angular_accel_rads2"] = ang_accel

        # 6. GPS Haversine Speed Verification
        if "gps_latitude_deg" in df_feat.columns and "gps_longitude_deg" in df_feat.columns:
            lats = df_feat["gps_latitude_deg"].to_numpy()
            lons = df_feat["gps_longitude_deg"].to_numpy()
            dist_m = np.zeros_like(lats)
            dist_m[1:] = self.haversine_distance_m(lats[:-1], lons[:-1], lats[1:], lons[1:])
            
            calc_speed_mps = np.zeros_like(lats)
            calc_speed_mps[1:] = dist_m[1:] / dt_s[1:]
            df_feat["calc_haversine_speed_mps"] = calc_speed_mps

        return df_feat
