"""
Replay Engine Module for Phase 6 End-to-End AI-IDR Prototype.
Streams real IO-VNBD dataset sequences sample-by-sample, executing sensor preprocessing,
phone-to-vehicle alignment, AI velocity prediction, EKF sensor fusion, GNSS outage simulator, and map matching.
"""

from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd

from navigation.alignment.phone_vehicle_aligner import PhoneVehicleAligner
from navigation.sensor_fusion.ekf import AdaptiveEKF, latlon_to_enu, enu_to_latlon, haversine
from navigation.sensor_fusion.gnss_quality import safe_parse_satellites
from navigation.sensor_fusion.ml_interface import MLCorrectionInterface
from navigation.map_matching.osm_matcher import OSMMapMatcher
from navigation.replay.gnss_outage_simulator import GNSSOutageSimulator
from ml.training.config import PROCESSED_DIR


class ReplayEngine:
    """
    End-to-end replay engine for real IO-VNBD trajectory playback and demo visualization.
    """

    def __init__(self, processed_dir: Optional[Path] = None) -> None:
        """
        Args:
            processed_dir: Path to directory containing processed sensor CSVs.
        """
        self.processed_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
        self.aligner = PhoneVehicleAligner()
        self.ml_interface = MLCorrectionInterface()
        self.map_matcher = OSMMapMatcher()
        self.simulator = GNSSOutageSimulator()

        self.current_sequence_name: Optional[str] = None
        self.df_data: Optional[pd.DataFrame] = None
        self.ekf: Optional[AdaptiveEKF] = None

    def load_sequence(self, sequence_name: str = "S-A1") -> pd.DataFrame:
        """
        Load a specific IO-VNBD sequence from dataset/processed.

        Args:
            sequence_name: Sequence filename or prefix (e.g. "S-A1").

        Returns:
            pd.DataFrame: Loaded processed DataFrame.
        """
        target_file = None
        if (self.processed_dir / f"{sequence_name}_processed.csv").exists():
            target_file = self.processed_dir / f"{sequence_name}_processed.csv"
        elif (self.processed_dir / f"{sequence_name}.csv").exists():
            target_file = self.processed_dir / f"{sequence_name}.csv"
        else:
            candidates = list(self.processed_dir.glob(f"*{sequence_name}*.csv"))
            if candidates:
                target_file = candidates[0]

        if not target_file or not target_file.exists():
            all_files = list(self.processed_dir.glob("*_processed.csv"))
            if not all_files:
                raise FileNotFoundError(f"No processed CSV files found in {self.processed_dir}")
            target_file = all_files[0]

        self.current_sequence_name = target_file.stem.replace("_processed", "")
        self.df_data = pd.read_csv(target_file)

        # Calibrate Phone-to-Vehicle Frame Aligner
        if "gravity_x_ms2" in self.df_data.columns:
            ax_dyn = self.df_data["accel_filtered_x_ms2"] - self.df_data["gravity_x_ms2"]
            ay_dyn = self.df_data["accel_filtered_y_ms2"] - self.df_data["gravity_y_ms2"]
            az_dyn = self.df_data["accel_filtered_z_ms2"] - self.df_data["gravity_z_ms2"]
            accel_raw = np.column_stack([ax_dyn, ay_dyn, az_dyn])
        else:
            accel_raw = self.df_data[["accel_filtered_x_ms2", "accel_filtered_y_ms2", "accel_filtered_z_ms2"]].to_numpy()

        gyro_raw = self.df_data[["gyro_filtered_pitch_rads", "gyro_filtered_roll_rads", "gyro_filtered_yaw_rads"]].to_numpy()

        stat_accel = np.mean(accel_raw[:min(50, len(accel_raw))], axis=0)
        dyn_accel = accel_raw[:min(100, len(accel_raw)), :2]
        self.aligner.compute_alignment_matrix(stat_accel, dyn_accel, speed_delta=5.0)

        # Build Map Matcher road graph from reference waypoints
        ref_lats = self.df_data["gps_latitude_deg"].to_numpy()
        ref_lons = self.df_data["gps_longitude_deg"].to_numpy()
        ref_waypoints = np.column_stack([ref_lats, ref_lons])
        self.map_matcher.build_synthetic_road_segments(ref_lats[0], ref_lons[0], ref_waypoints)

        # Initialize EKF
        init_heading_deg = 0.0
        for yaw_col in ["gps_orientation_deg", "gps_orientation_âdeg", "orientation_yaw_deg", "orientation_yaw_âdeg"]:
            if yaw_col in self.df_data.columns:
                init_heading_deg = float(self.df_data[yaw_col].iloc[0])
                break

        self.ekf = AdaptiveEKF(
            lat_ref=ref_lats[0],
            lon_ref=ref_lons[0],
            init_heading_rad=np.radians(init_heading_deg),
            gnss_timeout_seconds=2.0
        )

        return self.df_data

    def run_full_pipeline(
        self,
        sequence_name: str = "S-A1",
        blackout_start_sec: Optional[float] = 250.0,
        blackout_duration_sec: float = 30.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full end-to-end pipeline across a sequence.

        Args:
            sequence_name: Sequence identifier.
            blackout_start_sec: Optional outage start relative timestamp in seconds (default 250s for driving window).
            blackout_duration_sec: Outage duration in seconds.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Pipeline output DataFrame & summary metrics.
        """
        df = self.load_sequence(sequence_name)
        self.simulator = GNSSOutageSimulator(blackout_start_sec, blackout_duration_sec)

        if "gravity_x_ms2" in df.columns:
            ax_dyn = df["accel_filtered_x_ms2"] - df["gravity_x_ms2"]
            ay_dyn = df["accel_filtered_y_ms2"] - df["gravity_y_ms2"]
            az_dyn = df["accel_filtered_z_ms2"] - df["gravity_z_ms2"]
            accel_raw = np.column_stack([ax_dyn, ay_dyn, az_dyn])
        else:
            accel_raw = df[["accel_filtered_x_ms2", "accel_filtered_y_ms2", "accel_filtered_z_ms2"]].to_numpy()

        gyro_raw = df[["gyro_filtered_pitch_rads", "gyro_filtered_roll_rads", "gyro_filtered_yaw_rads"]].to_numpy()

        accel_aligned = self.aligner.transform_to_vehicle_frame(accel_raw)
        gyro_aligned = self.aligner.transform_to_vehicle_frame(gyro_raw)

        if "timestamp_sec" in df.columns:
            timestamps = df["timestamp_sec"].to_numpy()
        elif "time_since_start_ms" in df.columns:
            timestamps = df["time_since_start_ms"].to_numpy() / 1000.0
        else:
            timestamps = np.arange(len(df)) * 0.5

        lats = df["gps_latitude_deg"].to_numpy()
        lons = df["gps_longitude_deg"].to_numpy()
        speeds = df["gps_speed_mps"].to_numpy() if "gps_speed_mps" in df.columns else df["gps_speed_kmh"].to_numpy() / 3.6
        accuracies = df["gps_accuracy_m"].to_numpy() if "gps_accuracy_m" in df.columns else np.full(len(df), 15.0)

        sats_raw = df["gps_satellites_in_range"].to_numpy() if "gps_satellites_in_range" in df.columns else np.full(len(df), 8)
        sats = [safe_parse_satellites(v) for v in sats_raw]

        head_col = None
        for col in ["gps_orientation_deg", "gps_orientation_âdeg", "orientation_yaw_deg", "orientation_yaw_âdeg"]:
            if col in df.columns:
                head_col = col
                break

        t0 = timestamps[0]
        records = []

        for i in range(len(df)):
            t_curr = timestamps[i]
            rel_t = t_curr - t0
            dt = timestamps[i] - timestamps[i-1] if i > 0 else 0.5

            # 1. IMU Prediction
            state = self.ekf.update_imu(accel_aligned[i], gyro_aligned[i], dt, t_curr)

            # 2. Outage State Machine
            outage_info = self.simulator.update_state(rel_t, gnss_quality=state.gnss_quality)
            is_masked = outage_info["is_masked"]

            # 3. GNSS Update (if active)
            if not is_masked:
                gnss_h_rad = np.radians(df[head_col].iloc[i]) if head_col is not None else 0.0
                acc_m = accuracies[i] if (not np.isnan(accuracies[i]) and accuracies[i] > 0) else 15.0
                state = self.ekf.update_gnss(
                    latitude=lats[i],
                    longitude=lons[i],
                    speed_mps=speeds[i],
                    heading_rad=gnss_h_rad,
                    accuracy_m=acc_m,
                    satellites=sats[i],
                    timestamp_sec=t_curr
                )

            # 4. Map Matching
            match_res = self.map_matcher.match_point(
                lat=state.latitude,
                lon=state.longitude,
                heading_rad=state.heading_rad,
                lat_ref=lats[0],
                lon_ref=lons[0]
            )

            record = state.to_dict()
            record["nav_state"] = outage_info["state"]
            record["nav_badge"] = outage_info["badge"]
            record["snapped_lat"] = match_res["snapped_lat"]
            record["snapped_lon"] = match_res["snapped_lon"]
            record["snapped_x"] = match_res.get("snapped_x", state.pos_x)
            record["snapped_y"] = match_res.get("snapped_y", state.pos_y)
            record["map_matched"] = match_res["matched"]
            record["ref_lat"] = lats[i]
            record["ref_lon"] = lons[i]
            record["dt_sec"] = dt
            record["accel_mag"] = float(np.linalg.norm(accel_aligned[i]))
            records.append(record)

        df_results = pd.DataFrame(records)

        # Calculate reference ENU coords
        ref_x, ref_y = [], []
        for i in range(len(df)):
            ex, ny = latlon_to_enu(lats[i], lons[i], lats[0], lons[0])
            ref_x.append(ex)
            ref_y.append(ny)

        summary_metrics = {
            "sequence": self.current_sequence_name,
            "samples": len(df),
            "duration_sec": float(timestamps[-1] - t0),
            "ref_x": np.array(ref_x),
            "ref_y": np.array(ref_y),
            "fused_x": df_results["pos_x"].to_numpy(),
            "fused_y": df_results["pos_y"].to_numpy(),
            "dt_stats": self.ekf.log_dt_stats(),
        }

        return df_results, summary_metrics
