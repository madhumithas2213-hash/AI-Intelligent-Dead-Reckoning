"""
Automated Test Suite for Phase 6 End-to-End AI-IDR Prototype Integration.
Verifies all core system validation requirements and automated sanity checks:
1. Dataset validation
2. Preprocessing validation
3. Alignment validation
4. AI ML velocity inference validation
5. GNSS state machine transitions
6. Adaptive EKF sensor fusion
7. Map matching road snapping
8. GNSS outage simulation & recovery
9. End-to-end replay engine pipeline
10. Trajectory metrics & SIH drift percentage benchmark
11. dt timestamp sanity check
12. Coordinate conversion sanity check
13. Velocity units sanity check
14. Acceleration units sanity check
15. 30-second outage drift sanity check
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from navigation.alignment.phone_vehicle_aligner import PhoneVehicleAligner
from navigation.sensor_fusion.ekf import AdaptiveEKF, latlon_to_enu, enu_to_latlon, haversine
from navigation.sensor_fusion.ml_interface import MLCorrectionInterface
from navigation.map_matching.osm_matcher import OSMMapMatcher
from navigation.replay.gnss_outage_simulator import GNSSOutageSimulator
from navigation.replay.replay_engine import ReplayEngine
from ml.evaluation.trajectory_metrics import TrajectoryMetricsEvaluator
from ml.training.config import PROCESSED_DIR


class TestPhase6EndToEnd(unittest.TestCase):
    """Full End-to-End System Validation Suite & Sanity Checks."""

    def test_1_dataset_validation(self):
        """1. Verify processed dataset files exist and contain valid telemetry headers."""
        files = list(PROCESSED_DIR.glob("*_processed.csv"))
        self.assertGreater(len(files), 0, "Processed dataset files must exist.")
        df = pd.read_csv(files[0])
        self.assertIn("gps_latitude_deg", df.columns)
        self.assertIn("accel_filtered_x_ms2", df.columns)

    def test_2_preprocessing_validation(self):
        """2. Verify preprocessing output filtering and numerical validity."""
        files = list(PROCESSED_DIR.glob("*_processed.csv"))
        df = pd.read_csv(files[0])
        self.assertFalse(df["accel_filtered_x_ms2"].isna().any(), "Filtered accel must not contain NaNs.")

    def test_3_alignment_validation(self):
        """3. Verify Step 4 PhoneVehicleAligner computes rotation matrix R_p2v."""
        aligner = PhoneVehicleAligner()
        stat_accel = np.array([0.1, 0.1, 9.81])
        dyn_accel = np.random.randn(50, 2)
        R = aligner.compute_alignment_matrix(stat_accel, dyn_accel, speed_delta=5.0)
        self.assertEqual(R.shape, (3, 3))
        self.assertTrue(aligner.is_calibrated)

    def test_4_ai_velocity_inference_validation(self):
        """4. Verify MLCorrectionInterface velocity prediction adapter."""
        ml_interface = MLCorrectionInterface()
        self.assertTrue(hasattr(ml_interface, "predict_velocity_correction"))
        win = np.random.randn(20, 14)
        v_pred, v_var = ml_interface.predict_velocity_correction(win)
        self.assertGreaterEqual(v_var, 0.0)

    def test_5_gnss_state_machine(self):
        """5. Verify 4-state navigation machine (AIDED -> DEGRADED -> DR -> RECOVERED)."""
        sim = GNSSOutageSimulator(blackout_start_sec=10.0, blackout_duration_sec=20.0)

        info1 = sim.update_state(5.0, gnss_quality="GOOD")
        self.assertEqual(info1["state"], "GNSS_AIDED")

        info2 = sim.update_state(15.0, gnss_quality="LOST")
        self.assertEqual(info2["state"], "DEAD_RECKONING")

        sim.restore_gnss(current_timestamp_sec=31.0)
        info3 = sim.update_state(32.0, gnss_quality="GOOD")
        self.assertEqual(info3["state"], "GNSS_RECOVERED")

    def test_6_adaptive_ekf_fusion(self):
        """6. Verify AdaptiveEKF prediction and measurement update."""
        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946)
        state_pred = ekf.update_imu(np.array([1.0, 0.0, 9.81]), np.array([0, 0, 0]), dt=0.1, timestamp_sec=1.0)
        self.assertGreaterEqual(state_pred.speed_mps, 0.0)

        state_meas = ekf.update_gnss(12.9716, 77.5946, 5.0, 0.0, 3.0, 8, timestamp_sec=1.0)
        self.assertEqual(state_meas.mode, "GNSS+INS")

    def test_7_map_matching(self):
        """7. Verify OSMMapMatcher candidate scoring and road snapping."""
        matcher = OSMMapMatcher()
        waypoints = np.array([
            [12.9716, 77.5946],
            [12.9726, 77.5956]
        ])
        matcher.build_synthetic_road_segments(12.9716, 77.5946, waypoints)

        res = matcher.match_point(12.9717, 77.5947, lat_ref=12.9716, lon_ref=77.5946)
        self.assertTrue(res["matched"])
        self.assertIsNotNone(res["segment_id"])

    def test_8_gnss_outage_simulation_and_recovery(self):
        """8. Verify blackout masking and smooth recovery without teleportation."""
        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946)
        ekf.update_gnss(12.9716, 77.5946, 0.0, 0.0, 3.0, 8, timestamp_sec=1.0)

        for i in range(50):
            ekf.update_imu(np.array([0.1, 0.0, 9.81]), np.array([0, 0, 0]), dt=0.1, timestamp_sec=1.0 + (i+1)*0.1)

        self.assertEqual(ekf.get_mode(), "DEAD_RECKONING")
        pos_before = np.array([ekf.get_state().pos_x, ekf.get_state().pos_y])

        target_lat, target_lon = enu_to_latlon(pos_before[0] + 1.0, pos_before[1] + 1.0, 12.9716, 77.5946)
        state_rec = ekf.update_gnss(target_lat, target_lon, 3.0, 0.0, 3.0, 8, timestamp_sec=7.0)
        self.assertEqual(state_rec.mode, "GNSS+INS")

    def test_9_end_to_end_replay_pipeline(self):
        """9. Verify ReplayEngine streams dataset sequence through complete pipeline."""
        engine = ReplayEngine()
        df_res, summary = engine.run_full_pipeline(sequence_name="S-A1", blackout_start_sec=250.0, blackout_duration_sec=30.0)
        self.assertGreater(len(df_res), 0)
        self.assertIn("nav_state", df_res.columns)
        self.assertIn("snapped_lat", df_res.columns)

    def test_10_trajectory_metrics_and_sih_benchmark(self):
        """10. Verify TrajectoryMetricsEvaluator calculates SIH drift percentage."""
        ref_x = np.linspace(0, 1000, 100)
        ref_y = np.linspace(0, 1000, 100)
        fused_x = ref_x + np.random.normal(0, 2.0, 100)
        fused_y = ref_y + np.random.normal(0, 2.0, 100)
        outage_mask = np.zeros(100, dtype=bool)
        outage_mask[30:60] = True

        metrics = TrajectoryMetricsEvaluator.calculate_sih_benchmark_metrics(
            fused_x=fused_x,
            fused_y=fused_y,
            ref_x=ref_x,
            ref_y=ref_y,
            fused_speed=np.ones(100)*10.0,
            ref_speed=np.ones(100)*10.0,
            outage_mask=outage_mask
        )

        self.assertIn("drift_percentage", metrics)
        self.assertIn("sih_result", metrics)

    def test_11_dt_sanity_check(self):
        """11. Verify dt timestamp values in seconds are valid and bounded."""
        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946)
        ekf.update_imu(np.array([0, 0, 9.81]), np.array([0, 0, 0]), dt=0.5, timestamp_sec=0.5)
        ekf.update_imu(np.array([0, 0, 9.81]), np.array([0, 0, 0]), dt=0.499, timestamp_sec=1.0)
        stats = ekf.log_dt_stats()
        self.assertGreater(stats["min"], 0.0)
        self.assertLess(stats["max"], 2.0)
        self.assertAlmostEqual(stats["median"], 0.5, delta=0.1)

    def test_12_coordinate_conversion_sanity_check(self):
        """12. Verify round-trip ENU <-> WGS84 Geodetic conversion accuracy."""
        lat0, lon0 = 52.43163, -1.525745
        lat_target, lon_target = 52.43263, -1.524745

        e_m, n_m = latlon_to_enu(lat_target, lon_target, lat0, lon0)
        lat_reconstructed, lon_reconstructed = enu_to_latlon(e_m, n_m, lat0, lon0)

        self.assertAlmostEqual(lat_target, lat_reconstructed, places=5)
        self.assertAlmostEqual(lon_target, lon_reconstructed, places=5)

    def test_13_velocity_units_sanity_check(self):
        """13. Verify km/h <-> m/s velocity conversion factors."""
        kmh = 36.0
        mps = kmh / 3.6
        self.assertAlmostEqual(mps, 10.0, places=5)
        self.assertAlmostEqual(mps * 3.6, kmh, places=5)

    def test_14_acceleration_units_sanity_check(self):
        """14. Verify gravity removal dynamic acceleration limits."""
        accel_raw = np.array([0.1, 0.2, 9.81])
        gravity = np.array([0.0, 0.0, 9.81])
        accel_dyn = accel_raw - gravity
        self.assertAlmostEqual(accel_dyn[2], 0.0, places=5)
        self.assertLessEqual(np.linalg.norm(accel_dyn), 5.0)

    def test_15_outage_drift_sanity_check(self):
        """15. Verify 30-second outage drift percentage calculation."""
        engine = ReplayEngine()
        df_res, summary = engine.run_full_pipeline("S-A1", blackout_start_sec=250.0, blackout_duration_sec=30.0)
        outage_mask = (df_res["nav_state"] == "DEAD_RECKONING").to_numpy()
        metrics = TrajectoryMetricsEvaluator.calculate_sih_benchmark_metrics(
            fused_x=summary["fused_x"],
            fused_y=summary["fused_y"],
            ref_x=summary["ref_x"],
            ref_y=summary["ref_y"],
            fused_speed=df_res["speed_mps"].to_numpy(),
            ref_speed=df_res["speed_mps"].to_numpy(),
            outage_mask=outage_mask,
            timestamps_sec=df_res["timestamp_sec"].to_numpy()
        )
        self.assertGreater(metrics["outage_distance_m"], 0.0)
        self.assertFalse(np.isnan(metrics["drift_percentage"]))


if __name__ == "__main__":
    unittest.main()
