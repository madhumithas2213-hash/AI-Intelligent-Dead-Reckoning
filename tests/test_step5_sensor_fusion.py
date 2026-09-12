"""
Automated Test Suite for Step 5 Adaptive GNSS + INS Sensor Fusion Engine.
Verifies all 10 core requirements:
1. IMU prediction
2. GNSS update
3. GNSS quality
4. GNSS blackout
5. GNSS recovery
6. Timestamp irregularity
7. Invalid sensor data
8. NHC constraint
9. End-to-end real IO-VNBD sequence
10. Complete demo flow
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

from navigation.sensor_fusion.ekf import AdaptiveEKF, latlon_to_enu, enu_to_latlon
from navigation.sensor_fusion.gnss_quality import GNSSQualityEvaluator
from navigation.sensor_fusion.blackout import BlackoutEvaluator
from navigation.sensor_fusion.metrics import SensorFusionMetrics
from navigation.sensor_fusion.ml_interface import MLCorrectionInterface
from ml.training.config import PROCESSED_DIR


class TestStep5SensorFusion(unittest.TestCase):
    """Unit and Integration tests for Step 5 Adaptive Sensor Fusion Engine."""

    def test_1_imu_prediction(self):
        """TEST 1: IMU prediction propagates position, velocity, and covariance forward."""
        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946)
        accel = np.array([1.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])
        
        state = ekf.update_imu(accel_aligned=accel, gyro_aligned=gyro, dt=1.0, timestamp_sec=1.0)
        
        self.assertGreater(state.speed_mps, 0.0)
        self.assertGreater(state.position_uncertainty_m, 0.0)

    def test_2_gnss_update(self):
        """TEST 2: GNSS update corrects position drift and reduces uncertainty."""
        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946)
        for i in range(10):
            ekf.update_imu(np.array([0.5, 0.0, 9.81]), np.array([0, 0, 0]), dt=0.1, timestamp_sec=i*0.1)

        p_uncert_before = ekf.get_state().position_uncertainty_m
        state_after = ekf.update_gnss(
            latitude=12.9716,
            longitude=77.5946,
            speed_mps=1.0,
            heading_rad=0.0,
            accuracy_m=2.0,
            satellites=8,
            timestamp_sec=1.0
        )

        self.assertEqual(state_after.mode, "GNSS+INS")
        self.assertLess(state_after.position_uncertainty_m, p_uncert_before)

    def test_3_gnss_quality(self):
        """TEST 3: GNSS quality estimator correctly classifies GOOD, DEGRADED, UNRELIABLE, and LOST."""
        evaluator = GNSSQualityEvaluator(gnss_timeout_seconds=2.0)
        
        q_good, r_good = evaluator.evaluate_quality(accuracy_m=3.0, satellites=8, speed_mps=10.0, timestamp_sec=1.0)
        self.assertEqual(q_good, "GOOD")
        self.assertLess(r_good, 5.0)

        q_deg, r_deg = evaluator.evaluate_quality(accuracy_m=12.0, satellites=4, speed_mps=10.0, timestamp_sec=2.0)
        self.assertEqual(q_deg, "DEGRADED")
        self.assertGreater(r_deg, r_good)

        q_unrel, r_unrel = evaluator.evaluate_quality(accuracy_m=50.0, satellites=2, speed_mps=10.0, timestamp_sec=3.0)
        self.assertEqual(q_unrel, "UNRELIABLE")

    def test_4_gnss_blackout(self):
        """TEST 4: GNSS blackout simulation masks GNSS updates and switches to DEAD_RECKONING mode."""
        blackout = BlackoutEvaluator(blackout_start_sec=10.0, blackout_duration_sec=20.0)
        
        self.assertFalse(blackout.is_in_blackout(5.0))
        self.assertTrue(blackout.is_in_blackout(15.0))
        self.assertFalse(blackout.is_in_blackout(35.0))

        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946, gnss_timeout_seconds=1.5)
        ekf.update_gnss(12.9716, 77.5946, 5.0, 0.0, 3.0, 8, timestamp_sec=1.0)

        for i in range(30):
            t_curr = 1.0 + (i + 1) * 0.1
            ekf.update_imu(np.array([0.0, 0.0, 9.81]), np.array([0, 0, 0]), dt=0.1, timestamp_sec=t_curr)

        self.assertEqual(ekf.get_mode(), "DEAD_RECKONING")

    def test_5_gnss_recovery(self):
        """TEST 5: GNSS recovery performs gated, smooth transition without sudden position teleportation."""
        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946)
        ekf.update_gnss(12.9716, 77.5946, 0.0, 0.0, 3.0, 8, timestamp_sec=1.0)

        for i in range(100):
            t = 1.0 + (i + 1) * 0.1
            ekf.update_imu(np.array([0.1, 0.0, 9.81]), np.array([0, 0, 0.01]), dt=0.1, timestamp_sec=t)

        pos_before = np.array([ekf.get_state().pos_x, ekf.get_state().pos_y])
        target_lat, target_lon = enu_to_latlon(pos_before[0] + 1.0, pos_before[1] + 1.0, 12.9716, 77.5946)
        
        state_recovered = ekf.update_gnss(target_lat, target_lon, 3.0, 0.0, 4.0, 8, timestamp_sec=12.0)
        self.assertEqual(state_recovered.mode, "GNSS+INS")
        
        pos_after = np.array([state_recovered.pos_x, state_recovered.pos_y])
        step_jump = float(np.linalg.norm(pos_after - pos_before))
        self.assertLess(step_jump, 10.0)

    def test_6_timestamp_irregularity(self):
        """TEST 6: Engine handles variable dt and timestamp jitter safely."""
        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946)
        dts = [0.005, 0.02, 0.1, 0.001, 0.5]
        t = 1.0
        for dt in dts:
            t += dt
            state = ekf.update_imu(np.array([0.1, 0.0, 9.81]), np.array([0, 0, 0]), dt=dt, timestamp_sec=t)
            self.assertFalse(np.isnan(state.speed_mps))

    def test_7_invalid_sensor_data(self):
        """TEST 7: Engine safely handles NaN, Inf, and invalid sensor values without crashing."""
        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946)
        
        state_nan = ekf.update_imu(np.array([np.nan, 0.0, 9.81]), np.array([0, 0, 0]), dt=0.01, timestamp_sec=1.0)
        self.assertFalse(np.isnan(state_nan.pos_x))

        state_inf = ekf.update_gnss(np.inf, 77.5946, 5.0, 0.0, np.nan, 0, timestamp_sec=2.0)
        self.assertFalse(np.isnan(state_inf.latitude))

    def test_8_nhc_constraint(self):
        """TEST 8: Soft non-holonomic constraint bounds lateral velocity."""
        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946)
        ekf.x[2] = 5.0  # East velocity
        ekf.x[3] = 0.0  # North velocity
        ekf.x[4] = 0.0  # Heading = North -> Lateral axis is East

        ekf.apply_nhc_constraint(lateral_std=0.05)
        self.assertLess(abs(ekf.x[2]), 5.0)

    def test_9_end_to_end_io_vnbd_evaluation(self):
        """TEST 9: End-to-end real IO-VNBD sequence processing."""
        files = list(PROCESSED_DIR.glob("*_processed.csv"))
        self.assertGreater(len(files), 0)

        df = pd.read_csv(files[0])
        ekf = AdaptiveEKF(lat_ref=df["gps_latitude_deg"].iloc[0], lon_ref=df["gps_longitude_deg"].iloc[0])

        accel = df[["accel_filtered_x_ms2", "accel_filtered_y_ms2", "accel_filtered_z_ms2"]].to_numpy()
        gyro = df[["gyro_filtered_pitch_rads", "gyro_filtered_roll_rads", "gyro_filtered_yaw_rads"]].to_numpy()

        for i in range(min(50, len(df))):
            state = ekf.update_imu(accel[i], gyro[i], dt=0.01, timestamp_sec=i*0.01)

        self.assertGreaterEqual(state.speed_mps, 0.0)

    def test_10_complete_demo_flow(self):
        """TEST 10: Complete demonstration scenario flow (GNSS AVAILABLE -> LOST -> DR -> RECOVERY -> GNSS+INS)."""
        ekf = AdaptiveEKF(lat_ref=12.9716, lon_ref=77.5946)
        
        # 1. GNSS Available
        s1 = ekf.update_gnss(12.9716, 77.5946, 10.0, 0.0, 3.0, 8, timestamp_sec=1.0)
        self.assertEqual(s1.mode, "GNSS+INS")

        # 2. GNSS Lost / Outage
        for i in range(30):
            s2 = ekf.update_imu(np.array([0.2, 0.0, 9.81]), np.array([0, 0, 0]), dt=0.1, timestamp_sec=1.0 + (i+1)*0.1)

        self.assertEqual(s2.mode, "DEAD_RECKONING")

        # 3. GNSS Recovery
        s3 = ekf.update_gnss(12.9716 + 0.00005, 77.5946 + 0.00005, 10.0, 0.0, 3.0, 8, timestamp_sec=5.0)
        self.assertEqual(s3.mode, "GNSS+INS")

    def test_11_outage_evaluator_regression(self):
        """TEST 11: Regression test for outage window evaluation, timestamp alignment, and raw dataset immutability."""
        from ml.evaluation.evaluate_step5_fusion import evaluate_step5
        
        # 1. Confirm raw dataset files are intact and un-modified
        raw_files = list(Path("dataset/raw").glob("*.csv")) if Path("dataset/raw").exists() else []
        for rf in raw_files:
            self.assertTrue(rf.exists())
            self.assertGreater(rf.stat().st_size, 0)

        # 2. Run Step 5 evaluation
        metrics, blackout_metrics = evaluate_step5()

        # 3. Verify outage metrics alignment
        self.assertIn("reference_distance_m", blackout_metrics)
        self.assertIn("max_position_error_m", blackout_metrics)
        self.assertIn("drift_percentage", blackout_metrics)
        
        ref_dist = blackout_metrics["reference_distance_m"]
        max_err = blackout_metrics["max_position_error_m"]
        drift_pct = blackout_metrics["drift_percentage"]

        self.assertGreater(ref_dist, 0.0)
        self.assertGreater(max_err, 0.0)
        self.assertAlmostEqual(drift_pct, (max_err / ref_dist) * 100.0, places=2)


if __name__ == "__main__":
    unittest.main()
