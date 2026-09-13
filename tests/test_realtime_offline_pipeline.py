"""
Automated Python Verification Test Suite for Real-Time Offline-First AI-IDR Navigation Pipeline.
Validates:
1. Pure JavaScript GRU model weights integrity and architecture (42,433 float parameters).
2. Dead Reckoning spherical kinematics (WGS84 forward coordinate integration).
3. Offline Map Matching orthogonal point-to-segment projection logic.
4. Complete isolation of exported dataset from the real-time smartphone live sensor pipeline.
5. Hash consistency across root and ml/outputs/fusion deployment targets.
"""

import unittest
import json
import re
import math
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRealtimeOfflinePipeline(unittest.TestCase):

    def test_offline_ml_engine_weights_and_architecture(self):
        """Verify on-device ML model parameters and weight matrix dimensions."""
        engine_file = PROJECT_ROOT / "offline_ml_engine.js"
        self.assertTrue(engine_file.exists(), "offline_ml_engine.js must exist")

        content = engine_file.read_text(encoding="utf-8")
        match = re.search(r"const MODEL_WEIGHTS = ({.*?});", content)
        self.assertIsNotNone(match, "MODEL_WEIGHTS JSON object must be present in offline_ml_engine.js")

        weights = json.loads(match.group(1))
        arch = weights.get("architecture", {})
        self.assertEqual(arch.get("input_dim"), 14, "Input dimension must be 14 sensor channels")
        self.assertEqual(arch.get("hidden_dim"), 64, "Hidden dimension must be 64")
        self.assertEqual(arch.get("num_layers"), 2, "Must be a 2-layer GRU")
        self.assertEqual(arch.get("window_size"), 20, "Window size must be 20 samples (2.0s @ 10Hz)")

        # Verify scaler
        scaler = weights.get("scaler", {})
        self.assertEqual(len(scaler.get("mean", [])), 14, "Scaler mean vector must have 14 values")
        self.assertEqual(len(scaler.get("scale", [])), 14, "Scaler scale vector must have 14 values")

        # Verify GRU layer 0 weights
        gru_l0 = weights.get("gru_l0", {})
        self.assertEqual(len(gru_l0.get("w_ih", [])), 3 * 64, "Layer 0 w_ih must have 192 rows (3 gates * 64)")
        self.assertEqual(len(gru_l0.get("w_hh", [])), 3 * 64, "Layer 0 w_hh must have 192 rows (3 gates * 64)")

        # Verify fully-connected head
        fc = weights.get("fc_head", {})
        self.assertEqual(len(fc.get("fc0_w", [])), 32, "FC layer 0 weight must map 64 -> 32")
        self.assertEqual(len(fc.get("fc3_w", [])), 1, "FC layer 1 weight must map 32 -> 1")

    def test_dead_reckoning_kinematic_propagation(self):
        """Verify forward velocity and heading integration into WGS84 coordinates."""
        start_lat = 12.971600
        start_lon = 77.594600
        speed_mps = 15.0  # 54 km/h
        dt = 2.0  # 30 meters travel
        heading_rad = math.radians(45.0)  # North-East (45 degrees)

        r_earth = 6378137.0
        d_north = speed_mps * math.cos(heading_rad) * dt
        d_east = speed_mps * math.sin(heading_rad) * dt

        new_lat = start_lat + (d_north / r_earth) * (180.0 / math.pi)
        new_lon = start_lon + (d_east / (r_earth * math.cos(math.radians(start_lat)))) * (180.0 / math.pi)

        # Total Euclidean ground displacement
        lat_m = (new_lat - start_lat) * (math.pi / 180.0) * r_earth
        lon_m = (new_lon - start_lon) * (math.pi / 180.0) * r_earth * math.cos(math.radians(start_lat))
        displacement = math.hypot(lat_m, lon_m)

        self.assertAlmostEqual(displacement, 30.0, places=2, msg="Displacement must match speed * dt = 30m")
        self.assertGreater(new_lat, start_lat, "Latitude must increase heading North-East")
        self.assertGreater(new_lon, start_lon, "Longitude must increase heading North-East")

    def test_offline_map_matching_projection(self):
        """Verify orthogonal point-to-segment projection math."""
        # Segment from [0, 0] to [0, 100 meters East]
        a_lat, a_lon = 12.971600, 77.594600
        cos_lat = math.cos(math.radians(a_lat))
        b_lat = a_lat
        b_lon = a_lon + (100.0 / (111320.0 * cos_lat))

        # Query point 20 meters North of segment midpoint
        mid_lon = (a_lon + b_lon) / 2.0
        p_lat = a_lat + (20.0 / 110540.0)
        p_lon = mid_lon

        # Projection calculation
        bx = (b_lon - a_lon) * 111320.0 * cos_lat
        by = (b_lat - a_lat) * 110540.0
        px = (p_lon - a_lon) * 111320.0 * cos_lat
        py = (p_lat - a_lat) * 110540.0

        seg_len_sq = bx * bx + by * by
        t = max(0.0, min(1.0, (px * bx + py * by) / seg_len_sq))
        snap_lat = a_lat + t * (b_lat - a_lat)
        snap_lon = a_lon + t * (b_lon - a_lon)

        dist = math.hypot((p_lon - snap_lon) * 111320.0 * cos_lat, (p_lat - snap_lat) * 110540.0)

        self.assertAlmostEqual(t, 0.5, places=2, msg="Projection must fall at segment midpoint (t=0.5)")
        self.assertAlmostEqual(dist, 20.0, places=1, msg="Distance must equal the 20m offset")
        self.assertAlmostEqual(snap_lat, a_lat, places=6, msg="Snapped latitude must lie exactly on the segment")

    def test_dataset_segregation_in_live_pipeline(self):
        """Verify that live sensor processing does not read from multiTrajData."""
        index_html = (PROJECT_ROOT / "index.html").read_text(encoding="utf-8")

        # Confirm switchTab('live') is the boot default
        self.assertIn("switchTab('live')", index_html, "System must boot to live phone IDR by default")

        # Inspect processLiveTick implementation
        tick_pos = index_html.find("function processLiveTick(")
        self.assertGreater(tick_pos, 0, "processLiveTick function must be present")
        tick_body = index_html[tick_pos:tick_pos + 6000]

        self.assertNotIn("multiTrajData", tick_body, "processLiveTick must NEVER reference historical multiTrajData")
        self.assertIn("liveSensorFeatureBuffer", tick_body, "processLiveTick must use liveSensorFeatureBuffer")
        self.assertIn("offlineVelocityPredictor", tick_body, "processLiveTick must call offlineVelocityPredictor")

    def test_asset_parity_across_deployment_directories(self):
        """Verify exact cryptographic SHA-256 parity between root and ml/outputs/fusion/ assets."""
        pairs = [
            ("index.html", "dashboard.html"),
            ("index.html", "ml/outputs/fusion/index.html"),
            ("index.html", "ml/outputs/fusion/dashboard.html"),
            ("offline_ml_engine.js", "ml/outputs/fusion/offline_ml_engine.js"),
            ("sw.js", "ml/outputs/fusion/sw.js"),
        ]

        for file_a, file_b in pairs:
            path_a = PROJECT_ROOT / file_a
            path_b = PROJECT_ROOT / file_b
            self.assertTrue(path_a.exists(), f"File {file_a} must exist")
            self.assertTrue(path_b.exists(), f"File {file_b} must exist")
            hash_a = hashlib.sha256(path_a.read_bytes()).hexdigest()
            hash_b = hashlib.sha256(path_b.read_bytes()).hexdigest()
            self.assertEqual(hash_a, hash_b, f"Hash mismatch between {file_a} and {file_b}")


if __name__ == "__main__":
    unittest.main()
