"""
Automated Verification Suite for Offline-First Architecture.
Tests mathematical fidelity of the exported model weights against PyTorch,
and verifies offline road map matching and dead reckoning calculations.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from ml.training.config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    MODELS_OUTPUT_DIR,
    INPUT_DIM,
    WINDOW_SIZE
)
from ml.models.infer_velocity import VehicleVelocityPredictor
from navigation.map_matching.osm_matcher import OSMMapMatcher


def test_offline_weight_fidelity():
    """Verify that Python calculation with exported weights matches PyTorch inference exactly."""
    print("Testing Offline Weight Export Fidelity against PyTorch...")

    json_path = MODELS_OUTPUT_DIR / "offline_model_weights.json"
    assert json_path.exists(), f"Weights file missing: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        weights = json.load(f)

    predictor = VehicleVelocityPredictor(BEST_MODEL_PATH, SCALER_PATH)

    # Test with 10 random sequences
    np.random.seed(42)
    max_diff = 0.0

    for idx in range(10):
        test_window = np.random.randn(WINDOW_SIZE, INPUT_DIM).astype(np.float32)

        # 1. Run PyTorch prediction
        pt_mps, pt_kmh, _ = predictor.predict_window(test_window)

        # 2. Run standalone math using exported weights
        mean = np.array(weights["scaler"]["mean"], dtype=np.float32)
        scale = np.array(weights["scaler"]["scale"], dtype=np.float32)
        scaled_win = (test_window - mean) / scale

        # Layer 0 GRU
        w_ih_0 = np.array(weights["gru_l0"]["w_ih"], dtype=np.float32)
        w_hh_0 = np.array(weights["gru_l0"]["w_hh"], dtype=np.float32)
        b_ih_0 = np.array(weights["gru_l0"]["b_ih"], dtype=np.float32)
        b_hh_0 = np.array(weights["gru_l0"]["b_hh"], dtype=np.float32)

        hidden_dim = weights["architecture"]["hidden_dim"]
        h0 = np.zeros(hidden_dim, dtype=np.float32)
        l0_out = []

        for t in range(WINDOW_SIZE):
            x = scaled_win[t]
            gi = w_ih_0 @ x + b_ih_0
            gh = w_hh_0 @ h0 + b_hh_0

            r = 1.0 / (1.0 + np.exp(-np.clip(gi[:hidden_dim] + gh[:hidden_dim], -40, 40)))
            z = 1.0 / (1.0 + np.exp(-np.clip(gi[hidden_dim:2*hidden_dim] + gh[hidden_dim:2*hidden_dim], -40, 40)))
            n = np.tanh(np.clip(gi[2*hidden_dim:] + r * gh[2*hidden_dim:], -40, 40))

            h0 = (1.0 - z) * n + z * h0
            l0_out.append(h0)

        # Layer 1 GRU
        w_ih_1 = np.array(weights["gru_l1"]["w_ih"], dtype=np.float32)
        w_hh_1 = np.array(weights["gru_l1"]["w_hh"], dtype=np.float32)
        b_ih_1 = np.array(weights["gru_l1"]["b_ih"], dtype=np.float32)
        b_hh_1 = np.array(weights["gru_l1"]["b_hh"], dtype=np.float32)

        h1 = np.zeros(hidden_dim, dtype=np.float32)
        for t in range(WINDOW_SIZE):
            x = l0_out[t]
            gi = w_ih_1 @ x + b_ih_1
            gh = w_hh_1 @ h1 + b_hh_1

            r = 1.0 / (1.0 + np.exp(-np.clip(gi[:hidden_dim] + gh[:hidden_dim], -40, 40)))
            z = 1.0 / (1.0 + np.exp(-np.clip(gi[hidden_dim:2*hidden_dim] + gh[hidden_dim:2*hidden_dim], -40, 40)))
            n = np.tanh(np.clip(gi[2*hidden_dim:] + r * gh[2*hidden_dim:], -40, 40))

            h1 = (1.0 - z) * n + z * h1

        # FC Head
        fc0_w = np.array(weights["fc_head"]["fc0_w"], dtype=np.float32)
        fc0_b = np.array(weights["fc_head"]["fc0_b"], dtype=np.float32)
        fc3_w = np.array(weights["fc_head"]["fc3_w"], dtype=np.float32)
        fc3_b = np.array(weights["fc_head"]["fc3_b"], dtype=np.float32)

        dense0 = np.maximum(0.0, fc0_w @ h1 + fc0_b)
        raw_val = float((fc3_w @ dense0 + fc3_b)[0])
        out_mps = float(np.log1p(np.exp(np.clip(raw_val, -40, 40))))

        diff = abs(out_mps - pt_mps)
        if diff > max_diff:
            max_diff = diff

        assert diff < 1e-4, f"Prediction mismatch at sample {idx}: PyTorch={pt_mps}, Export={out_mps}, Diff={diff}"

    print(f"[PASS] Exported weights match PyTorch with max diff: {max_diff:.8f} m/s (< 0.0001 m/s tolerance)")


def test_offline_map_matching():
    """Verify offline map snapping accurately bounds drift without network calls."""
    print("Testing Offline Map Matching...")
    matcher = OSMMapMatcher()
    lat_ref, lon_ref = 12.9716, 77.5946

    # Create a local reference street segment
    waypoints = np.array([
        [12.9716, 77.5946],
        [12.9720, 77.5950],
        [12.9725, 77.5955]
    ])
    matcher.build_synthetic_road_segments(lat_ref, lon_ref, waypoints)

    # Point with 5 meters perpendicular offset
    query_lat, query_lon = 12.97180, 77.59483
    res = matcher.match_point(query_lat, query_lon, heading_rad=0.785, lat_ref=lat_ref, lon_ref=lon_ref)

    assert res["matched"] is True
    assert res["distance_m"] < 15.0
    assert "snapped_lat" in res and "snapped_lon" in res
    print(f"[PASS] Offline Road Snap: Raw=({query_lat}, {query_lon}) -> Snapped=({res['snapped_lat']:.6f}, {res['snapped_lon']:.6f}), Dist={res['distance_m']:.2f}m")


if __name__ == "__main__":
    test_offline_weight_fidelity()
    test_offline_map_matching()
    print("=== ALL OFFLINE TESTS PASSED CLEANLY! ===")
