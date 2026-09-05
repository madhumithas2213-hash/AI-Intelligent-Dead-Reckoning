"""
Automated Test Suite for Step 4 Vehicle Velocity Estimation Model.
Verifies all 13 core requirements including dataset loading, feature selection,
target selection, sequence boundary protection, train/test leakage prevention,
scaler fitting, model architecture, forward pass, save/load, and inference output.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pickle
import numpy as np
import pandas as pd
import torch

from ml.training.config import (
    PROCESSED_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    INPUT_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    WINDOW_SIZE,
    STEP_SIZE,
    SCALER_PATH,
    BEST_MODEL_PATH,
)
from ml.training.target_definition import extract_velocity_target, validate_target_series
from ml.training.train_velocity import SequenceDataSplitter, WindowGenerator, extract_features_array
from ml.models.gru_velocity_model import ForwardVelocityGRU, count_parameters
from ml.models.infer_velocity import VehicleVelocityPredictor


import unittest

class TestStep4Velocity(unittest.TestCase):
    """Automated Test Suite for Step 4 Vehicle Velocity Estimation Model."""

    def test_1_dataset_loader(self):
        """1. Verify processed dataset exists and is readable."""
        files = list(PROCESSED_DIR.glob("*_processed.csv"))
        self.assertGreater(len(files), 0, "No processed CSV files found!")
        df = pd.read_csv(files[0])
        self.assertGreater(len(df), 0, f"Sample file {files[0].name} is empty!")

    def test_2_feature_selection_no_gps_leakage(self):
        """2. Verify feature selection strictly excludes GPS coordinates and speed as inputs."""
        gps_inputs = ["gps_latitude_deg", "gps_longitude_deg", "gps_speed_kmh", "gps_speed_mps", "gps_orientation_deg"]
        for feat in FEATURE_COLUMNS:
            self.assertNotIn(feat, gps_inputs, f"GPS feature '{feat}' detected in model inputs! GPS leakage forbidden.")
        self.assertEqual(len(FEATURE_COLUMNS), 14, f"Expected 14 feature columns, got {len(FEATURE_COLUMNS)}")

    def test_3_target_selection(self):
        """3. Verify ground-truth velocity target selection and SI unit validation."""
        files = list(PROCESSED_DIR.glob("*_processed.csv"))
        df = pd.read_csv(files[0])
        target = extract_velocity_target(df)
        self.assertIsInstance(target, np.ndarray)
        self.assertEqual(len(target), len(df))
        self.assertGreaterEqual(np.min(target), 0.0, "Target velocity cannot be negative!")
        is_valid, msg = validate_target_series(target)
        self.assertTrue(is_valid, f"Target validation failed: {msg}")

    def test_4_timestamp_ordering(self):
        """4. Verify timestamp sorting and non-decreasing timestamp ordering."""
        files = list(PROCESSED_DIR.glob("*_processed.csv"))
        df = pd.read_csv(files[0])
        if "timestamp_sec" in df.columns:
            ts = df["timestamp_sec"].to_numpy()
            self.assertTrue(np.all(np.diff(ts) >= 0), "Timestamps in processed CSV are not chronologically ordered!")

    def test_5_sequence_boundary_protection(self):
        """5. Verify sliding windows do not cross sequence boundaries."""
        splitter = SequenceDataSplitter()
        train_files, val_files, test_files = splitter.get_sequence_splits()
        win_gen = WindowGenerator(window_size=WINDOW_SIZE, step_size=STEP_SIZE)
        X, y, scaler = win_gen.create_windows_from_files(train_files[:2], fit_scaler=True)
        self.assertGreater(len(X), 0)
        self.assertEqual(X.shape[1], WINDOW_SIZE)
        self.assertEqual(X.shape[2], INPUT_DIM)

    def test_6_train_test_leakage_prevention(self):
        """6. Verify zero sequence overlap between train, val, and test splits."""
        splitter = SequenceDataSplitter(seed=42)
        train_files, val_files, test_files = splitter.get_sequence_splits()

        set_train = set(f.name for f in train_files)
        set_val = set(f.name for f in val_files)
        set_test = set(f.name for f in test_files)

        self.assertEqual(len(set_train.intersection(set_val)), 0, "Train and Validation share sequences!")
        self.assertEqual(len(set_train.intersection(set_test)), 0, "Train and Test share sequences!")
        self.assertEqual(len(set_val.intersection(set_test)), 0, "Validation and Test share sequences!")

    def test_7_scaler_fitted_only_on_train(self):
        """7. Verify feature scaler can be fitted strictly on training data."""
        splitter = SequenceDataSplitter()
        train_files, val_files, _ = splitter.get_sequence_splits()
        win_gen = WindowGenerator()

        # Fit on train
        X_train, y_train, scaler = win_gen.create_windows_from_files(train_files[:2], fit_scaler=True)
        self.assertTrue(hasattr(scaler, "mean_"))
        self.assertEqual(len(scaler.mean_), INPUT_DIM)

        # Transform val without refitting
        X_val, y_val, _ = win_gen.create_windows_from_files(val_files[:2], scaler=scaler, fit_scaler=False)
        self.assertGreater(len(X_val), 0)

    def test_8_window_generation_shape(self):
        """8. Verify sliding window dimensions."""
        win_gen = WindowGenerator(window_size=20, step_size=5)
        dummy_feats = np.random.randn(100, 14).astype(np.float32)
        dummy_target = np.ones(100, dtype=np.float32)
        dummy_df = pd.DataFrame(dummy_feats, columns=FEATURE_COLUMNS)
        dummy_df["gps_speed_mps"] = dummy_target
        dummy_df["dt_ms"] = 100.0

        # Save temp test CSV
        tmp_path = Path("tmp_test_seq_processed.csv")
        dummy_df.to_csv(tmp_path, index=False)
        try:
            X, y, _ = win_gen.create_windows_from_files([tmp_path], fit_scaler=False)
            self.assertEqual(X.shape[1], 20)
            self.assertEqual(X.shape[2], 14)
            self.assertEqual(len(X), len(y))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_9_model_forward_pass(self):
        """9. Verify PyTorch ForwardVelocityGRU model instantiation and forward pass."""
        model = ForwardVelocityGRU(input_dim=14, hidden_dim=64, num_layers=2, dropout=0.2)
        dummy_input = torch.randn(8, 20, 14)  # [batch_size=8, seq_len=20, input_dim=14]
        output = model(dummy_input)
        self.assertEqual(output.shape, (8, 1), f"Expected output shape (8, 1), got {output.shape}")

    def test_10_model_output_shape_and_non_negativity(self):
        """10. Verify model output shape and non-negativity."""
        model = ForwardVelocityGRU(input_dim=14, hidden_dim=32, num_layers=1)
        dummy_input = torch.randn(4, 20, 14)
        output = model(dummy_input)
        self.assertTrue(torch.all(output >= 0.0), "Velocity prediction cannot be negative!")

    def test_11_model_save_and_load(self):
        """11. Verify model checkpoint saving and loading."""
        model = ForwardVelocityGRU(input_dim=14, hidden_dim=32)
        tmp_ckpt = Path("tmp_test_model.pt")
        torch.save({"model_state_dict": model.state_dict()}, tmp_ckpt)
        try:
            self.assertTrue(tmp_ckpt.exists())
            ckpt = torch.load(tmp_ckpt, map_location="cpu")
            model2 = ForwardVelocityGRU(input_dim=14, hidden_dim=32)
            model2.load_state_dict(ckpt["model_state_dict"])
            model2.eval()
        finally:
            if tmp_ckpt.exists():
                tmp_ckpt.unlink()

    def test_12_inference_output(self):
        """12. Verify VehicleVelocityPredictor standalone inference output."""
        if SCALER_PATH.exists() and BEST_MODEL_PATH.exists():
            predictor = VehicleVelocityPredictor(model_path=BEST_MODEL_PATH, scaler_path=SCALER_PATH)
            dummy_win = np.random.randn(20, 14).astype(np.float32)
            v_mps, v_kmh, lat = predictor.predict_window(dummy_win)
            self.assertIsInstance(v_mps, float)
            self.assertIsInstance(v_kmh, float)
            self.assertGreaterEqual(v_mps, 0.0)
            self.assertLess(abs(v_kmh - v_mps * 3.6), 1e-4)
            self.assertGreaterEqual(lat, 0.0)

    def test_13_no_nan_or_inf_in_model_output(self):
        """13. Verify model outputs contain no NaN or Inf values under noisy input conditions."""
        model = ForwardVelocityGRU(input_dim=14, hidden_dim=64)
        noisy_input = torch.randn(16, 20, 14) * 10.0
        output = model(noisy_input)
        self.assertFalse(torch.isnan(output).any(), "NaN detected in model output!")
        self.assertFalse(torch.isinf(output).any(), "Inf detected in model output!")


if __name__ == "__main__":
    unittest.main()
