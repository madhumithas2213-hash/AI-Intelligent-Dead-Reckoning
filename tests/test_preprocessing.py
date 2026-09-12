"""
Unit Tests for ml/preprocessing module.
Tests Importer schema mapping & resampling, Cleaner Butterworth filtering & windowing,
and PyTorch IMUWindowDataset DataLoader integration.
"""

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ml.preprocessing.importer import SensorDataImporter
from ml.preprocessing.cleaner import SensorDataCleaner
from ml.preprocessing.dataset import IMUWindowDataset


def sample_raw_df() -> pd.DataFrame:
    """Generate synthetic irregular raw sensor log dataframe."""
    np.random.seed(42)
    # Irregular timestamps around 90-110 Hz
    dt_samples = np.random.uniform(0.009, 0.011, size=500)
    timestamps = np.cumsum(dt_samples)

    # Synthetic IMU signals
    ax = np.sin(2 * np.pi * 1.0 * timestamps) + np.random.normal(0, 0.1, size=500)
    ay = np.cos(2 * np.pi * 1.0 * timestamps) + np.random.normal(0, 0.1, size=500)
    az = 9.81 + np.random.normal(0, 0.05, size=500)

    gx = np.random.normal(0, 0.01, size=500)
    gy = np.random.normal(0, 0.01, size=500)
    gz = np.random.normal(0, 0.01, size=500)

    df = pd.DataFrame({
        "t": timestamps,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
        "lat": 12.9716 + timestamps * 1e-5,
        "lon": 77.5946 + timestamps * 1e-5,
        "spd": 10.0 + np.sin(timestamps),
    })
    return df


class TestPreprocessing(unittest.TestCase):
    """Unit tests for ml/preprocessing module."""

    def test_importer_schema_normalization(self):
        """Test column alias normalization."""
        raw_df = sample_raw_df()
        importer = SensorDataImporter(target_sample_rate_hz=100.0)
        df_clean = importer.normalize_schema(raw_df)

        self.assertIn("timestamp", df_clean.columns)
        self.assertIn("acc_x", df_clean.columns)
        self.assertIn("acc_y", df_clean.columns)
        self.assertIn("acc_z", df_clean.columns)
        self.assertIn("gyro_x", df_clean.columns)
        self.assertIn("latitude", df_clean.columns)
        self.assertIn("speed", df_clean.columns)

        self.assertTrue(importer.validate_schema(df_clean))

    def test_importer_uniform_resampling(self):
        """Test interpolation to uniform 100 Hz time grid."""
        raw_df = sample_raw_df()
        importer = SensorDataImporter(target_sample_rate_hz=100.0)
        df_clean = importer.normalize_schema(raw_df)
        df_uniform = importer.resample_to_uniform_grid(df_clean)

        timestamps = df_uniform["timestamp"].to_numpy()
        dt_intervals = np.diff(timestamps)

        # Verify uniform time step of 0.01s (100 Hz)
        np.testing.assert_allclose(dt_intervals, 0.01, atol=1e-5)
        self.assertGreater(len(df_uniform), 400)

    def test_cleaner_butterworth_filter(self):
        """Test zero-phase Butterworth filtering."""
        cleaner = SensorDataCleaner(sample_rate_hz=100.0)
        t = np.linspace(0, 2.0, 200)

        # Signal with 2 Hz clean sine wave + 40 Hz high frequency noise
        clean_signal = np.sin(2 * np.pi * 2 * t)
        noisy_signal = clean_signal + 0.5 * np.sin(2 * np.pi * 40 * t)

        filtered = cleaner.apply_butterworth_filter(noisy_signal, cutoff_hz=5.0, filter_type="low")

        # High frequency noise should be significantly attenuated
        rmse_raw = np.sqrt(np.mean((noisy_signal - clean_signal) ** 2))
        rmse_filtered = np.sqrt(np.mean((filtered - clean_signal) ** 2))

        self.assertLess(rmse_filtered, rmse_raw * 0.3)

    def test_cleaner_zupt_detection(self):
        """Test Zero-Velocity Update stationary period detection."""
        cleaner = SensorDataCleaner(sample_rate_hz=100.0)

        # 100 stationary samples followed by 100 dynamic samples
        stat_accel = np.random.normal(0, 0.001, size=(100, 3)) + np.array([0, 0, 9.81])
        dyn_accel = np.random.normal(0, 1.5, size=(100, 3)) + np.array([0, 0, 9.81])
        accel = np.vstack([stat_accel, dyn_accel])

        stat_gyro = np.random.normal(0, 0.0001, size=(100, 3))
        dyn_gyro = np.random.normal(0, 0.5, size=(100, 3))
        gyro = np.vstack([stat_gyro, dyn_gyro])

        is_stationary = cleaner.detect_zero_velocity(accel, gyro, window_size=10)

        # First 50 samples should be marked stationary
        self.assertTrue(np.all(is_stationary[10:80]))
        # Dynamic region should NOT be marked stationary
        self.assertFalse(np.any(is_stationary[120:190]))

    def test_cleaner_sliding_windows(self):
        """Test temporal window segmentation shape."""
        cleaner = SensorDataCleaner(sample_rate_hz=100.0)
        data = np.random.randn(500, 6)
        labels = np.random.randn(500)

        windows, window_labels = cleaner.create_sliding_windows(
            data, labels=labels, window_size=200, step_size=10
        )

        # Expected num_windows = (500 - 200) // 10 + 1 = 31
        self.assertEqual(windows.shape, (31, 200, 6))
        self.assertEqual(window_labels.shape, (31,))

    def test_imu_window_dataset_pytorch(self):
        """Test IMUWindowDataset indexing and PyTorch DataLoader compatibility."""
        features = np.random.randn(50, 200, 6).astype(np.float32)
        labels = np.random.randn(50).astype(np.float32)

        dataset = IMUWindowDataset(features, labels)

        if HAS_TORCH:
            loader = DataLoader(dataset, batch_size=8, shuffle=True)
            batch_x, batch_y = next(iter(loader))
            self.assertEqual(batch_x.shape, (8, 200, 6))
            self.assertEqual(batch_y.shape, (8,))
            self.assertIsInstance(batch_x, torch.Tensor)
            self.assertIsInstance(batch_y, torch.Tensor)
        else:
            x, y = dataset[0]
            self.assertEqual(x.shape, (200, 6))
            self.assertEqual(y.shape, ())
            self.assertIsInstance(x, np.ndarray)


if __name__ == "__main__":
    unittest.main()
