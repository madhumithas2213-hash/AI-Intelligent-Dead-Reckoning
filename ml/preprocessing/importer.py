"""
Sensor Data Importer Module.
Handles raw sensor file loading, flexible column schema normalization,
and time-series resampling to a uniform sampling frequency (100 Hz).
"""

from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# Standard canonical column names expected by the IDR pipeline
CANONICAL_COLUMNS = {
    "timestamp": "timestamp",
    "acc_x": "acc_x",
    "acc_y": "acc_y",
    "acc_z": "acc_z",
    "gyro_x": "gyro_x",
    "gyro_y": "gyro_y",
    "gyro_z": "gyro_z",
    "mag_x": "mag_x",
    "mag_y": "mag_y",
    "mag_z": "mag_z",
    "latitude": "latitude",
    "longitude": "longitude",
    "speed": "speed",
    "heading": "heading",
}

# Alias dictionary for auto-mapping raw log column variants
COLUMN_ALIASES = {
    # Timestamps
    "time": "timestamp",
    "timestamp_ms": "timestamp",
    "time_ms": "timestamp",
    "t": "timestamp",
    
    # Accelerometer
    "ax": "acc_x",
    "ay": "acc_y",
    "az": "acc_z",
    "accel_x": "acc_x",
    "accel_y": "acc_y",
    "accel_z": "acc_z",
    "accelerometer_x": "acc_x",
    "accelerometer_y": "acc_y",
    "accelerometer_z": "acc_z",
    
    # Gyroscope
    "gx": "gyro_x",
    "gy": "gyro_y",
    "gz": "gyro_z",
    "gyr_x": "gyro_x",
    "gyr_y": "gyro_y",
    "gyr_z": "gyro_z",
    "gyroscope_x": "gyro_x",
    "gyroscope_y": "gyro_y",
    "gyroscope_z": "gyro_z",
    
    # Magnetometer
    "mx": "mag_x",
    "my": "mag_y",
    "mz": "mag_z",
    "magnetometer_x": "mag_x",
    "magnetometer_y": "mag_y",
    "magnetometer_z": "mag_z",
    
    # GNSS
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "spd": "speed",
    "velocity": "speed",
}


class SensorDataImporter:
    """
    Loads raw sensor telemetry files, maps column aliases to canonical names,
    and interpolates irregular timestamps into a uniform time grid.
    """

    def __init__(self, target_sample_rate_hz: float = 100.0) -> None:
        """
        Args:
            target_sample_rate_hz: Target output uniform sampling rate in Hz (default: 100.0 Hz).
        """
        self.target_sample_rate_hz = target_sample_rate_hz
        self.dt = 1.0 / target_sample_rate_hz

    def load_raw_log(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load raw sensor log file into a Pandas DataFrame.

        Args:
            file_path: Path to the raw sensor log file (.csv, .parquet, .h5).

        Returns:
            pd.DataFrame: Loaded raw DataFrame.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Sensor log file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
        elif suffix in [".h5", ".hdf5"]:
            df = pd.read_hdf(path)
        else:
            raise ValueError(f"Unsupported file extension '{suffix}'. Use .csv, .parquet, or .h5")

        return df

    def normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename column headers matching known aliases to canonical IDR names.

        Args:
            df: Input DataFrame with arbitrary headers.

        Returns:
            pd.DataFrame: DataFrame with canonical column names.
        """
        df_clean = df.copy()
        rename_map = {}
        for col in df_clean.columns:
            col_lower = col.strip().lower()
            if col_lower in COLUMN_ALIASES:
                rename_map[col] = COLUMN_ALIASES[col_lower]

        if rename_map:
            df_clean = df_clean.rename(columns=rename_map)

        return df_clean

    def validate_schema(self, df: pd.DataFrame, required_channels: Optional[List[str]] = None) -> bool:
        """
        Check if required canonical channels exist in DataFrame.

        Args:
            df: Canonicalized DataFrame.
            required_channels: List of channel names required (default: IMU channels + timestamp).

        Returns:
            bool: True if valid, raises ValueError if missing.
        """
        if required_channels is None:
            required_channels = ["timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

        missing = [ch for ch in required_channels if ch not in df.columns]
        if missing:
            raise ValueError(f"Missing required sensor channels: {missing}")

        return True

    def resample_to_uniform_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample irregular time-series data to a uniform time grid.

        Args:
            df: DataFrame containing a 'timestamp' column (seconds or ms) and numeric sensor channels.

        Returns:
            pd.DataFrame: Resampled DataFrame on uniform time grid at target_sample_rate_hz.
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column for resampling.")

        df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)
        timestamps = df_sorted["timestamp"].to_numpy(dtype=np.float64)

        # Automatically convert millisecond epoch timestamps to seconds if values are large
        if timestamps[0] > 1e9:
            timestamps = timestamps / 1000.0  # ms -> seconds
        elif np.mean(np.diff(timestamps)) > 1.0:
            timestamps = timestamps / 1000.0

        t_min = timestamps[0]
        t_max = timestamps[-1]

        # Generate uniform target timestamp grid
        uniform_timestamps = np.arange(t_min, t_max, self.dt)

        resampled_data: Dict[str, np.ndarray] = {"timestamp": uniform_timestamps}

        # Interpolate numeric columns
        numeric_cols = [c for c in df_sorted.select_dtypes(include=[np.number]).columns if c != "timestamp"]

        for col in numeric_cols:
            vals = df_sorted[col].to_numpy(dtype=np.float64)
            # 1D Cubic / Linear spline interpolation
            if len(timestamps) > 3:
                interp_func = interp1d(timestamps, vals, kind="linear", fill_value="extrapolate")
            else:
                interp_func = interp1d(timestamps, vals, kind="nearest", fill_value="extrapolate")
            
            resampled_data[col] = interp_func(uniform_timestamps)

        return pd.DataFrame(resampled_data)
