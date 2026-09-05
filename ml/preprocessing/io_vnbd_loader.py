"""
IO-VNBD Dataset Loader Module.
Discovers and loads raw smartphone sensor CSV files from dataset/raw/IO-VNBD-master/
without modifying raw files, mapping raw column headers safely to canonical snake_case names,
and preserving individual trip sequence boundaries to prevent data leakage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd


# Explicit dictionary mapping all 36 raw column headers discovered in IO-VNBD
# to clean canonical variable names.
COLUMN_MAPPING: Dict[str, str] = {
    # GPS
    "GPS LATITUDE (degrees)": "gps_latitude_deg",
    "GPS LONGITUDE (degrees)": "gps_longitude_deg",
    " GPS LONGITUDE (degrees)": "gps_longitude_deg",
    " GPS ALTITUDE (m)": "gps_altitude_m",
    "GPS SPEED (Kmh)": "gps_speed_kmh",
    " GPS SPEED (Kmh)": "gps_speed_kmh",
    "GPS ACCURACY (m)": "gps_accuracy_m",
    " GPS ACCURACY (m)": "gps_accuracy_m",
    "GPS ORIENTATION (°)": "gps_orientation_deg",
    " GPS ORIENTATION (°)": "gps_orientation_deg",
    "GPS SATELLITES IN RANGE": "gps_satellites_in_range",
    " SATELLITES IN RANGE": "gps_satellites_in_range",

    # Timestamps
    " TIME SINCE START (ms)": "time_since_start_ms",
    " DATE (YYYY-MO-DD HH-MI-SS_SSS)": "date_timestamp_str",
    " DATE (YYYY-MO-DD HH-MI-SS_SSS": "date_timestamp_str",

    # Accelerometer
    "ACCELEROMETER X (m/s\ufffd) ": "accel_raw_x_ms2",
    " ACCELEROMETER X (m/s\ufffd) ": "accel_raw_x_ms2",
    " ACCELEROMETER Y (m/s\ufffd)": "accel_raw_y_ms2",
    " ACCELEROMETER Z (m/s\ufffd)": "accel_raw_z_ms2",
    " ACCELEROMETER X (m/s²) ": "accel_raw_x_ms2",
    " ACCELEROMETER Y (m/s²)": "accel_raw_y_ms2",
    " ACCELEROMETER Z (m/s²)": "accel_raw_z_ms2",

    # Gravity
    " GRAVITY X (m/s\ufffd)": "gravity_x_ms2",
    " GRAVITY Y (m/s\ufffd)": "gravity_y_ms2",
    " GRAVITY Z (m/s\ufffd)": "gravity_z_ms2",
    " GRAVITY X (m/s²)": "gravity_x_ms2",
    " GRAVITY Y (m/s²)": "gravity_y_ms2",
    " GRAVITY Z (m/s²)": "gravity_z_ms2",

    # Gyroscope (Yaw/Pitch/Roll or X/Y/Z)
    " GYROSCOPE Yaw (rad/s)": "gyro_raw_yaw_rads",
    " GYROSCOPE Pitch (rad/s)": "gyro_raw_pitch_rads",
    " GYROSCOPE Roll (rad/s)": "gyro_raw_roll_rads",
    " GYROSCOPE X (rad/s)": "gyro_raw_pitch_rads",
    " GYROSCOPE Y (rad/s)": "gyro_raw_roll_rads",
    " GYROSCOPE Z (rad/s)": "gyro_raw_yaw_rads",

    # Magnetometer
    " MAGNETIC FIELD X (μT)": "mag_x_ut",
    " MAGNETIC FIELD Y (μT)": "mag_y_ut",
    " MAGNETIC FIELD Z (μT)": "mag_z_ut",

    # Orientation
    " ORIENTATION (Yaw) (°)": "orientation_yaw_deg",
    " ORIENTATION (Azimuth) (°)": "orientation_yaw_deg",
    " ORIENTATION (Pitch) (°)": "orientation_pitch_deg",
    " ORIENTATION (Roll ) (°)": "orientation_roll_deg",
    " ORIENTATION (Roll) (°)": "orientation_roll_deg",
}


@dataclass
class SequenceData:
    """Encapsulates a single smartphone trip sequence to preserve sequence isolation."""
    sequence_id: str
    file_path: Path
    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    column_mapping: Dict[str, str]


class IOVNBDLoader:
    """
    Discovers and loads smartphone sensor CSV logs from the IO-VNBD dataset.
    """

    def __init__(self, raw_dataset_dir: Union[str, Path] = "dataset/raw/IO-VNBD-master") -> None:
        """
        Args:
            raw_dataset_dir: Directory path containing the raw IO-VNBD dataset.
        """
        self.raw_dataset_dir = Path(raw_dataset_dir)
        if not self.raw_dataset_dir.exists():
            raise FileNotFoundError(f"Raw dataset directory not found at: {self.raw_dataset_dir}")

    def discover_smartphone_sequences(self) -> List[Path]:
        """
        Recursively find all smartphone telemetry CSV files in IO-VNBD.
        Matches files starting with 'S-' or housed inside smartphone folders.

        Returns:
            List[Path]: Sorted list of smartphone CSV file paths.
        """
        csv_files = list(self.raw_dataset_dir.rglob("*.csv"))
        smartphone_files = []

        for f in csv_files:
            # Check if file size > 1KB (avoid Git LFS pointers if unpulled)
            if f.stat().st_size > 1000:
                name_upper = f.name.upper()
                parent_upper = str(f.parent).upper()
                # Exclude vehicle CAN telemetry CSV files (V-*.csv or V_*.csv)
                if name_upper.startswith("V-") or name_upper.startswith("V_"):
                    continue
                if name_upper.startswith("S-") or name_upper.startswith("S_") or "S-DATASET" in parent_upper or "S (" in parent_upper:
                    smartphone_files.append(f)

        smartphone_files.sort(key=lambda p: p.name)
        return smartphone_files

    def load_sequence(self, file_path: Union[str, Path]) -> SequenceData:
        """
        Load a single raw smartphone sequence CSV file and apply clean column mapping.

        Args:
            file_path: Path to the smartphone CSV log.

        Returns:
            SequenceData: Sequence container with raw and normalized DataFrames.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Read CSV using latin1 encoding to handle degree symbols and special characters safely
        df_raw = pd.read_csv(path, encoding="latin1")

        # Map original headers to canonical names
        rename_dict = {}
        for col in df_raw.columns:
            if col == "Unnamed: 24":
                continue
            
            col_strip = col.strip()
            # Explicit match
            if col in COLUMN_MAPPING:
                rename_dict[col] = COLUMN_MAPPING[col]
            elif col_strip in COLUMN_MAPPING:
                rename_dict[col] = COLUMN_MAPPING[col_strip]
            else:
                # Safe fallback mapping
                col_clean = (
                    col_strip.lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("°", "deg")
                    .replace("m/s\ufffd", "ms2")
                    .replace("m/s²", "ms2")
                    .replace("rad/s", "rads")
                    .replace("μt", "ut")
                )
                rename_dict[col] = col_clean

        df_clean = df_raw.rename(columns=rename_dict)
        if "Unnamed: 24" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Unnamed: 24"])

        # Create unique sequence_id incorporating parent folder name to prevent collisions
        parent_tag = path.parent.name.replace(" ", "_").replace("(", "").replace(")", "")
        sequence_id = f"{parent_tag}_{path.stem}"

        return SequenceData(
            sequence_id=sequence_id,
            file_path=path,
            raw_df=df_raw,
            clean_df=df_clean,
            column_mapping=rename_dict,
        )

    def get_column_mapping_dict(self) -> Dict[str, str]:
        """Get the master raw-to-canonical column mapping dictionary."""
        return COLUMN_MAPPING.copy()
