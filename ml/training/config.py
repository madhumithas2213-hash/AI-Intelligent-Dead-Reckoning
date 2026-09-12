"""
Centralized Configuration for AI/ML Vehicle Velocity Estimation (Step 4).
Contains model hyper-parameters, windowing settings, feature lists, target definitions,
and project directory paths.
"""

from pathlib import Path
from typing import List

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# File Paths
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "outputs"
MODELS_OUTPUT_DIR = OUTPUT_DIR / "models"
PLOTS_OUTPUT_DIR = OUTPUT_DIR / "plots" / "velocity"
SCALER_PATH = PROJECT_ROOT / "ml" / "models" / "velocity_scaler.pkl"
BEST_MODEL_PATH = MODELS_OUTPUT_DIR / "best_gru_velocity.pt"
EXPORT_TORCHSCRIPT_PATH = MODELS_OUTPUT_DIR / "gru_velocity_model.pt"
EXPORT_ONNX_PATH = MODELS_OUTPUT_DIR / "gru_velocity_model.onnx"
REPORT_PATH = OUTPUT_DIR / "velocity_model_report.txt"

# Ensure Output Directories Exist
MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data Splitting Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Time-Series Windowing Parameters
WINDOW_SECONDS = 2.0     # 2.0 seconds window duration
STRIDE_SECONDS = 0.5     # 0.5 seconds stride overlap
NOMINAL_SAMPLING_RATE_HZ = 10.0  # Nominal 10 Hz sampling rate
WINDOW_SIZE = int(WINDOW_SECONDS * NOMINAL_SAMPLING_RATE_HZ)  # 20 samples
STEP_SIZE = int(STRIDE_SECONDS * NOMINAL_SAMPLING_RATE_HZ)     # 5 samples
MAX_TIMESTAMP_GAP_MS = 2000.0  # Gaps > 2s break sequence windowing

# Model Hyper-parameters
INPUT_DIM = 14
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
EPOCHS = 35
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7

# Verified 14 IMU/Sensor Feature Channels (STRICTLY NO GPS INPUTS)
FEATURE_COLUMNS: List[str] = [
    "accel_raw_x_ms2",
    "accel_raw_y_ms2",
    "accel_raw_z_ms2",
    "gyro_raw_yaw_rads",
    "gyro_raw_pitch_rads",
    "gyro_raw_roll_rads",
    "gravity_x_ms2",
    "gravity_y_ms2",
    "gravity_z_ms2",
    "mag_x_ut",
    "mag_y_ut",
    "mag_z_ut",
    "accel_raw_mag_ms2",
    "gyro_raw_mag_rads",
]

# Ground-Truth Target Column & Units
TARGET_COLUMN = "gps_speed_mps"
TARGET_UNIT = "m/s"
KMH_TO_MPS = 1.0 / 3.6
MPS_TO_KMH = 3.6
