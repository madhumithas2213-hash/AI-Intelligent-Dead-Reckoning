"""
Training Script for GRU Vehicle Velocity Estimator (Step 4).
Splits dataset by complete trip sequences (70% train, 15% val, 15% test) to prevent data leakage,
fits feature scalers strictly on training sequences, trains GRU with MAE/Huber loss and early stopping,
saves model checkpoints, and exports scaler for mobile deployment.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import pickle
from typing import Dict, List, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ml.training.config import (
    PROCESSED_DIR,
    MODELS_OUTPUT_DIR,
    SCALER_PATH,
    BEST_MODEL_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    WINDOW_SIZE,
    STEP_SIZE,
    MAX_TIMESTAMP_GAP_MS,
    INPUT_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    FEATURE_COLUMNS,
)
from ml.training.target_definition import extract_velocity_target
from ml.models.gru_velocity_model import ForwardVelocityGRU, NumPyGRUVelocityModel, count_parameters


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def extract_features_array(df: pd.DataFrame) -> np.ndarray:
    """Extract 14 verified feature channels from sequence DataFrame safely."""
    cols = df.columns.tolist()

    # 1. Accel 3D
    ax = pd.to_numeric(df["accel_raw_x_ms2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "accel_raw_x_ms2" in cols else np.zeros(len(df), dtype=np.float32)
    ay = pd.to_numeric(df["accel_raw_y_ms2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "accel_raw_y_ms2" in cols else np.zeros(len(df), dtype=np.float32)
    az = pd.to_numeric(df["accel_raw_z_ms2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "accel_raw_z_ms2" in cols else np.zeros(len(df), dtype=np.float32)

    # 2. Gyro 3D
    gx = pd.to_numeric(df["gyro_raw_yaw_rads"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "gyro_raw_yaw_rads" in cols else np.zeros(len(df), dtype=np.float32)
    gy = pd.to_numeric(df["gyro_raw_pitch_rads"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "gyro_raw_pitch_rads" in cols else np.zeros(len(df), dtype=np.float32)
    gz = pd.to_numeric(df["gyro_raw_roll_rads"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "gyro_raw_roll_rads" in cols else np.zeros(len(df), dtype=np.float32)

    # 3. Gravity 3D
    grx = pd.to_numeric(df["gravity_x_ms2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "gravity_x_ms2" in cols else np.zeros(len(df), dtype=np.float32)
    gry = pd.to_numeric(df["gravity_y_ms2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "gravity_y_ms2" in cols else np.zeros(len(df), dtype=np.float32)
    grz = pd.to_numeric(df["gravity_z_ms2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "gravity_z_ms2" in cols else np.zeros(len(df), dtype=np.float32)

    # 4. Magnetometer 3D
    mx_col = [c for c in cols if ("mag" in c.lower() or "magnetic" in c.lower()) and "x" in c.lower() and "gyro" not in c.lower()]
    my_col = [c for c in cols if ("mag" in c.lower() or "magnetic" in c.lower()) and "y" in c.lower() and "gyro" not in c.lower()]
    mz_col = [c for c in cols if ("mag" in c.lower() or "magnetic" in c.lower()) and "z" in c.lower() and "gyro" not in c.lower()]

    mx = pd.to_numeric(df[mx_col[0]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if mx_col else np.zeros(len(df), dtype=np.float32)
    my = pd.to_numeric(df[my_col[0]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if my_col else np.zeros(len(df), dtype=np.float32)
    mz = pd.to_numeric(df[mz_col[0]], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if mz_col else np.zeros(len(df), dtype=np.float32)

    # 5. Magnitudes
    amag = pd.to_numeric(df["accel_raw_mag_ms2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "accel_raw_mag_ms2" in cols else np.sqrt(ax**2 + ay**2 + az**2)
    gmag = pd.to_numeric(df["gyro_raw_mag_rads"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if "gyro_raw_mag_rads" in cols else np.sqrt(gx**2 + gy**2 + gz**2)

    feats = np.column_stack([ax, ay, az, gx, gy, gz, grx, gry, grz, mx, my, mz, amag, gmag])
    return feats.astype(np.float32)


class SequenceDataSplitter:
    """
    Splits sequence dataset files into Train (70%), Validation (15%), and Test (15%) splits
    at sequence level to prevent data leakage.
    """

    def __init__(self, processed_dir: Path = PROCESSED_DIR, seed: int = RANDOM_SEED) -> None:
        self.processed_dir = Path(processed_dir)
        self.seed = seed

    def get_sequence_splits(
        self, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Discover sequence files and perform sequence-level splitting.
        """
        files = sorted(list(self.processed_dir.glob("*_processed.csv")))
        if not files:
            raise FileNotFoundError(f"No processed CSV files found in {self.processed_dir}")

        np.random.seed(self.seed)
        indices = np.random.permutation(len(files))

        num_files = len(files)
        num_train = int(num_files * train_ratio)
        num_val = int(num_files * val_ratio)

        train_files = [files[i] for i in indices[:num_train]]
        val_files = [files[i] for i in indices[num_train : num_train + num_val]]
        test_files = [files[i] for i in indices[num_train + num_val :]]

        # Confirm zero sequence overlap
        set_train = set(f.name for f in train_files)
        set_val = set(f.name for f in val_files)
        set_test = set(f.name for f in test_files)

        assert len(set_train.intersection(set_val)) == 0, "Train and Val sequence overlap detected!"
        assert len(set_train.intersection(set_test)) == 0, "Train and Test sequence overlap detected!"
        assert len(set_val.intersection(set_test)) == 0, "Val and Test sequence overlap detected!"

        print(f"[SequenceSplitter] Split {num_files} sequence files (Seed {self.seed}):")
        print(f"  - Train: {len(train_files)} sequences ({len(train_files)/num_files*100:.1f}%)")
        print(f"  - Val:   {len(val_files)} sequences ({len(val_files)/num_files*100:.1f}%)")
        print(f"  - Test:  {len(test_files)} sequences ({len(test_files)/num_files*100:.1f}%)")

        return train_files, val_files, test_files


class WindowGenerator:
    """
    Generates sequence-aware sliding windows for time-series velocity training.
    Prevents windows from crossing sequence boundaries or large timestamp gaps.
    """

    def __init__(self, window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE) -> None:
        self.window_size = window_size
        self.step_size = step_size

    def create_windows_from_files(
        self, files: List[Path], scaler: StandardScaler = None, fit_scaler: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Load sequence files, apply/fit scaler on features, and generate windows.
        """
        all_feature_rows = []
        file_dfs = []

        # 1. Read files and extract raw features & targets
        for f in files:
            df = pd.read_csv(f)
            feats = extract_features_array(df)
            target = extract_velocity_target(df)
            dt_ms = pd.to_numeric(df.get("dt_ms", 100.0), errors="coerce").fillna(100.0).to_numpy()
            file_dfs.append((df, feats, target, dt_ms))

            if fit_scaler:
                all_feature_rows.append(feats)

        # 2. Fit scaler strictly on training feature rows
        if fit_scaler:
            scaler = StandardScaler()
            cat_feats = np.vstack(all_feature_rows)
            scaler.fit(cat_feats)
            print(f"[Scaler] Fitted StandardScaler on {len(cat_feats):,} training feature samples.")

        # 3. Create sequence-aware sliding windows
        X_windows = []
        y_windows = []

        for df, feats, target, dt_ms in file_dfs:
            # Transform features
            feats_scaled = scaler.transform(feats) if scaler is not None else feats
            n_samples = len(feats_scaled)

            if n_samples < self.window_size:
                continue

            for start in range(0, n_samples - self.window_size + 1, self.step_size):
                end = start + self.window_size

                # Check timestamp gap inside window to prevent bad concatenation
                window_dts = dt_ms[start:end]
                if np.max(window_dts) > MAX_TIMESTAMP_GAP_MS:
                    continue  # Skip window crossing temporal gap

                win_X = feats_scaled[start:end]
                win_y = target[end - 1]  # Predict velocity at target timestamp (end of window)

                X_windows.append(win_X)
                y_windows.append(win_y)

        X_arr = np.array(X_windows, dtype=np.float32) if X_windows else np.zeros((0, self.window_size, INPUT_DIM), dtype=np.float32)
        y_arr = np.array(y_windows, dtype=np.float32).reshape(-1, 1) if y_windows else np.zeros((0, 1), dtype=np.float32)

        return X_arr, y_arr, scaler


def train_velocity_model() -> Dict[str, Any]:
    """
    Execute complete Step 4 GRU training pipeline.
    """
    set_seed(RANDOM_SEED)

    print("================================================================================")
    print("        STEP 4 — VEHICLE VELOCITY ESTIMATION MODEL TRAINING")
    print("================================================================================")

    # 1. Split Sequences
    splitter = SequenceDataSplitter()
    train_files, val_files, test_files = splitter.get_sequence_splits()

    # 2. Generate Windowed Datasets
    win_gen = WindowGenerator(window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    print("[Windows] Processing training sequences...")
    X_train, y_train, scaler = win_gen.create_windows_from_files(train_files, fit_scaler=True)
    
    print("[Windows] Processing validation sequences...")
    X_val, y_val, _ = win_gen.create_windows_from_files(val_files, scaler=scaler, fit_scaler=False)
    
    print("[Windows] Processing test sequences...")
    X_test, y_test, _ = win_gen.create_windows_from_files(test_files, scaler=scaler, fit_scaler=False)

    print(f"[Dataset Summary]")
    print(f"  - Train windows: {X_train.shape}")
    print(f"  - Val windows:   {X_val.shape}")
    print(f"  - Test windows:  {X_test.shape}")

    # Save Scaler to ml/models/velocity_scaler.pkl
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Scaler] Saved velocity scaler to {SCALER_PATH}")

    # 3. Model Building & Parameter Counting
    device = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")

    model = ForwardVelocityGRU(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    num_params = count_parameters(model)
    print(f"[Model] ForwardVelocityGRU initialized with {num_params:,} trainable parameters.")

    # 4. Training Loop setup
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.SmoothL1Loss()  # Huber loss for robust velocity regression
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    print("--------------------------------------------------------------------------------")
    print("Starting Model Training...")
    print("--------------------------------------------------------------------------------")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_X)

        train_loss /= len(train_dataset)

        # Validation Pass
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_X)
                val_mae += torch.sum(torch.abs(preds - batch_y)).item()

        val_loss /= len(val_dataset)
        val_mae /= len(val_dataset)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} m/s ({val_mae*3.6:.2f} km/h)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "hyperparams": {
                    "input_dim": INPUT_DIM,
                    "hidden_dim": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                }
            }, BEST_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"[Early Stopping] Validation loss did not improve for {EARLY_STOPPING_PATIENCE} epochs. Stopping.")
                break

    print(f"[Training Complete] Best Model Checkpoint saved to: {BEST_MODEL_PATH}")

    # Save metadata & history
    meta_path = MODELS_OUTPUT_DIR / "velocity_training_history.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "model": model,
        "scaler": scaler,
        "history": history,
        "num_params": num_params,
        "best_val_loss": best_val_loss,
        "X_test": X_test,
        "y_test": y_test,
        "test_files": test_files,
    }


if __name__ == "__main__":
    train_velocity_model()
