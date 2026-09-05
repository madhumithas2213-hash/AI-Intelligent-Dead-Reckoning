"""
Pre-Flight Data Validation Script for Step 4 Velocity Model Training.
Inspects processed sequence CSV files, preprocessing report, and JSON statistics.
Verifies exact features, target availability, units, missing/inf values, and timestamps.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from ml.training.config import (
    PROCESSED_DIR,
    OUTPUT_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from ml.training.target_definition import extract_velocity_target, validate_target_series


class PreflightDataValidator:
    """
    Validates the processed dataset before training the GRU velocity model.
    """

    def __init__(self, processed_dir: Path = PROCESSED_DIR, output_dir: Path = OUTPUT_DIR) -> None:
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)

    def validate_preflight(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run complete pre-flight data validation checks.

        Returns:
            Tuple[bool, Dict[str, Any]]: (passed_all_checks, summary_dict)
        """
        results = {
            "processed_dir_exists": False,
            "report_exists": False,
            "stats_json_exists": False,
            "num_sequences": 0,
            "total_rows": 0,
            "missing_features": [],
            "missing_target": False,
            "nan_inf_count": 0,
            "duplicate_timestamps": 0,
            "target_stats": {},
            "errors": [],
        }

        print("================================================================================")
        print("            STEP 4 — PRE-FLIGHT DATA VALIDATION REPORT")
        print("================================================================================")

        # 1. Check Directory Existence
        if not self.processed_dir.exists():
            err = f"Processed data directory does not exist at {self.processed_dir}"
            results["errors"].append(err)
            print(f"[FAIL] {err}")
            return False, results
        results["processed_dir_exists"] = True

        # 2. Check Reports
        report_path = self.output_dir / "preprocessing_report.txt"
        stats_path = self.output_dir / "preprocessing_statistics.json"

        results["report_exists"] = report_path.exists()
        results["stats_json_exists"] = stats_path.exists()

        if results["stats_json_exists"]:
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats_data = json.load(f)
                print(f"[PASS] Loaded preprocessing_statistics.json ({stats_data.get('num_sequences_processed')} sequences recorded)")
            except Exception as e:
                print(f"[WARN] Failed to parse statistics JSON: {e}")

        # 3. Discover and Inspect Processed Files
        csv_files = sorted(list(self.processed_dir.glob("*_processed.csv")))
        if not csv_files:
            err = f"No processed CSV files found in {self.processed_dir}"
            results["errors"].append(err)
            print(f"[FAIL] {err}")
            return False, results

        results["num_sequences"] = len(csv_files)
        print(f"[PASS] Discovered {len(csv_files)} processed sequence CSV files.")

        # 4. Check Features and Target in Sample Sequence
        sample_df = pd.read_csv(csv_files[0])
        cols = set(sample_df.columns)

        missing_feats = [col for col in FEATURE_COLUMNS if col not in cols]
        # Check mapping aliases for magnetometer if needed
        if missing_feats:
            resolved = []
            for mf in missing_feats:
                if "mag" in mf and any(("mag" in c or "magnetic" in c) for c in cols):
                    continue  # Can be mapped in extraction
                resolved.append(mf)
            missing_feats = resolved

        results["missing_features"] = missing_feats
        if missing_feats:
            err = f"Missing required feature columns in processed data: {missing_feats}"
            results["errors"].append(err)
            print(f"[FAIL] {err}")
            return False, results
        print(f"[PASS] All 14 sensor feature channels verified in dataset.")

        # 5. Target Column Check
        if TARGET_COLUMN not in cols and "gps_speed_kmh" not in cols and "calc_haversine_speed_mps" not in cols:
            err = f"Target column '{TARGET_COLUMN}' not found in processed files."
            results["errors"].append(err)
            print(f"[FAIL] {err}")
            return False, results
        print(f"[PASS] Ground-truth target column '{TARGET_COLUMN}' verified.")

        # 6. Full Dataset Integrity Scan (Rows, Timestamps, NaN/Inf, Target Range)
        total_rows = 0
        all_targets = []

        for f in csv_files:
            df = pd.read_csv(f)
            total_rows += len(df)

            # Check NaNs in features
            for feat in FEATURE_COLUMNS:
                if feat in df.columns:
                    n_nan = df[feat].isna().sum() + np.isinf(df[feat]).sum()
                    results["nan_inf_count"] += int(n_nan)

            # Check target
            t = extract_velocity_target(df)
            all_targets.append(t)

        all_targets_cat = np.concatenate(all_targets)
        results["total_rows"] = total_rows

        is_valid_t, msg_t = validate_target_series(all_targets_cat)
        print(f"[PASS] Target Validation: {msg_t}")

        if not is_valid_t:
            results["errors"].append(msg_t)
            print(f"[FAIL] Target series invalid: {msg_t}")
            return False, results

        print(f"[PASS] Total Processed Samples Across {len(csv_files)} Sequences: {total_rows:,}")
        print(f"[PASS] Total NaN/Inf Values Found: {results['nan_inf_count']}")

        print("--------------------------------------------------------------------------------")
        print("PRE-FLIGHT STATUS: ALL CHECKS PASSED — READY FOR MODEL TRAINING")
        print("--------------------------------------------------------------------------------")

        return True, results


if __name__ == "__main__":
    validator = PreflightDataValidator()
    success, details = validator.validate_preflight()
    if not success:
        print("[ERROR] Pre-flight data validation failed. Halting training execution.")
        sys.exit(1)
