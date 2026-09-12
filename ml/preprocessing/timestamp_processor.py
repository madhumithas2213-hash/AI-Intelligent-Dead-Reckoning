"""
Timestamp Processor Module.
Parses, validates, sorts, and analyzes time-series telemetry streams from IO-VNBD.
Detects duplicate timestamps, missing intervals, irregular sampling, and temporal gaps
without deleting rows with irregular spacing.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class TimestampStatistics:
    """Statistical summary of timestamp intervals for a sequence."""
    total_samples: int
    num_duplicates: int
    num_temporal_gaps: int  # Gaps > 5000ms
    mean_dt_ms: float
    median_dt_ms: float
    std_dt_ms: float
    min_dt_ms: float
    max_dt_ms: float
    pct_intervals_close_to_100ms: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class TimestampProcessor:
    """
    Processes and analyzes sequence time-series timestamps.
    """

    def __init__(self, gap_threshold_ms: float = 5000.0, nominal_dt_ms: float = 100.0) -> None:
        """
        Args:
            gap_threshold_ms: Threshold in ms to classify a temporal gap (default: 5000ms = 5s).
            nominal_dt_ms: Expected nominal sampling interval in ms (default: 100ms = 10Hz).
        """
        self.gap_threshold_ms = gap_threshold_ms
        self.nominal_dt_ms = nominal_dt_ms

    def process_sequence_timestamps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, TimestampStatistics]:
        """
        Parse, sort, calculate relative time basis, and compute interval statistics.

        Args:
            df: Normalized DataFrame containing 'time_since_start_ms' or 'date_timestamp_str'.

        Returns:
            Tuple[pd.DataFrame, TimestampStatistics]: (Processed DataFrame with timestamp_sec & dt_ms, stats).
        """
        df_processed = df.copy()

        # Determine best timestamp source
        if "time_since_start_ms" in df_processed.columns:
            ts_ms = pd.to_numeric(df_processed["time_since_start_ms"], errors="coerce").to_numpy(dtype=np.float64)
        elif "date_timestamp_str" in df_processed.columns:
            # Parse datetime string
            dt_series = pd.to_datetime(df_processed["date_timestamp_str"], errors="coerce")
            ts_ms = (dt_series.view("int64") // 10**6).to_numpy(dtype=np.float64)
        else:
            raise ValueError("DataFrame contains no recognized timestamp column.")

        df_processed["_raw_ts_ms"] = ts_ms

        # Sort chronologically by timestamp
        df_processed = df_processed.sort_values(by="_raw_ts_ms").reset_index(drop=True)
        ts_sorted = df_processed["_raw_ts_ms"].to_numpy()

        # Compute interval delta dt in ms
        dt_ms = np.zeros_like(ts_sorted)
        dt_ms[1:] = np.diff(ts_sorted)
        dt_ms[0] = self.nominal_dt_ms  # Default first timestep delta to nominal

        df_processed["dt_ms"] = dt_ms

        # Compute normalized relative timestamp in seconds starting from 0.0
        t_start = ts_sorted[0]
        df_processed["timestamp_sec"] = (ts_sorted - t_start) / 1000.0

        # Calculate Statistics
        total_samples = len(ts_sorted)
        num_duplicates = int(np.sum(dt_ms[1:] == 0))
        num_temporal_gaps = int(np.sum(dt_ms > self.gap_threshold_ms))

        valid_dt = dt_ms[1:] if len(dt_ms) > 1 else np.array([self.nominal_dt_ms])

        mean_dt = float(np.mean(valid_dt))
        median_dt = float(np.median(valid_dt))
        std_dt = float(np.std(valid_dt))
        min_dt = float(np.min(valid_dt))
        max_dt = float(np.max(valid_dt))

        # Percentage of intervals close to 100 ms (between 90ms and 110ms)
        close_to_100ms = np.logical_and(valid_dt >= 90.0, valid_dt <= 110.0)
        pct_100ms = float(np.mean(close_to_100ms) * 100.0)

        stats = TimestampStatistics(
            total_samples=total_samples,
            num_duplicates=num_duplicates,
            num_temporal_gaps=num_temporal_gaps,
            mean_dt_ms=mean_dt,
            median_dt_ms=median_dt,
            std_dt_ms=std_dt,
            min_dt_ms=min_dt,
            max_dt_ms=max_dt,
            pct_intervals_close_to_100ms=pct_100ms,
        )

        return df_processed, stats
