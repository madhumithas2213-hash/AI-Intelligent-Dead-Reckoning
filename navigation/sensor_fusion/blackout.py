"""
GNSS Blackout Evaluator Module.
Simulates GNSS signal blackouts of configurable duration in memory without modifying raw/processed dataset files,
and records position drift, outage duration, and recovery behavior metrics.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


class BlackoutEvaluator:
    """
    Simulates GNSS blackouts during dataset trajectory playback and tracks dead-reckoning drift.
    """

    def __init__(
        self,
        blackout_start_sec: float = 30.0,
        blackout_duration_sec: float = 30.0
    ) -> None:
        """
        Args:
            blackout_start_sec: Timestamp in seconds relative to trajectory start to begin blackout.
            blackout_duration_sec: Duration of simulated GNSS blackout in seconds.
        """
        self.blackout_start_sec = blackout_start_sec
        self.blackout_duration_sec = blackout_duration_sec
        self.blackout_end_sec = blackout_start_sec + blackout_duration_sec

    def is_in_blackout(self, relative_timestamp_sec: float) -> bool:
        """
        Check if the given relative timestamp falls within the simulated blackout window.

        Args:
            relative_timestamp_sec: Time in seconds since trajectory start.

        Returns:
            bool: True if inside blackout interval, False otherwise.
        """
        return self.blackout_start_sec <= relative_timestamp_sec <= self.blackout_end_sec

    def create_blackout_mask(self, df: pd.DataFrame, time_col: str = "timestamp_sec") -> np.ndarray:
        """
        Generate a boolean mask for DataFrame rows where GNSS is available.
        True = GNSS active, False = GNSS blackout.

        Args:
            df: Trajectory DataFrame.
            time_col: Name of timestamp column in seconds.

        Returns:
            np.ndarray: Boolean array of shape [len(df)].
        """
        if time_col not in df.columns:
            if "time_since_start_ms" in df.columns:
                ts = df["time_since_start_ms"].to_numpy() / 1000.0
            else:
                ts = np.arange(len(df)) * 0.01
        else:
            ts = df[time_col].to_numpy()

        t0 = ts[0]
        rel_t = ts - t0

        # Mask is True when GNSS is available (OUTSIDE blackout window)
        mask = ~((rel_t >= self.blackout_start_sec) & (rel_t <= self.blackout_end_sec))
        return mask
