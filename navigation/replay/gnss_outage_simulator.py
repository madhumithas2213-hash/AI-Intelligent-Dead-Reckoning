"""
GNSS Outage Simulator & 4-State Navigation State Machine.
Manages automated and interactive GNSS blackout masking and state transitions:
1. GNSS_AIDED    (🟢 GNSS AIDED)
2. GNSS_DEGRADED (🟡 GNSS DEGRADED)
3. DEAD_RECKONING(🔴 DEAD RECKONING)
4. GNSS_RECOVERED(🔵 GNSS RECOVERED)
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np


class GNSSOutageSimulator:
    """
    Simulates GNSS signal outages and maintains navigation state machine.
    """

    def __init__(
        self,
        blackout_start_sec: Optional[float] = None,
        blackout_duration_sec: float = 30.0
    ) -> None:
        """
        Args:
            blackout_start_sec: Optional timestamp in seconds relative to trajectory start to auto-trigger blackout.
            blackout_duration_sec: Duration of blackout in seconds.
        """
        self.blackout_start_sec = blackout_start_sec
        self.blackout_duration_sec = blackout_duration_sec
        self.blackout_end_sec = (blackout_start_sec + blackout_duration_sec) if blackout_start_sec is not None else None

        self.manual_outage_active: bool = False
        self.current_state: str = "GNSS_AIDED"
        self.recovery_window_sec: float = 5.0
        self.recovery_start_time: Optional[float] = None

    def trigger_outage(self) -> None:
        """Manually force GNSS signal outage."""
        self.manual_outage_active = True
        self.current_state = "DEAD_RECKONING"

    def restore_gnss(self, current_timestamp_sec: float = 0.0) -> None:
        """Restore GNSS signal and initiate smooth recovery transition."""
        self.manual_outage_active = False
        self.current_state = "GNSS_RECOVERED"
        self.recovery_start_time = current_timestamp_sec

    def reset(self) -> None:
        """Reset state machine to initial GNSS_AIDED state."""
        self.manual_outage_active = False
        self.current_state = "GNSS_AIDED"
        self.recovery_start_time = None

    def is_outage_active(self, relative_timestamp_sec: float) -> bool:
        """
        Check if GNSS signal should be masked at current relative timestamp.

        Args:
            relative_timestamp_sec: Time in seconds since trajectory start.

        Returns:
            bool: True if GNSS is masked (outage active), False otherwise.
        """
        if self.manual_outage_active:
            return True

        if self.blackout_start_sec is not None and self.blackout_end_sec is not None:
            return self.blackout_start_sec <= relative_timestamp_sec <= self.blackout_end_sec

        return False

    def update_state(
        self,
        relative_timestamp_sec: float,
        gnss_quality: str = "GOOD",
        is_gnss_available: bool = True
    ) -> Dict[str, Any]:
        """
        Update navigation state machine based on GNSS availability, quality, and recovery state.

        Returns:
            Dict[str, Any]: State info dictionary (state_enum, badge_html, display_text, is_masked).
        """
        is_masked = self.is_outage_active(relative_timestamp_sec)

        # Check if recovering
        if self.recovery_start_time is not None:
            elapsed_recovery = relative_timestamp_sec - self.recovery_start_time
            if 0.0 <= elapsed_recovery <= self.recovery_window_sec:
                self.current_state = "GNSS_RECOVERED"
            else:
                self.recovery_start_time = None
                self.current_state = "GNSS_AIDED" if gnss_quality in ["GOOD", "DEGRADED"] else "GNSS_DEGRADED"

        if is_masked or not is_gnss_available or gnss_quality == "LOST":
            self.current_state = "DEAD_RECKONING"
        elif self.current_state != "GNSS_RECOVERED":
            if gnss_quality == "DEGRADED":
                self.current_state = "GNSS_DEGRADED"
            elif gnss_quality == "GOOD":
                self.current_state = "GNSS_AIDED"

        # Format visual badge representations & Real-Time / Offline Mode Indicator
        badge_map = {
            "GNSS_AIDED": {
                "badge": "🟢 GNSS AIDED",
                "class": "badge-gnss",
                "text": "GNSS + INS Aided Navigation Active",
                "connectivity_mode": "ONLINE GNSS",
                "indicator_label": "REAL-TIME / GNSS AVAILABLE",
                "desc": "Live GNSS Assisted • Telemetry Synced with Edge Engine",
                "is_offline": False,
            },
            "GNSS_DEGRADED": {
                "badge": "🟡 GNSS DEGRADED",
                "class": "badge-degraded",
                "text": "GNSS Quality Degraded — Weighting Reduced",
                "connectivity_mode": "DEGRADED GNSS",
                "indicator_label": "GNSS DEGRADED",
                "desc": "Weak Satellite Geometry • Elevating IMU & AI Velocity Weights",
                "is_offline": False,
            },
            "DEAD_RECKONING": {
                "badge": "🔴 DEAD RECKONING",
                "class": "badge-dr",
                "text": "GNSS Signal Lost — AI Dead Reckoning Active",
                "connectivity_mode": "OFFLINE MODE",
                "indicator_label": "OFFLINE / DEAD RECKONING",
                "desc": "No Internet / GNSS Required • Edge AI + IMU Local Inference Active",
                "is_offline": True,
            },
            "GNSS_RECOVERED": {
                "badge": "🔵 GNSS RECOVERED",
                "class": "badge-recovered",
                "text": "GNSS Restored — Performing Gated Smooth Recovery",
                "connectivity_mode": "RECOVERY VALIDATION",
                "indicator_label": "GNSS RECOVERING",
                "desc": "Gated Innovation Verification • Smooth EKF Spatial Re-convergence",
                "is_offline": False,
            },
        }

        info = badge_map.get(self.current_state, badge_map["GNSS_AIDED"])
        return {
            "state": self.current_state,
            "badge": info["badge"],
            "class": info["class"],
            "text": info["text"],
            "connectivity_mode": info["connectivity_mode"],
            "indicator_label": info["indicator_label"],
            "desc": info["desc"],
            "is_offline": info["is_offline"],
            "tooltip_explanation": "Navigation continues locally when GNSS or internet connectivity is unavailable.",
            "is_masked": is_masked,
        }
