"""
Non-Holonomic Constraints (NHC) Module.
Enforces physical kinematic motion constraints for wheeled land vehicles
(zero lateral velocity v_y = 0 and zero vertical velocity v_z = 0 in vehicle body frame).
"""

import numpy as np


class NonHolonomicConstraintEnforcer:
    """
    Applies Non-Holonomic Constraints (NHC) measurement updates to bound lateral and vertical velocity drift.
    """

    def __init__(self, lateral_std: float = 0.1, vertical_std: float = 0.1) -> None:
        """
        Args:
            lateral_std: Expected standard deviation of lateral motion noise (m/s).
            vertical_std: Expected standard deviation of vertical motion noise (m/s).
        """
        self.R_nhc = np.diag([lateral_std**2, vertical_std**2])

    def get_nhc_measurement(self) -> np.ndarray:
        """
        Get pseudo-observation vector for lateral and vertical body velocities.

        Returns:
            np.ndarray: Pseudo-measurement vector [0.0, 0.0] representing v_y = 0, v_z = 0.
        """
        return np.array([0.0, 0.0], dtype=np.float64)

    def get_nhc_covariance(self) -> np.ndarray:
        """
        Get noise covariance matrix R_nhc for NHC updates.

        Returns:
            np.ndarray: 2x2 Covariance matrix.
        """
        return self.R_nhc.copy()
