"""
Adaptive Extended Kalman Filter (EKF) Engine Wrapper.
Re-exports AdaptiveEKF from .ekf for backward compatibility.
"""

from navigation.sensor_fusion.ekf import AdaptiveEKF

__all__ = ["AdaptiveEKF"]
