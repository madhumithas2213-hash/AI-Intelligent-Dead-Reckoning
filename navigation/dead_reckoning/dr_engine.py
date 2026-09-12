"""
3D Kinematic Dead Reckoning Engine.
Integrates forward velocity estimates and turn rates (gyro/mag) to maintain 3D position and orientation state.
"""

import numpy as np


class DeadReckoningEngine:
    """
    Maintains vehicle 2D/3D navigation state: position (East, North, Up),
    velocity vector, and heading/orientation angle.
    """

    def __init__(self, init_pos: np.ndarray = None, init_heading: float = 0.0) -> None:
        """
        Args:
            init_pos: Initial position vector [East, North, Up] in meters (or lat/lon reference offset).
            init_heading: Initial yaw angle in radians (0 = East/North relative).
        """
        self.position: np.ndarray = init_pos if init_pos is not None else np.zeros(3)
        self.heading: float = init_heading
        self.velocity: np.ndarray = np.zeros(3)

    def predict_step(self, forward_speed: float, yaw_rate: float, dt: float) -> np.ndarray:
        """
        Propagate dead reckoning state by one time interval dt.

        Args:
            forward_speed: 1D vehicle forward velocity in m/s (e.g. from GRU regressor).
            yaw_rate: Vehicle angular rate about vertical axis in rad/s (from gyroscope).
            dt: Integration time step in seconds (e.g. 0.01s for 100Hz).

        Returns:
            np.ndarray: Updated 3D position vector [East, North, Up].
        """
        # Update heading angle
        self.heading += yaw_rate * dt
        # Normalize heading to [-pi, pi]
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        # Compute velocity components in navigation frame (East-North-Up)
        vel_east = forward_speed * np.cos(self.heading)
        vel_north = forward_speed * np.sin(self.heading)

        self.velocity[0] = vel_east
        self.velocity[1] = vel_north

        # Integrate position
        self.position[0] += vel_east * dt
        self.position[1] += vel_north * dt

        return self.position.copy()

    def reset_state(self, new_pos: np.ndarray, new_heading: float) -> None:
        """
        Reset engine state using an authoritative fix (e.g. GNSS update or Map Match fix).

        Args:
            new_pos: Corrected position vector.
            new_heading: Corrected yaw angle in radians.
        """
        self.position = new_pos.copy()
        self.heading = new_heading
