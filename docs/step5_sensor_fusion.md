# Step 5 — Adaptive GNSS + INS Sensor Fusion Engine Technical Documentation

**Project**: SIH26168 — AI/ML Based Intelligent Dead Reckoning  
**Module**: Adaptive EKF Sensor Fusion (`navigation/sensor_fusion/`)  
**Status**: Completed & Verified  

---

## 1. System Architecture

The Adaptive GNSS + INS Sensor Fusion Engine combines smartphone IMU observations (accelerometer, gyroscope, magnetometer), vehicle-frame aligned IMU telemetry from Step 4, and GNSS satellite fixes to maintain continuous, high-precision vehicle position, velocity, and orientation estimates.

```
                    ┌─────────────────────────┐
                    │   Smartphone IMU Data   │
                    │ (Accel, Gyro, Mag, dt)  │
                    └────────────┬────────────┘
                                 │
                                 ↓
                    ┌─────────────────────────┐
                    │ Phone-to-Vehicle Frame  │
                    │   Alignment (Step 4)    │
                    └────────────┬────────────┘
                                 │
                                 ↓
┌─────────────────┐ ┌─────────────────────────┐
│  GNSS Telemetry │ │   Strapdown Prediction  │
│ (Lat, Lon, Spd, │ │  State & Covariance P   │
│  Acc, Sats, t)  │ └────────────┬────────────┘
└────────┬────────┘              │
         │                       │
         ↓                       │
┌─────────────────┐              │
│ Adaptive GNSS   │              │
│ Quality & Gating│              │
└────────┬────────┘              │
         │                       │
         ↓                       ↓
┌────────────────────────────────────────────────┐
│    Adaptive Extended Kalman Filter (EKF)       │
│  State: [px, py, vx, vy, yaw, bax, bay, bgz]   │
│  Constraints: Soft Non-Holonomic (NHC v_lat=0) │
└───────────────────────┬────────────────────────┘
                        │
                        ↓
┌────────────────────────────────────────────────┐
│             Navigation State Output            │
│ (lat, lon, vel, heading, accel, mode, P, conf) │
└────────────────────────────────────────────────┘
```

---

## 2. Navigation State Vector

The system maintains an 8-dimensional continuous state vector:

$$x = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \\ \psi \\ b_{ax} \\ b_{ay} \\ b_{\omega z} \end{bmatrix}$$

| Index | State Symbol | Description | Units |
| :--- | :--- | :--- | :--- |
| `0` | $p_x$ | Local East position relative to reference origin | Meters (m) |
| `1` | $p_y$ | Local North position relative to reference origin | Meters (m) |
| `2` | $v_x$ | Local East velocity component | m/s |
| `3` | $v_y$ | Local North velocity component | m/s |
| `4` | $\psi$ | Vehicle heading angle (0 = North, clockwise) | Radians |
| `5` | $b_{ax}$ | Estimated longitudinal accelerometer bias | $\text{m/s}^2$ |
| `6` | $b_{ay}$ | Estimated lateral accelerometer bias | $\text{m/s}^2$ |
| `7` | $b_{\omega z}$ | Estimated yaw rate gyroscope bias | rad/s |

---

## 3. Strapdown Prediction Step

At each IMU step $\Delta t$:
1. De-bias raw aligned acceleration and gyro readings:
   $$a_{\text{fwd}} = a_x - b_{ax}$$
   $$\omega_z = g_z - b_{\omega z}$$

2. Update heading angle $\psi$:
   $$\psi_k = \psi_{k-1} + \omega_z \cdot \Delta t$$

3. Propagate velocity vector in Local East-North-Up (ENU) frame:
   $$v_{x, k} = v_{\text{fwd}} \sin \psi_k$$
   $$v_{y, k} = v_{\text{fwd}} \cos \psi_k$$

4. Propagate local position:
   $$p_{x, k} = p_{x, k-1} + v_{x, k} \cdot \Delta t + \frac{1}{2} a_{\text{fwd}} \sin \psi_k \cdot \Delta t^2$$
   $$p_{y, k} = p_{y, k-1} + v_{y, k} \cdot \Delta t + \frac{1}{2} a_{\text{fwd}} \cos \psi_k \cdot \Delta t^2$$

5. Propagate error covariance $P$ using the State Transition Jacobian $F$:
   $$P_{k|k-1} = F P_{k-1|k-1} F^T + Q \cdot \Delta t$$

---

## 4. Adaptive GNSS Measurement Update

When GNSS fix $(lat, lon, speed, accuracy, satellites)$ arrives:

1. Convert WGS84 Geodetic coordinates $(lat, lon)$ into Local ENU metric coordinates $(z_{e}, z_{n})$ relative to $(lat_0, lon_0)$:
   $$z_e = R_E \cdot (\text{lon} - \text{lon}_0) \cdot \frac{\pi}{180} \cdot \cos(\text{lat}_0 \cdot \frac{\pi}{180})$$
   $$z_n = R_E \cdot (\text{lat} - \text{lat}_0) \cdot \frac{\pi}{180}$$

2. Calculate innovation residual $y = z - Hx$.
3. Perform Mahalanobis distance gating to filter multipath jumps.
4. Scale measurement noise covariance $R = R_0 \cdot r_{\text{scale}}$ adaptively based on reported GNSS accuracy and satellite counts.
5. Perform state update with Joseph-form error covariance stabilization:
   $$K = P H^T (H P H^T + R)^{-1}$$
   $$x = x + K y$$
   $$P = (I - KH) P (I - KH)^T + K R K^T$$

---

## 5. GNSS Quality & Outage Management

GNSS quality is dynamically classified into four operational states:
- `GOOD`: $\text{Accuracy} \le 5.0\text{m}$, Satellites $\ge 6$, low innovation residual.
- `DEGRADED`: $5.0\text{m} < \text{Accuracy} \le 15.0\text{m}$, Satellites 4–5.
- `UNRELIABLE`: $\text{Accuracy} > 15.0\text{m}$, Satellites $< 4$, NaN values, or Mahalanobis gating failure.
- `LOST`: Elapsed time since last GNSS fix $> \text{GNSS\_TIMEOUT\_SECONDS}$ (default 2.0s).

When GNSS is lost or unreliable, the engine automatically switches to **`DEAD_RECKONING`** mode and maintains state using strapdown IMU integration and soft kinematic constraints.

When GNSS returns, the innovation is gated and Kalman gain is applied smoothly to prevent sudden position teleportation.

---

## 6. Non-Holonomic Kinematic Constraints (NHC)

For wheeled ground vehicles, lateral body velocity $v_{\text{lat}} \approx 0$.
The engine applies a soft NHC measurement update:
$$v_{\text{lat}} = -v_x \cos\psi + v_y \sin\psi$$
$$y_{\text{nhc}} = 0 - v_{\text{lat}}$$
This bounds lateral drift during prolonged dead reckoning intervals.

---

## 7. Edge & Mobile Deployment Considerations

- **Dependencies**: Relies strictly on standard NumPy and SciPy.
- **Computational Overhead**: Execution time per IMU step is $< 0.1$ ms on standard mobile CPUs.
- **Memory Footprint**: State vector (8 elements) and covariance (8x8) consume $< 2$ KB.
