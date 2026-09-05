# Dataset Management & Verification Report – IO-VNBD

This document outlines the verified dataset structure, channel specifications, sampling frequencies, and suitability matrix for the **SIH26168 Intelligent Dead Reckoning (IDR)** project based on the **IO-VNBD (Inertial Odometry Vehicle & Smartphone Navigation Benchmark Dataset)**.

---

## 📌 IO-VNBD Dataset Discovered Structure

The dataset contains **731 files** structured into two main sub-folders:

1. **`Synchronised V abd S datasets/`**:
   - `Categorised IOVNB Dataset/`: Categorized by driver (`Driver A` through `Driver E`) and trajectory routes (`Vta`, `Vf`, `S1`..`S4`).
   - `Uncategorised IOVNB Dataset/`: Paired synchronized Smartphone (`S`) and Vehicle CAN/GNSS (`V`) trajectories.

2. **`Unsynchronised V and S Dataset/`**:
   - `Categorised IOVNB (V) Dataset/`: Raw vehicle-side CAN/OBD telemetry recordings.
   - `Uncategorised IOVNB (V and S) Dataset/`: Unsynchronized smartphone (`S-Dataset/`) and vehicle (`V-Dataset/`) files.

---

## 📊 Sensor Channels & Discovered Schema

### 📱 Smartphone Sensor Streams (`S-Dataset` / `S-*.csv`)
- **Total Columns**: 24
- **Discovered Column Names**:
  1. `GPS LATITUDE (degrees)`
  2. ` GPS LONGITUDE (degrees)`
  3. ` GPS ALTITUDE (m)`
  4. ` GPS SPEED (Kmh)`
  5. ` GPS ACCURACY (m)`
  6. ` GPS ORIENTATION (°)`
  7. `GPS SATELLITES IN RANGE`
  8. ` TIME SINCE START (ms)`
  9. ` DATE (YYYY-MO-DD HH-MI-SS_SSS)`
  10. ` ACCELEROMETER X (m/s²)`
  11. ` ACCELEROMETER Y (m/s²)`
  12. ` ACCELEROMETER Z (m/s²)`
  13. ` GRAVITY X (m/s²)`
  14. ` GRAVITY Y (m/s²)`
  15. ` GRAVITY Z (m/s²)`
  16. ` GYROSCOPE Yaw (rad/s)`
  17. ` GYROSCOPE Pitch (rad/s)`
  18. ` GYROSCOPE Roll (rad/s)`
  19. ` MAGNETIC FIELD X (μT)`
  20. ` MAGNETIC FIELD Y (μT)`
  21. ` MAGNETIC FIELD Z (μT)`
  22. ` ORIENTATION (Yaw) (°)`
  23. ` ORIENTATION (Pitch) (°)`
  24. ` ORIENTATION (Roll ) (°)`

- **Sampling Rate**: Median $\Delta t = 100\text{ ms} \rightarrow 10.0\text{ Hz}$ (resampled to $100\text{ Hz}$ via cubic spline in `ml/preprocessing/importer.py`).

---

### 🚘 Vehicle CAN / Reference Sensor Streams (`V-Dataset` / `V-*.csv`)
- **Total Columns**: 29
- **Discovered Column Names**:
  1. `No of GPS Satellites Available`
  2. ` Time Since Start of Day (seconds)`
  3. ` Latitude (degrees)`
  4. ` Longitude (degrees)`
  5. ` Velocity (km/hr)`
  6. ` Heading (degrees)`
  7. ` Height (km)`
  8. ` Vertical velocity (km/hr)`
  9. ` Sample period (seconds)`
  10. ` Steering Angle (degrees)`
  11. ` Wheel Speed Front Left (rad/sec)`
  12. ` Wheel Speed Front Right (rad/sec)`
  13. ` Wheel Speed Rear Left (rad/sec)`
  14. ` Wheel Speed Rear Right (rad/sec)`
  15. ` Yaw Rate (deg/sec)`
  16. ` Indicated Vehicle Speed (km/hr)`
  17. ` Indicated Longitudinal Acceleration (g)`
  18. ` Indicated Lateral Acceleration (g)`
  19. ` Handbrake (0 or 1)`
  20. ` Gear Requested (Number fof gear employed 1-5)`
  21. ` Gear (Number fof gear employed 1-5)`
  22. ` Engine Speed (rev/min)`
  23. ` Coolant Temperature (degrees)`
  24. ` Clutch Position (0 or 1)`
  25. ` Brake Pressure (psi)`
  26. ` Brake Position (0 or 1)`
  27. ` Battery Voltage (volts)`
  28. ` Air Temperature (degrees)`
  29. ` Accelerator Pedal Position (0 or 1)`

---

## 🎯 Role Mapping Matrix for SIH26168

| Data Source | Feature Columns | Pipeline Role |
| :--- | :--- | :--- |
| **Smartphone IMU** | `ACCELEROMETER X/Y/Z`, `GYROSCOPE Yaw/Pitch/Roll` | Input to `1D CNN Classifier`, `GRU Velocity Regressor`, `Dead Reckoning Engine` |
| **Smartphone Orientation** | `GRAVITY X/Y/Z`, `ORIENTATION (Yaw/Pitch/Roll)` | Input to `PhoneVehicleAligner` ($\mathbf{R}_{p2v}$) |
| **Smartphone Magnetometer** | `MAGNETIC FIELD X/Y/Z` | Heading reference update in EKF/UKF |
| **Smartphone GNSS** | `GPS LATITUDE`, `GPS LONGITUDE`, `GPS SPEED`, `GPS ACCURACY` | Observation update in Adaptive EKF/UKF |
| **Vehicle Wheel Speed** | `Wheel Speed Front/Rear (rad/sec)`, `Indicated Vehicle Speed` | Target label for GRU speed regressor ($v_x$) |
| **Vehicle Reference Fix** | `Latitude (degrees)`, `Longitude (degrees)`, `Velocity` | Ground truth reference for ATE / RPE trajectory evaluation |

---

## 📋 Final Dataset Status Verification

```
DATASET STATUS:
- Dataset found: YES
- Smartphone IMU: YES
- Smartphone GNSS: YES
- Vehicle IMU: YES
- Odometry: YES
- Ground truth/reference: YES
- Sampling rates verified: YES
- Suitable for SIH26168: YES
```
