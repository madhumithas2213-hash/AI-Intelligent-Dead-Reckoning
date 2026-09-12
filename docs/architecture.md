# System Architecture Specification – Intelligent Dead Reckoning (IDR)

---

## 1. High-Level Modular Design

The **SIH26168 Intelligent Dead Reckoning** system fuses edge deep learning estimators with classic state-space inertial navigation equations and offline GIS map matching.

```
       ┌─────────────────────────────────────────────────────────┐
       │                Smartphone Sensor Array                  │
       │   (Accel 100Hz | Gyro 100Hz | Mag 50Hz | GNSS 1Hz)      │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │          Navigation & Denoising Pipeline                │
       │   - Denoising & Downsampling (Cleaner)                  │
       │   - Automatic Phone-to-Vehicle Alignment (R_p2v)       │
       └────────────────────────────┬────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
       ┌──────────────────────┐           ┌──────────────────────┐
       │ Edge Deep Learning   │           │ Strapdown Kinematics │
       │  (ONNX Runtime)      │           │ (Dead Reckoning)     │
       │  - 1D CNN Classifier │           │  - 3D Position Integr│
       │  - GRU Regressor (v) │           │  - NHC (vy=0, vz=0)  │
       │  - Context MLP (R,Q) │           └──────────┬───────────┘
       └──────────┬───────────┘                      │
                  │                                  │
                  └─────────────────┬────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │            Adaptive EKF / UKF Sensor Fusion             │
       │  State: [px, py, pz, vx, vy, vz, roll, pitch, yaw]      │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │            Offline Map Matcher (HMM/OSM)                │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │           Real-time Navigation UI (Android)             │
       └─────────────────────────────────────────────────────────┘
```

---

## 2. Core Functional Modules

### Module 1: Sensor Acquisition (`android/`)
Receives raw 100Hz accelerometer ($a_p$), gyroscope ($\omega_p$), magnetometer ($m_p$), and 1Hz GNSS location fixes ($p_{gnss}, v_{gnss}$).

### Module 2: Phone-to-Vehicle Alignment (`navigation/alignment/`)
Transforms phone IMU frame $\{P\}$ into vehicle body frame $\{V\}$:
$$\mathbf{a}_v = \mathbf{R}_{p2v} \mathbf{a}_p, \quad \boldsymbol{\omega}_v = \mathbf{R}_{p2v} \boldsymbol{\omega}_p$$

### Module 3: 1D CNN Motion/Disturbance Classifier (`ml/models/cnn_classifier.py`)
Predicts vehicle motion context $C_k \in \{\text{Stationary}, \text{Straight Smooth}, \text{Curved Road}, \text{Bumpy Surface}\}$.

### Module 4: GRU Vehicle Speed Estimator (`ml/models/gru_velocity.py`)
Regresses scalar forward velocity $v_x(t)$ from windowed IMU temporal sequences, bypassing accelerometer drift integration.

### Module 5: Context/Confidence MLP (`ml/models/confidence_mlp.py`)
Outputs continuous measurement noise covariance scaling factors ($R_{scale}, Q_{scale}$) according to GNSS dilution of precision (DOP) and IMU disturbance probabilities.

### Module 6: Adaptive EKF/UKF (`navigation/sensor_fusion/`)
Combines strapdown propagation equations with measurement updates from GNSS, ML speed estimates, and Non-Holonomic Constraints (NHC).

### Module 7: Map Matching (`navigation/map_matching/`)
Snaps filtered trajectory states to valid OSM road network centerlines during prolonged GNSS outages using Viterbi optimization.
