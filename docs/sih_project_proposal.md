# SIH26168 Project Specification & Proposal

---

## 🎯 Problem Statement
**Title**: AI-ML based Intelligent Dead Reckoning System for Seamless Navigation
**ID**: SIH26168
**Domain**: AI/ML, Edge Computing, Navigation Systems

---

## 💡 Key Challenges Addressed
1. **GNSS Outages**: Satellite signal loss in tunnels, urban canyons, flyovers, and subterranean passes causes standard navigation apps to freeze or drift wildly.
2. **Inertial Sensor Noise & Drift**: Pure double-integration of raw smartphone accelerometer data leads to exponential position error accumulation within seconds.
3. **Unconstrained Phone Placement**: Users mount, hold, or place smartphones arbitrarily inside vehicles.
4. **Edge Computational Constraints**: Solutions must run lightweight on low-power mobile processors without requiring cloud connectivity.

---

## 🚀 Novel Solution Design

1. **Self-Calibrating Coordinate Frame Alignment**:
   - Automatically computes pitch, roll, and yaw offset matrices between phone and vehicle frames.
2. **Hybrid Physics-ML Architecture**:
   - Replaces traditional double-integration velocity calculation with a temporal GRU regressor trained on multi-modal driving datasets.
3. **Non-Holonomic Kinematic Constraints (NHC)**:
   - Zero lateral ($v_y=0$) and zero vertical ($v_z=0$) velocity assumptions bound trajectory lateral drift.
4. **Adaptive Noise Covariance EKF**:
   - Neural network MLP dynamically inflates/deflates measurement covariance $R$ during GNSS degradation.
5. **Offline Map Matching**:
   - Hidden Markov Model (HMM) constrained to local OpenStreetMap graphs guarantees on-road map matching during extended satellite outages.
