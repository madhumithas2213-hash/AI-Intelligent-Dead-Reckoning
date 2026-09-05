# Android Client Application – Intelligent Dead Reckoning (IDR)

This module contains the **Android Studio** project codebase for real-time mobile data acquisition, on-device ONNX ML inference, dead reckoning navigation processing, and real-time map rendering.

---

## 📱 Mobile Application Features

1. **Background High-Frequency Sensor Service**:
   - Acquires 100Hz `Sensor.TYPE_ACCELEROMETER`, `Sensor.TYPE_GYROSCOPE`, and `Sensor.TYPE_MAGNETOMETER` events.
   - Listens for `LocationListener` GNSS location fixes.
   - Implements Android `ForegroundService` with persistent ongoing notification to prevent OS background process throttling.

2. **On-Device ONNX Runtime Integration**:
   - Executes converted PyTorch ONNX models (`cnn_classifier.onnx`, `gru_velocity.onnx`, `confidence_mlp.onnx`) directly on Android CPU/NNAPI NPU.
   - Low latency forward pass (< 15ms execution window).

3. **Navigation Engine & Map Visualization**:
   - Runs the adaptive Extended Kalman Filter state estimator.
   - Renders live position polyline and heading arrow on offline vector maps (Mapbox SDK / OsmDroid).
   - Indicates GNSS connection state (Active Satellite Fix vs. Inertial Dead Reckoning Mode).

---

## 🛠 Recommended Android Studio Setup

- **IDE**: Android Studio Hedgehog (2023.1.1) or newer
- **Language**: Kotlin 1.9+
- **Min SDK**: API 26 (Android 8.0 Oreo)
- **Target SDK**: API 34 (Android 14)
- **Dependencies**:
  - `com.microsoft.onnxruntime:onnxruntime-android:1.15.0`
  - `org.osmdroid:osmdroid-android:6.1.16`
  - `androidx.core:core-ktx:1.12.0`
