# Android Client Application – Intelligent Dead Reckoning (IDR)

This module contains the **Android Studio** project codebase for real-time mobile data acquisition, on-device ONNX ML inference, dead reckoning navigation processing, and real-time map rendering.

---

## 📱 Mobile Application Architecture

```
[ Android Smartphone Sensors ]
  ├── Accelerometer (100Hz) ──┐
  ├── Gyroscope (100Hz)     ──┼──> [ SensorService (ForegroundService) ]
  ├── Magnetometer (50Hz)   ──┤               │
  └── GNSS Receiver (1Hz)   ──┘               ▼
                                   [ IDRInferenceEngine (ONNX Runtime) ]
                                                  │
                                                  ▼
                                    [ AdaptiveFusionEKF State Estimator ]
                                                  │
                                                  ▼
                                     [ MainActivity Navigation HUD ]
```

1. **Background High-Frequency Sensor Service** (`SensorService.kt`):
   - Acquires 100Hz `Sensor.TYPE_ACCELEROMETER`, `Sensor.TYPE_GYROSCOPE`, and `Sensor.TYPE_MAGNETOMETER` events.
   - Listens for `LocationListener` GNSS location fixes.
   - Implements Android `ForegroundService` with persistent ongoing notification to prevent OS background process throttling.

2. **On-Device ONNX Runtime Integration** (`IDRInferenceEngine.kt`):
   - Executes converted PyTorch ONNX models (`cnn_classifier.onnx`, `gru_velocity.onnx`, `confidence_mlp.onnx`) directly on Android CPU/NNAPI NPU.
   - Low latency forward pass (< 15ms execution window).

3. **Navigation Engine & Map Visualization** (`AdaptiveFusionEKF.kt` & `MainActivity.kt`):
   - Runs the adaptive Extended Kalman Filter state estimator.
   - Renders live position polyline and heading arrow on offline vector maps (OsmDroid).
   - Dynamic mode transition handling: `GNSS-AIDED` → `GNSS-DEGRADED` → `DEAD RECKONING` → `GNSS-RECOVERED` → `GNSS-AIDED`.

---

## 🛠 Android Studio Project Structure

- `app/src/main/java/com/aiidr/navigation/MainActivity.kt`
- `app/src/main/java/com/aiidr/navigation/SensorService.kt`
- `app/src/main/java/com/aiidr/navigation/IDRInferenceEngine.kt`
- `app/src/main/java/com/aiidr/navigation/AdaptiveFusionEKF.kt`
- `app/src/main/res/layout/activity_main.xml`
- `app/src/main/AndroidManifest.xml`
- `app/build.gradle.kts`
- `build.gradle.kts`

---

## ⚙️ Minimum System Requirements

- **IDE**: Android Studio Hedgehog (2023.1.1) or newer
- **Language**: Kotlin 1.9+
- **Min SDK**: API 26 (Android 8.0 Oreo)
- **Target SDK**: API 34 (Android 14)
- **Dependencies**:
  - `com.microsoft.onnxruntime:onnxruntime-android:1.16.3`
  - `org.osmdroid:osmdroid-android:6.1.16`
  - `androidx.core:core-ktx:1.12.0`
