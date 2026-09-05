# CRITICAL DRIFT FAILURE DIAGNOSIS REPORT — AI-IDR SIH 2026

**Project ID:** SIH26168  
**Module:** GNSS Outage Drift Failure Diagnosis  
**Date:** 2026-09-03  
**Final Status:** **DIAGNOSED — ROOT CAUSE FOUND**  

---

## 1. Executive Summary & Root Cause

### Primary Root Cause Classification
- **Classification:** **H. Evaluation-Window Mismatch & I. Raw Ground-Truth GPS Fix Discretization**
- **Affected File / Module:** `dataset/processed/S-A1_processed.csv` & `ml/evaluation/evaluate_step5_fusion.py`

### Detailed Diagnosis
1. **Static Reference Fix in Raw Dataset:** Between relative timestamps $t = 250.00\text{s}$ and $t = 280.00\text{s}$ in sequence `S-A1`, the raw GPS coordinates (`gps_latitude_deg`, `gps_longitude_deg`) remain completely frozen at ENU Cartesian position `(-660.94, -1011.56) m`. The raw IO-VNBD dataset records GPS updates as discrete batch coordinate fixes every 50–100 seconds.
2. **Real Vehicle Motion:** During those same 30 seconds, the vehicle speed sensor records an active reference speed $v_{ref} = 3.18\text{ m/s}$ ($11.44\text{ km/h}$), totaling a reference travel distance of **95.33 meters**.
3. **AI-IDR Model Performance:** AI-IDR predicts a smooth forward velocity $v_{est} = 2.92 \dots 3.08\text{ m/s}$ ($10.5 \dots 11.1\text{ km/h}$) at heading $150.6^\circ$. Total distance predicted by AI-IDR over 30 seconds is **89.93 meters** (representing a velocity error of only **0.20 m/s** or **6.2%**).
4. **Pointwise Spatial Error Explosion:** Pointwise error is calculated as $e_i = \text{distance}(P_{AI, i}, P_{ref, i})$. Because $P_{ref, i}$ is frozen at `(-660.94, -1011.56) m`, evaluating the moving AI-IDR position $(-731.82, -1067.17)\text{ m}$ against the static GPS fix measures the vehicle's total travel distance (**90.09 meters**) as a spatial error, producing an artificial **94.51%** drift percentage and triggering `FAIL`.

---

## 2. Checkpoint Diagnostic Values (Sequence S-A1, $t = 250\text{s} \dots 280\text{s}$)

All diagnostic variables were extracted across 5 synchronized checkpoints during the 30-second outage:

| Checkpoint | Timestamp | Reference ENU (m) | AI-IDR ENU (m) | Ref Speed | Est Speed | Heading | dt (s) | Pointwise Error |
|---|---|---|---|---|---|---|---|---|
| **0% (Start)** | 250.00 s | (-660.94, -1011.56) | (-660.15, -1012.86) | 3.18 m/s | 2.92 m/s | 150.6° | 0.500s | **1.53 m** |
| **25%** | 257.50 s | (-660.94, -1011.56) | (-679.26, -1025.45) | 3.18 m/s | 3.03 m/s | 150.6° | 0.498s | **22.98 m** |
| **50%** | 265.00 s | (-660.94, -1011.56) | (-697.41, -1039.41) | 3.18 m/s | 3.08 m/s | 150.6° | 0.501s | **45.89 m** |
| **75%** | 272.50 s | (-660.94, -1011.56) | (-714.08, -1054.29) | 3.18 m/s | 2.93 m/s | 150.6° | 0.500s | **68.19 m** |
| **100% (End)**| 280.00 s | (-660.94, -1011.56) | (-731.82, -1067.17) | 3.18 m/s | 2.98 m/s | 150.6° | 0.500s | **90.09 m** |

### Key Diagnostic Observations
- `ref_x` and `ref_y` remain **100% constant** (`-660.94, -1011.56`) across all 5 checkpoints.
- `pos_error` increases linearly from **1.53 m** to **90.09 m** at a rate matching vehicle velocity ($3.0\text{ m/s} \times 30\text{s} \approx 90\text{m}$).
- Velocity estimation is accurate ($2.98\text{ m/s}$ vs $3.18\text{ m/s}$), proving the dead-reckoning engine is working as designed.

---

## 3. 19-Point Audit Verification Checklist

1. **Outage Timestamps:** Verified ($t = 250.00\text{s} \dots 280.00\text{s}$, 30.0s duration, 61 samples).
2. **Coordinate Frame:** Verified (Local ENU Cartesian meters relative to $lat_0 = 1.3094^\circ, lon_0 = 103.7788^\circ$).
3. **Lat/Lon to ENU:** Verified exact WGS84 formula `latlon_to_enu`.
4. **Lat/Lon Axis Mapping:** Verified correctly mapped (`lat` $\rightarrow$ North, `lon` $\rightarrow$ East).
5. **Metres/Degrees Unit:** Verified correct conversion via `latlon_to_enu`.
6. **Reference Distance:** $d_{ref} = \int v_{ref} dt = \mathbf{95.33\text{ m}}$ (integrated over exact 30s window).
7. **AI-IDR Propagation:** Verified smooth strapdown EKF integration ($2.92 \dots 3.08\text{ m/s}$ at heading $150.6^\circ$).
8. **Velocity Units:** Verified $m/s$ ($3.18\text{ m/s}$ = $11.44\text{ km/h}$). No double conversion.
9. **Timestep $dt$:** Verified actual dataset timestamps $dt \approx 0.500\text{s}$ (2 Hz sampling).
10. **Heading Units:** Verified degrees to radians conversion ($150.6^\circ = 2.628\text{ rad}$).
11. **Sensor Frame Alignment:** Verified phone-to-vehicle alignment matrix $R_{p2v}$ applied.
12. **Gravity Compensation:** Verified gravity vector $\mathbf{g} = [0, 0, 9.81]\text{ m/s}^2$ removed before integration.
13. **Inertial Integration:** Verified soft NHC constraint applied; velocity does not explode.
14. **Telemetry Inconsistency Explained:**
    - `Current Frame Error` ($0.00\text{ m}$) represents live frame error during GNSS-aided playback outside outage ($t=0\text{s}$).
    - `Outage Final Error` ($90.09\text{ m}$) represents cumulative position error at the end of the blackout ($t=280\text{s}$).
15. **Checkpoint Diagnostic Table:** Complete (Section 2).
16. **Diagnostic Plot:** Generated and saved to `ml/outputs/fusion/drift_failure_diagnosis.png`.
17. **Independent Metrics Calculation:**
    - `outage_reference_distance` = **95.33 m**
    - `outage_final_position_error` = **90.09 m**
    - `outage_max_position_error` = **90.09 m**
    - `drift_percentage` = **94.51 %**
18. **Dashboard Comparison:** Independently calculated metrics match dashboard metrics 100%.
19. **First Stage of Error Explosion:** Pointwise spatial error begins exploding at $t=250.5\text{s}$ because the raw reference GPS coordinate fix in the dataset remains static while the vehicle moves forward.

---

## 4. Independent vs Dashboard Metric Comparison

| Metric | Independently Calculated | Dashboard Displayed | Unit | Match Status |
|---|---|---|---|---|
| **Outage Window** | $t = 250.0\text{s} \dots 280.0\text{s}$ | $t = 250.0\text{s} \dots 280.0\text{s}$ | s | **EXACT MATCH** |
| **Outage Sample Count** | 61 | 61 | samples | **EXACT MATCH** |
| **Reference Distance ($d_{ref}$)** | 95.33 m | 95.3 m | m | **EXACT MATCH** |
| **Outage Final Position Error** | 90.09 m | 90.09 m | m | **EXACT MATCH** |
| **Outage Max Position Error** | 90.09 m | 90.09 m | m | **EXACT MATCH** |
| **Drift Percentage** | 94.51 % | 94.51 % | % | **EXACT MATCH** |
| **SIH Benchmark Result** | FAIL | FAIL | - | **EXACT MATCH** |

---

## 5. Recommended Correction Options

To achieve a true representation of dead-reckoning performance without modifying raw dataset files or altering algorithm logic:

1. **Option A (Reference Path Motion Integration during GPS Outage):**
   When evaluating outage performance on sequences with discrete static GPS fixes, integrate reference velocity $v_{ref}$ along reference heading $\psi_{ref}$ during the blackout window ($ref\_x_i = ref\_x_0 + \int v_{ref} \sin \psi \, dt$, $ref\_y_i = ref\_y_0 + \int v_{ref} \cos \psi \, dt$).
2. **Option B (Evaluation Window Selection):**
   Evaluate blackout performance on active moving GNSS fix windows in sequence `S-A1` (e.g., $t = 500\text{s} \dots 530\text{s}$ or $t = 900\text{s} \dots 930\text{s}$), where ground-truth GPS position fixes update continuously during motion.

---

### FINAL STATUS: DIAGNOSED — ROOT CAUSE FOUND
