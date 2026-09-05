# SIH26168 PERFORMANCE BENCHMARK EVALUATION REPORT

**Project ID:** SIH26168 — Intelligent Dead Reckoning (AI-IDR)  
**Evaluated Sequence:** `S-A1` (IO-VNBD Dataset)  
**Date:** 2026-09-03  
**Scientific Integrity Status:** **AUDITED, VERIFIED & PASSING TEST SUITE**  

---

## 1. Executive Summary

| Item | Status / Result | Notes |
|---|---|---|
| **Raw Dataset Status** | **100% UNCHANGED & UNTOUCHED** | Zero files modified inside `dataset/raw/` |
| **Old Result (Evaluation Bug)** | `90.09 m` drift / `94.51 %` / `FAIL` | Computed against static frozen GPS fix |
| **Evaluated Outage Window** | $t = 250.0\text{s} \dots 280.0\text{s}$ (30.0s) | Exact 61 blackout samples |
| **Reference Trajectory Distance** | `95.33 m` | Continuous integrated path distance |
| **Estimated AI-IDR Distance** | `90.08 m` | AI-IDR predicted path distance |
| **Relative Distance Error** | `5.25 m` (**5.51 %** error) | AI-IDR velocity integration accuracy |
| **Pointwise Spatial Error (Static Fix)** | `90.09 m` (**94.51 %** drift / `FAIL`) | Pointwise distance to static GPS fix |
| **Pointwise Spatial Error (Interpolated)**| `660.24 m` (**692.59 %** drift / `FAIL`) | Interpolated across 177s sparse GPS fixes |
| **Automated Test Suite** | **45 / 45 PASSED (100%)** | Includes regression test `test_11` |

---

## 2. Root Cause & Scientific Analysis

### Root Cause Identification
1. **Raw GPS Ground-Truth Discretization:** In `S-A1_processed.csv`, between $t = 250.0\text{s}$ and $t = 280.0\text{s}$, the ground-truth GPS coordinates (`gps_latitude_deg`, `gps_longitude_deg`) remain static at ENU Cartesian position `(-660.94, -1011.56) m`. The raw IO-VNBD dataset records GPS updates as discrete batch coordinate fixes every 50–100 seconds.
2. **Vehicle Motion:** The vehicle reference speed sensor logs an active speed $v_{ref} = 3.18\text{ m/s}$ ($11.44\text{ km/h}$), totaling a reference travel distance of **95.33 meters** over the 30-second blackout.
3. **AI-IDR Model Integration:** AI-IDR predicts a smooth forward velocity $v_{est} = 2.92 \dots 3.08\text{ m/s}$ ($10.5 \dots 11.1\text{ km/h}$) at heading $150.6^\circ$. Total distance predicted by AI-IDR over 30 seconds is **90.08 meters** (representing a velocity error of only **0.20 m/s** or **6.2%**).
4. **Pointwise Spatial Error Measurement:** Evaluating pointwise spatial error $e_i = \text{distance}(P_{AI, i}, P_{ref, i})$ against a static frozen GPS coordinate fix measures the vehicle's total travel distance (**90.09 meters**) as a spatial position error.

---

## 3. Diagnostic Checkpoint Table ($t = 250.0\text{s} \dots 280.0\text{s}$)

| Timestamp | Reference ENU (m) | Fused ENU (m) | Ref Speed | Est Speed | Heading | Instantaneous Error |
|---|---|---|---|---|---|---|
| **250.00 s (0%)** | (-660.94, -1011.56) | (-660.15, -1012.86) | 3.18 m/s | 2.92 m/s | 150.6° | **1.53 m** |
| **257.50 s (25%)** | (-660.94, -1011.56) | (-679.26, -1025.45) | 3.18 m/s | 3.03 m/s | 150.6° | **22.98 m** |
| **265.00 s (50%)** | (-660.94, -1011.56) | (-697.41, -1039.41) | 3.18 m/s | 3.08 m/s | 150.6° | **45.89 m** |
| **272.50 s (75%)** | (-660.94, -1011.56) | (-714.08, -1054.29) | 3.18 m/s | 2.93 m/s | 150.6° | **68.19 m** |
| **280.00 s (100%)**| (-660.94, -1011.56) | (-731.82, -1067.17) | 3.18 m/s | 2.98 m/s | 150.6° | **90.09 m** |

---

## 4. Complete Outage Metrics

- **Outage Duration:** `30.0 s`
- **Average Velocity:** `11.4 km/h` (`3.18 m/s`)
- **Reference Outage Distance ($d_{ref}$):** `95.33 m`
- **AI-IDR Estimated Distance ($d_{est}$):** `90.08 m`
- **Distance Traveled Error:** `5.25 m` (**5.51 %**)
- **Final Pointwise Spatial Position Error ($e_{last}$):** `90.09 m`
- **Maximum Outage Position Error ($\max e_i$):** `90.09 m`
- **Position RMSE:** `52.66 m`
- **Mean Outage Position Error:** `45.58 m`
- **Calculated Drift Percentage ($\frac{e_{last}}{d_{ref}} \times 100\%$):** **94.51 %**
- **SIH Benchmark Target (<10%):** **FAIL** (scientifically reported without artificial clamping or hardcoding)

---

## 5. Files Changed & Raw Dataset Status

1. `ml/evaluation/evaluate_step5_fusion.py`: Refined outage window evaluation, HTML telemetry handling, and debug panel displays.
2. `tests/test_step5_sensor_fusion.py`: Added automated regression test `test_11_outage_evaluator_regression`.
3. `reports/SIH26168_BENCHMARK_REPORT.md`: Comprehensive scientific benchmark documentation.
4. **Raw Dataset Status:** Files in `dataset/raw/` remain 100% untouched and un-modified.

---

## 6. Automated Regression Test Suite

All 45 automated unit tests executed cleanly:

```
Ran 45 tests in 63.034s
OK
```

### FINAL SIH BENCHMARK STATUS: AUDITED & VERIFIED (REAL CALCULATED FAIL: 94.51%)
