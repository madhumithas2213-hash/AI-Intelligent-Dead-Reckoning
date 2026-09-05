# FINAL METRIC CONSISTENCY REPORT — AI-IDR SIH 2026

**Project ID:** SIH26168  
**Module:** Metric Consistency Trace & Evaluation Audit  
**Date:** 2026-09-03  
**Status:** **AUDITED & FULLY CONSISTENT (REAL CALCULATED FAIL)**  

---

## 1. Trace of Exact Metric Values & Array Indexes

All metrics for sequence `S-A1` (IO-VNBD dataset) were traced directly to their array index positions over the 30-second blackout window ($t = 250.0\text{s} \dots 280.0\text{s}$):

- **Outage Start Time:** `250.00 s` (Relative timestamp `250.00 s`, Array Index: `500`)
- **Outage End Time:** `280.00 s` (Relative timestamp `280.00 s`, Array Index: `560`)
- **Outage Sample Count:** `61 samples` ($0.5\text{s}$ sampling interval)
- **Reference Start Position (`reference_start`):** `(-660.94, -1011.56) m` ENU (Index `500`)
- **Reference End Position (`reference_end`):** `(-660.94, -1011.56) m` ENU (Index `560`, static GPS fix in raw dataset)
- **AI-IDR Fused Start Position (`fused_start`):** `(-660.15, -1012.86) m` ENU (Index `500`)
- **AI-IDR Fused End Position (`fused_end`):** `(-731.82, -1067.17) m` ENU (Index `560`)
- **Final Position Error (`final_position_error`):** **90.09 m** (Pointwise Euclidean error at index `560`, $t = 280.0\text{s}$)
- **Maximum Position Error (`maximum_position_error`):** **90.09 m** (Peak pointwise Euclidean error across indexes `500..560`)
- **Outage Reference Distance (`outage_distance`):** **95.33 m** (Integrated reference speed $\sum v_{ref, i} \cdot \Delta t_i$)
- **Position Drift (`position_drift`):** **90.09 m** ($\max_{i} e_i$ during outage window)
- **Drift Percentage (`drift_percentage`):** **94.51 %** ($\frac{90.09\text{ m}}{95.33\text{ m}} \times 100\%$)
- **SIH Target Status:** **FAIL** (calculated $94.51\% \ge 10.0\%$)

---

## 2. Root Cause of Previous `Position Error = 0.0 m` Inconsistency

Tracing the frontend telemetry in `generate_html_dashboard` (`ml/evaluation/evaluate_step5_fusion.py`) revealed two root causes for why `Position Error: 0.0 m` appeared alongside `Maximum Position Error: 90.09 m`:

1. **Static HTML Telemetry Placeholder:**
   The static HTML template previously contained `<div class="debug-val" id="dbg-pos-err">0.0 m</div>`. Before replay, the element displayed `"0.0 m"`, while `Max Pos Error` displayed `{outage_drift_m:.2f} m` ($90.09\text{ m}$).
2. **Playback Frame Evaluation at $t = 0\text{s}$:**
   On page load, `drawTrajectory()` executed for frame `currentIndex = 0` ($t = 0\text{s}$). Outside outage, GNSS is active, so instantaneous position error $e_0 = \sqrt{(x_{fused, 0} - x_{ref, 0})^2 + (y_{fused, 0} - y_{ref, 0})^2} = 0.00\text{ m}$. `drawTrajectory()` updated `dbg-pos-err` with the frame error ($0.00\text{ m}$), while `Max Pos Error` remained fixed at $90.09\text{ m}$.

### Corrective Action Implemented
The debug telemetry panel labels and JavaScript handlers were updated to eliminate ambiguity:
- **Current Frame Error:** `<div id="dbg-current-err">0.00 m</div>` (Live instantaneous error at current playback frame $i$).
- **Outage Final Error:** `<div id="dbg-final-err">90.09 m</div>` (Evaluated at the end of the outage window, index `560`).
- **Outage Max Error:** `<div id="dbg-max-err">90.09 m</div>` (Maximum position error during the blackout window).

Now, both `Outage Final Error` ($90.09\text{ m}$) and `Outage Max Error` ($90.09\text{ m}$) match `Position Drift` ($90.09\text{ m}$) consistently across all panels.

---

## 3. Position Error & Outage Metric Formulas

For each sample $i \in \{1 \dots 61\}$ in the blackout window ($t = 250.0\text{s} \dots 280.0\text{s}$):

$$e_i = \sqrt{(x_{est, i} - x_{ref, i})^2 + (y_{est, i} - y_{ref, i})^2}$$

1. **Pointwise Position Error:**
   - At $t = 250.0\text{s}$ (index 500): $e_0 = \sqrt{(-660.15 - (-660.94))^2 + (-1012.86 - (-1011.56))^2} = \mathbf{1.53\text{ m}}$
   - At $t = 280.0\text{s}$ (index 560): $e_{60} = \sqrt{(-731.82 - (-660.94))^2 + (-1067.17 - (-1011.56))^2} = \mathbf{90.09\text{ m}}$

2. **Final Position Error:**
   $$\text{final\_error} = e_{60} = \mathbf{90.09\text{ m}}$$

3. **Maximum Position Error:**
   $$\text{max\_error} = \max_{0 \le i \le 60} e_i = \mathbf{90.09\text{ m}}$$

4. **Mean Position Error:**
   $$\text{mean\_error} = \frac{1}{61} \sum_{i=0}^{60} e_i = \mathbf{45.58\text{ m}}$$

5. **Position RMSE:**
   $$\text{RMSE} = \sqrt{\frac{1}{61} \sum_{i=0}^{60} e_i^2} = \mathbf{52.66\text{ m}}$$

6. **Outage Reference Distance:**
   $$d_{ref} = \sum_{i=0}^{59} v_{ref, i} \cdot \Delta t_i = \mathbf{95.33\text{ m}}$$

7. **Drift Percentage:**
   $$\text{Drift } \% = \frac{\text{max\_error}}{d_{ref}} \times 100\% = \frac{90.09}{95.33} \times 100\% = \mathbf{94.51\%}$$

---

## 4. Coordinate System & Trajectory Alignment Verification

- **Coordinate System:** Local ENU Cartesian meters relative to initial GNSS fix $(lat_0 = 1.3094^\circ, lon_0 = 103.7788^\circ)$.
- **Timestamp Alignment:** Both reference $P_{ref}[i]$ and AI-IDR fused $P_{AI}[i]$ are sampled at identical timestamps $t_i = t_0 + i \cdot 0.5\text{s}$.
- **Units:** Positions in meters ($m$), speeds in meters per second ($m/s$), time in seconds ($s$).

---

## 5. Summary of Metric Audit Results

| Metric | Pointwise Formula / Source | Exact Value | Unit | Consistency |
|---|---|---|---|---|
| **Outage Start Window** | $t = 250.0\text{s}$ (Index 500) | 250.00 s | s | Synchronized |
| **Outage End Window** | $t = 280.0\text{s}$ (Index 560) | 280.00 s | s | Synchronized |
| **Sample Count** | Outage mask $t \in [250s, 280s]$ | 61 | - | Synchronized |
| **Reference Distance** | $\int v_{ref} dt$ | 95.33 m | m | Synchronized |
| **Final Position Error** | $e_{60} = \|P_{AI, 60} - P_{ref, 60}\|$ | 90.09 m | m | **Aligned** |
| **Maximum Position Error** | $\max_{i} e_i$ | 90.09 m | m | **Aligned** |
| **Outage Position RMSE** | $\sqrt{\text{mean } e_i^2}$ | 52.66 m | m | Synchronized |
| **Position Drift** | Defined as $\max_{i} e_i$ | 90.09 m | m | **Aligned** |
| **Drift Percentage** | $\frac{90.09}{95.33} \times 100\%$ | 94.51 % | % | **Aligned** |
| **SIH Target Status** | Real calculated value (<10%) | **FAIL** | - | **Calculated** |

---

## 6. Automated Verification Test Suite

All 44 system unit tests executed cleanly:

```
Ran 44 tests in 93.85s
OK
```

### FINAL EVALUATION RESULT: FAIL (Real calculated drift: 94.51% on sequence S-A1)
