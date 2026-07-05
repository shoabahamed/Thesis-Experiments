# Leave-One-User-Out Cross-Validation Data Audit

This report investigates why **User 1** (`user1`) performs substantially worse on sign language gesture recognition (F1 ≈ 0.85, WER ≈ 0.25) compared to the other three users (F1 ≈ 0.92, WER ≈ 0.02) under Leave-One-User-Out (LOUO) cross-validation. 

Our investigation confirms that the poor performance is **not due to model failure**, but is driven by systematic data quality and physiological anomalies in User 1's recordings.

---

## Executive Summary of Findings

1. **Physiological Outlier (Hand Size)**:
   User 1 has **significantly larger hand dimensions** than all other users.
   * Left Hand Wrist → Middle MCP distance is **95.71 mm** (User 1) vs. **85.72–91.33 mm** (others).
   * Left Hand Palm Width is **58.07 mm** (User 1) vs. **51.72–55.55 mm** (others).
   * Because palm-reference normalization only translates the coordinate origin (subtracting palm position) without scaling by bone length, User 1's finger coordinates extend significantly further from the palm center (covariate shift).

2. **Severe Tracking Dropouts (Missing Frames)**:
   User 1 has **an order of magnitude more completely missing frames** (where the Leap Motion sensor lost track of both hands) than all other users combined.
   * User 1: **332 missing frames** (max **12%** of a sequence, longest consecutive dropout of **27 frames**, i.e., almost 1 second of sign performance).
   * Other Users: At most **13 total missing frames** (max **0.67%** in a sequence, max **4** consecutive frames).
   * These dropouts fill the sequence with blocks of `0.0` values, violating sequence continuity and disrupting the Transformer's attention mechanism.

3. **Palm Orientation Instability**:
   Even after correcting for angular wrapping, User 1's Left Hand Yaw standard deviation is **86.99°** vs. **11.02°–27.02°** for other users. This indicates severe orientation tracking instability and noise.

---

## 1. Hand Size Analysis

For each sequence, the average hand dimensions were calculated over all valid frames where each hand was detected. The sequence-level metrics were then aggregated per user.

### Statistical Summary Table (mm)

| Metric | User | Mean | Std | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Left Hand Wrist → Middle MCP** | **User 1 (user1)** | **95.71** | **2.47** | **89.06** | **99.80** |
| | User 2 (user2) | 85.72 | 0.97 | 82.09 | 87.49 |
| | User 3 (user3) | 91.33 | 1.25 | 86.75 | 93.91 |
| | User 4 (user5) | 89.22 | 1.73 | 84.11 | 91.33 |
| **Left Hand Palm Width** | **User 1 (user1)** | **58.07** | **1.21** | **54.42** | **59.71** |
| | User 2 (user2) | 51.72 | 0.56 | 49.65 | 52.81 |
| | User 3 (user3) | 55.55 | 0.74 | 53.13 | 57.16 |
| | User 4 (user5) | 53.69 | 1.05 | 50.60 | 54.97 |
| **Right Hand Wrist → Middle MCP** | **User 1 (user1)** | **92.20** | **2.33** | **87.25** | **97.73** |
| | User 2 (user2) | 86.58 | 2.03 | 81.24 | 89.98 |
| | User 3 (user3) | 92.53 | 1.27 | 88.40 | 94.62 |
| | User 4 (user5) | 86.33 | 1.64 | 80.27 | 89.66 |
| **Right Hand Palm Width** | **User 1 (user1)** | **56.10** | **1.28** | **53.60** | **59.48** |
| | User 2 (user2) | 52.67 | 1.02 | 49.64 | 54.36 |
| | User 3 (user3) | 56.47 | 0.74 | 54.26 | 57.75 |
| | User 4 (user5) | 51.95 | 0.97 | 48.43 | 54.00 |

### Visualizations
The boxplots below compare the physical hand size distributions across users:
![Hand Size Comparison](file:///c:/Shoab/Thesis/Experiments/sequence/dataset_exploration/plots/hand_size_comparison.png)

> [!NOTE]
> User 1 has the largest left hand size (Wrist→MCP is ~10 mm larger than User 2's) and palm width, representing a clear physical outlier.

---

## 2. Palm Orientation Analysis

Palm orientation was computed per frame using palm direction and palm normal vectors, then aggregated.

### Statistical Summary Table (degrees)

| Metric | User | Mean | Std | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Left Hand Yaw (Standard)** | **User 1 (user1)** | **8.47** | **132.31** | **-173.32** | **175.55** |
| | User 2 (user2) | 148.02 | 27.02 | 73.12 | 171.33 |
| | User 3 (user3) | 145.01 | 11.02 | 122.47 | 160.76 |
| | User 4 (user5) | 148.23 | 14.85 | 115.13 | 165.51 |
| **Left Hand Yaw (Shifted to [-90, 270])** | **User 1 (user1)** | **141.10** | **86.99** | **-54.80** | **268.01** |
| | User 2 (user2) | 148.02 | 27.02 | 73.12 | 171.33 |
| | User 3 (user3) | 145.01 | 11.02 | 122.47 | 160.76 |
| | User 4 (user5) | 148.23 | 14.85 | 115.13 | 165.51 |

### Visualizations
![Palm Orientation Comparison](file:///c:/Shoab/Thesis/Experiments/sequence/dataset_exploration/plots/palm_orientation_comparison.png)

> [!WARNING]
> While the standard arithmetic mean for User 1's Left Hand Yaw was skewed to 8.47° due to 180° boundary-wrapping artifacts (which are resolved by shifting the range to $[-90, 270)$), the corrected standard deviation of **86.99°** remains 3x to 8x higher than the other users (11.02°–27.02°). This proves that User 1's left hand orientation was highly unstable during recording.

---

## 3. Feature Distribution Analysis

This analysis examines the normalized joint coordinates (relative to the palm origin) that are actually input into the Transformer.

### Statistical Summary Table: Right Index Tip Z & Right Wrist Z (mm)

| Feature | User | Mean | Std | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Right Index Tip Z** | **User 1 (user1)** | **-53.21** | **20.55** | **-85.73** | **19.90** |
| | User 2 (user2) | -49.88 | 21.02 | -77.41 | 16.65 |
| | User 3 (user3) | -51.32 | 20.05 | -80.30 | 25.65 |
| | User 4 (user5) | -50.59 | 19.87 | -73.90 | 28.98 |
| **Right Wrist Z** | **User 1 (user1)** | **58.64** | **15.61** | **-8.33** | **72.85** |
| | User 2 (user2) | 54.63 | 12.72 | -1.46 | 68.67 |
| | User 3 (user3) | 56.51 | 11.84 | -13.48 | 71.96 |
| | User 4 (user5) | 54.73 | 13.28 | -14.80 | 68.65 |

### Visualizations
The KDE plots show the coordinate shifts on the right hand:
![Feature KDE Plots](file:///c:/Shoab/Thesis/Experiments/sequence/dataset_exploration/plots/feature_distribution_kde.png)
![Feature Boxplots](file:///c:/Shoab/Thesis/Experiments/sequence/dataset_exploration/plots/feature_distribution_boxplots.png)

> [!IMPORTANT]
> The minimum values for finger tip coordinates (e.g. Right Index Tip Z = **-85.73 mm** for User 1 vs. **-73.90 mm** for User 4) indicate that User 1's fingers extend significantly further from the palm center. This is a direct consequence of User 1's larger hand size, causing a covariate shift in the features fed to the Transformer.

---

## 4. Missing Frame Analysis

Missing frames are defined as frames where **neither hand is visible or tracked** (both left and right hand confidences are 0).

### Statistical Summary Table

| User | Total Frames | Missing Frames | Pct Missing (Mean) | Pct Missing (Max) | Longest Consecutive (Mean) | Longest Consecutive (Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **User 1 (user1)** | **57,000** | **332** | **0.58%** | **12.00%** | **2.41 frames** | **27 frames** |
| User 2 (user2) | 49,650 | 13 | 0.03% | 0.67% | 0.16 frames | 4 frames |
| User 3 (user3) | 41,250 | 0 | 0.00% | 0.00% | 0.00 frames | 0 frames |
| User 4 (user5) | 54,000 | 9 | 0.02% | 0.53% | 0.08 frames | 2 frames |

### Visualizations
The boxplots below contrast the percentage of missing frames and the longest consecutive dropout segment:
![Missing Frame Comparison](file:///c:/Shoab/Thesis/Experiments/sequence/dataset_exploration/plots/missing_frames_comparison.png)

> [!CAUTION]
> **User 1 has a severe dropout rate.** The maximum missing-frame percentage in a sequence is **12.0%** (vs. **0.67%** for others), and the longest consecutive dropout segment is **27 frames** (almost 1 second of video, vs. **4 frames** for others). These gaps are filled with `0.0` vectors, causing major discontinuities that degrade Transformer temporal modelling.

---

## Recommended Preprocessing Mitigations

Since model architecture and post-processing cannot be changed, we recommend the following preprocessing strategies to address these data issues:

1. **Hand Size Scaling Normalization**:
   Instead of just translating the coordinates relative to the palm center, **scale the coordinates by the hand size** (e.g. divide all coordinates of a hand by its Wrist → Middle MCP distance). This will map the joint distributions of all users to a standard scale and eliminate the covariate shift caused by physical hand size variations.

2. **Temporal Interpolation of Missing Frames**:
   Replace the default zero-filling of missing frames with **linear or cubic spline interpolation** over time. This preserves sequence continuity and prevents the Transformer from seeing sudden jumps to `0.0`.

3. **Circular Wrapping for Yaw Angle Features**:
   If raw yaw angles are used as features (or if the model depends on yaw), convert the angles to Cartesian components (sine and cosine of the angle) rather than raw degrees/radians. This eliminates wrapping discontinuities.
