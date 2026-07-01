import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Custom dummy torch module to prevent scipy/seaborn introspection crashes
class DummyDevice:
    def __init__(self, *args, **kwargs):
        pass
    def __str__(self):
        return "cpu"
    def __repr__(self):
        return "device(type='cpu')"

class DummyTensor:
    pass

class DummyTorch:
    Tensor = DummyTensor
    device = DummyDevice
    
    @staticmethod
    def manual_seed(seed):
        pass
        
    class cuda:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def manual_seed_all(seed):
            pass
            
    class backends:
        class cudnn:
            deterministic = True
            benchmark = False

import sys
sys.modules['torch'] = DummyTorch
sys.path.append(str(Path(__file__).resolve().parents[1] / "tchct_net"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import FEATURE_KEYS, FEATURE_INDEX, PALM_TRIPLETS, HANDS
from features import palm_reference_normalize_sequence

# Output directories
OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ROOT = Path("c:/Shoab/Thesis/Experiments/dataset")
USERS = ["user1", "user2", "user3", "user5"]

# Define mapping for reporting (User1-User4)
USER_MAP = {
    "user1": "User1 (user1)",
    "user2": "User2 (user2)",
    "user3": "User3 (user3)",
    "user5": "User4 (user5)"
}

# ──────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────

def get_dist(df, p1_cols, p2_cols):
    """Compute Euclidean distance between two points in a DataFrame."""
    dx = df[p1_cols[0]] - df[p2_cols[0]]
    dy = df[p1_cols[1]] - df[p2_cols[1]]
    dz = df[p1_cols[2]] - df[p2_cols[2]]
    return np.sqrt(dx**2 + dy**2 + dz**2)

def longest_consecutive_missing(missing_mask):
    """Compute the longest consecutive sequence of True values."""
    max_consec = 0
    curr_consec = 0
    for m in missing_mask:
        if m:
            curr_consec += 1
            max_consec = max(max_consec, curr_consec)
        else:
            curr_consec = 0
    return max_consec

def df_to_markdown(df):
    """Convert pandas DataFrame to Markdown table format."""
    headers = list(df.columns)
    header_line = "| " + " | ".join(map(str, headers)) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for _, row in df.iterrows():
        row_vals = []
        for x in row:
            if isinstance(x, (float, np.float32, np.float64)):
                row_vals.append(f"{x:.4f}")
            else:
                row_vals.append(str(x))
        rows.append("| " + " | ".join(row_vals) + " |")
    return "\n".join([header_line, separator_line] + rows)

# Set seaborn design aesthetics
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.titlesize": 20,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (10, 6)
})

# Color palette for consistent reporting
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"] # User1: Red (problematic), others: blue, green, orange
USER_COLORS = {user: COLORS[i] for i, user in enumerate(USERS)}

print("Starting Sign Language leave-one-out data validation and outlier analysis...")

# ──────────────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
# ──────────────────────────────────────────────────────────────────────

# We will collect data for all analyses in a single pass over the files
hand_size_records = []
orientation_records = []
missing_frame_records = []
feature_dist_samples = {u: [] for u in USERS}

# We want to select a subset of representative joints for Feature Distribution Analysis
# Palm, Wrist, Thumb Tip, Index Tip, Middle Tip, Pinky Tip
# Since distal_ex is not in FEATURE_KEYS, we use distal_sx (DIP joint) as the tip feature
representative_joints = {
    "Palm": ("palm_x", "palm_y", "palm_z"),
    "Wrist": ("wrist_x", "wrist_y", "wrist_z"),
    "Thumb Tip": ("thumb_distal_sx", "thumb_distal_sy", "thumb_distal_sz"),
    "Index Tip": ("index_distal_sx", "index_distal_sy", "index_distal_sz"),
    "Middle Tip": ("middle_distal_sx", "middle_distal_sy", "middle_distal_sz"),
    "Pinky Tip": ("pinky_distal_sx", "pinky_distal_sy", "pinky_distal_sz")
}

for user in USERS:
    user_dir = DATASET_ROOT / user / "leap_data"
    csv_files = sorted(list(user_dir.glob("*.csv")))
    print(f"Loading {len(csv_files)} files for {user}...")
    
    for f_path in tqdm(csv_files, desc=f"Processing {user}"):
        seq_name = f_path.stem
        df = pd.read_csv(f_path)
        if len(df) == 0:
            continue
            
        # 1. Missing Frame Analysis
        # Missing frame = neither hand is visible
        left_visible = (df["left_confidence"] > 0) & (df["left_palm_x"] != 0)
        right_visible = (df["right_confidence"] > 0) & (df["right_palm_x"] != 0)
        missing_mask = ~(left_visible | right_visible)
        
        num_missing = missing_mask.sum()
        total_len = len(df)
        pct_missing = (num_missing / total_len) * 100.0
        longest_consec = longest_consecutive_missing(missing_mask)
        
        missing_frame_records.append({
            "User": USER_MAP[user],
            "Sequence": seq_name,
            "TotalFrames": total_len,
            "MissingFrames": num_missing,
            "PctMissing": pct_missing,
            "LongestConsecutiveMissing": longest_consec
        })
        
        # 2. Hand Size Analysis & Palm Orientation Analysis
        # We calculate hand dimensions and orientations for frames where the hand is tracked.
        # We will collect per-frame metrics for Left and Right hands separately.
        l_hand_sizes_wrist_mid = []
        l_hand_sizes_palm_width = []
        l_hand_sizes_mid_bone = []
        
        r_hand_sizes_wrist_mid = []
        r_hand_sizes_palm_width = []
        r_hand_sizes_mid_bone = []
        
        l_orientations = []
        r_orientations = []
        
        # Extract orientation vectors
        # Left Hand
        l_val_idx = df[left_visible].index
        if len(l_val_idx) > 0:
            # Hand dimensions (using raw columns)
            # Wrist to Middle MCP (middle_metacarpal_ex)
            l_wrist_mid = get_dist(df.loc[l_val_idx], 
                                   ["left_wrist_x", "left_wrist_y", "left_wrist_z"],
                                   ["left_middle_metacarpal_ex", "left_middle_metacarpal_ey", "left_middle_metacarpal_ez"])
            l_hand_sizes_wrist_mid.extend(l_wrist_mid)
            
            # Palm width (Index MCP to Pinky MCP)
            l_palm_w = get_dist(df.loc[l_val_idx],
                                ["left_index_metacarpal_ex", "left_index_metacarpal_ey", "left_index_metacarpal_ez"],
                                ["left_pinky_metacarpal_ex", "left_pinky_metacarpal_ey", "left_pinky_metacarpal_ez"])
            l_hand_sizes_palm_width.extend(l_palm_w)
            
            # Middle Metacarpal bone length
            l_mid_bone = get_dist(df.loc[l_val_idx],
                                 ["left_middle_metacarpal_sx", "left_middle_metacarpal_sy", "left_middle_metacarpal_sz"],
                                 ["left_middle_metacarpal_ex", "left_middle_metacarpal_ey", "left_middle_metacarpal_ez"])
            l_hand_sizes_mid_bone.extend(l_mid_bone)
            
            # Palm Orientation (Roll, Pitch, Yaw)
            # Left palm normal and direction
            ldx, ldy, ldz = df.loc[l_val_idx, "left_palm_dx"], df.loc[l_val_idx, "left_palm_dy"], df.loc[l_val_idx, "left_palm_dz"]
            lnx, lny, lnz = df.loc[l_val_idx, "left_palm_nx"], df.loc[l_val_idx, "left_palm_ny"], df.loc[l_val_idx, "left_palm_nz"]
            
            l_pitch = np.arctan2(-ldy, np.sqrt(ldx**2 + ldz**2))
            l_yaw = np.arctan2(ldx, ldz)
            l_roll = np.arctan2(lnx, -lny)
            
            l_orientations.append(np.column_stack([np.degrees(l_roll), np.degrees(l_pitch), np.degrees(l_yaw)]))
            
        # Right Hand
        r_val_idx = df[right_visible].index
        if len(r_val_idx) > 0:
            # Hand dimensions
            r_wrist_mid = get_dist(df.loc[r_val_idx], 
                                   ["right_wrist_x", "right_wrist_y", "right_wrist_z"],
                                   ["right_middle_metacarpal_ex", "right_middle_metacarpal_ey", "right_middle_metacarpal_ez"])
            r_hand_sizes_wrist_mid.extend(r_wrist_mid)
            
            r_palm_w = get_dist(df.loc[r_val_idx],
                                ["right_index_metacarpal_ex", "right_index_metacarpal_ey", "right_index_metacarpal_ez"],
                                ["right_pinky_metacarpal_ex", "right_pinky_metacarpal_ey", "right_pinky_metacarpal_ez"])
            r_hand_sizes_palm_width.extend(r_palm_w)
            
            r_mid_bone = get_dist(df.loc[r_val_idx],
                                 ["right_middle_metacarpal_sx", "right_middle_metacarpal_sy", "right_middle_metacarpal_sz"],
                                 ["right_middle_metacarpal_ex", "right_middle_metacarpal_ey", "right_middle_metacarpal_ez"])
            r_hand_sizes_mid_bone.extend(r_mid_bone)
            
            # Palm Orientation
            rdx, rdy, rdz = df.loc[r_val_idx, "right_palm_dx"], df.loc[r_val_idx, "right_palm_dy"], df.loc[r_val_idx, "right_palm_dz"]
            rnx, rny, rnz = df.loc[r_val_idx, "right_palm_nx"], df.loc[r_val_idx, "right_palm_ny"], df.loc[r_val_idx, "right_palm_nz"]
            
            r_pitch = np.arctan2(-rdy, np.sqrt(rdx**2 + rdz**2))
            r_yaw = np.arctan2(rdx, rdz)
            r_roll = np.arctan2(rnx, -rny)
            
            r_orientations.append(np.column_stack([np.degrees(r_roll), np.degrees(r_pitch), np.degrees(r_yaw)]))
            
        # Append sequence averages to records
        hand_size_records.append({
            "User": USER_MAP[user],
            "Sequence": seq_name,
            # Left Hand Sizes
            "Left_WristMid": np.mean(l_hand_sizes_wrist_mid) if l_hand_sizes_wrist_mid else np.nan,
            "Left_PalmWidth": np.mean(l_hand_sizes_palm_width) if l_hand_sizes_palm_width else np.nan,
            "Left_MiddleBone": np.mean(l_hand_sizes_mid_bone) if l_hand_sizes_mid_bone else np.nan,
            # Right Hand Sizes
            "Right_WristMid": np.mean(r_hand_sizes_wrist_mid) if r_hand_sizes_wrist_mid else np.nan,
            "Right_PalmWidth": np.mean(r_hand_sizes_palm_width) if r_hand_sizes_palm_width else np.nan,
            "Right_MiddleBone": np.mean(r_hand_sizes_mid_bone) if r_hand_sizes_mid_bone else np.nan,
        })
        
        # Append orientation sequence averages
        l_ori_mean = np.mean(np.vstack(l_orientations), axis=0) if l_orientations else [np.nan, np.nan, np.nan]
        r_ori_mean = np.mean(np.vstack(r_orientations), axis=0) if r_orientations else [np.nan, np.nan, np.nan]
        
        orientation_records.append({
            "User": USER_MAP[user],
            "Sequence": seq_name,
            "Left_Roll": l_ori_mean[0], "Left_Pitch": l_ori_mean[1], "Left_Yaw": l_ori_mean[2],
            "Right_Roll": r_ori_mean[0], "Right_Pitch": r_ori_mean[1], "Right_Yaw": r_ori_mean[2],
        })
        
        # 3. Feature Distribution Analysis (normalized coordinates)
        # Load the raw features as 132-D vectors
        raw_features = df.reindex(columns=FEATURE_KEYS).fillna(0.0).to_numpy(dtype=np.float32)
        
        # Normalize
        norm_features = palm_reference_normalize_sequence(raw_features)
        
        # Extract features for representative joints for valid frames
        # We'll collect a sample (e.g., up to 200 frames per sequence to avoid huge memory usage)
        # Let's take frames where the hand is visible
        for hand in HANDS:
            is_visible = left_visible if hand == "left" else right_visible
            val_indices = np.where(is_visible)[0]
            if len(val_indices) == 0:
                continue
            
            # Subsample to at most 150 frames per sequence to keep memory reasonable
            if len(val_indices) > 150:
                sampled_idx = np.random.choice(val_indices, size=150, replace=False)
            else:
                sampled_idx = val_indices
                
            for idx in sampled_idx:
                frame_data = norm_features[idx]
                feat_sample = {}
                for joint_name, (jx, jy, jz) in representative_joints.items():
                    # Map names to keys
                    jx_key = f"{hand}_{jx}"
                    jy_key = f"{hand}_{jy}"
                    jz_key = f"{hand}_{jz}"
                    
                    # Find indices in FEATURE_KEYS
                    ix = FEATURE_INDEX[jx_key]
                    iy = FEATURE_INDEX[jy_key]
                    iz = FEATURE_INDEX[jz_key]
                    
                    feat_sample[f"{hand}_{joint_name}_x"] = frame_data[ix]
                    feat_sample[f"{hand}_{joint_name}_y"] = frame_data[iy]
                    feat_sample[f"{hand}_{joint_name}_z"] = frame_data[iz]
                
                feat_sample["Hand"] = hand
                feature_dist_samples[user].append(feat_sample)

# Convert collected lists to DataFrames
df_hand_size = pd.DataFrame(hand_size_records)
df_orientation = pd.DataFrame(orientation_records)
df_missing = pd.DataFrame(missing_frame_records)

# Save raw collected metrics for sanity check
df_hand_size.to_csv(OUTPUT_DIR / "hand_sizes.csv", index=False)
df_orientation.to_csv(OUTPUT_DIR / "palm_orientations.csv", index=False)
df_missing.to_csv(OUTPUT_DIR / "missing_frames.csv", index=False)

print("\nData extraction complete. Analyzing and generating figures...")

# ──────────────────────────────────────────────────────────────────────
# SECTION 1: Hand Size Analysis
# ──────────────────────────────────────────────────────────────────────
print("\n--- SECTION 1: Hand Size Analysis ---")
hand_size_stats = []

# Analyze Left and Right Hand metrics separately
metrics = {
    "Left Hand Wrist->MCP": "Left_WristMid",
    "Left Hand Palm Width": "Left_PalmWidth",
    "Right Hand Wrist->MCP": "Right_WristMid",
    "Right Hand Palm Width": "Right_PalmWidth"
}

markdown_stats = ""
for metric_name, col_name in metrics.items():
    stats = df_hand_size.groupby("User")[col_name].agg(["mean", "std", "min", "max", "count"]).reset_index()
    stats["Metric"] = metric_name
    hand_size_stats.append(stats)
    
    markdown_stats += f"\n### {metric_name} Statistics (mm)\n"
    markdown_stats += df_to_markdown(stats) + "\n"

# Output stats to file
with open(OUTPUT_DIR / "hand_size_stats.md", "w") as f:
    f.write(markdown_stats)
print(markdown_stats)

# Generate Boxplots for Hand Size
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (metric_name, col_name) in enumerate(metrics.items()):
    ax = axes[i]
    sns.boxplot(
        data=df_hand_size, x="User", y=col_name, ax=ax,
        palette=[USER_COLORS[u] for u in USERS],
        showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":8}
    )
    ax.set_title(metric_name)
    ax.set_ylabel("Distance (mm)")
    ax.set_xlabel("User")

plt.suptitle("Comparison of Hand Size Dimensions across Users", y=0.98)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "hand_size_comparison.png", dpi=300, bbox_inches='tight')
plt.close()


# ──────────────────────────────────────────────────────────────────────
# SECTION 2: Palm Orientation Analysis
# ──────────────────────────────────────────────────────────────────────
print("\n--- SECTION 2: Palm Orientation Analysis ---")
orientation_stats = ""

# For roll, pitch, yaw, compute sequence-level aggregates
orientation_metrics = [
    ("Left Hand Roll", "Left_Roll"),
    ("Left Hand Pitch", "Left_Pitch"),
    ("Left Hand Yaw", "Left_Yaw"),
    ("Right Hand Roll", "Right_Roll"),
    ("Right Hand Pitch", "Right_Pitch"),
    ("Right Hand Yaw", "Right_Yaw")
]

for metric_name, col_name in orientation_metrics:
    stats = df_orientation.groupby("User")[col_name].agg(["mean", "std", "min", "max", "count"]).reset_index()
    orientation_stats += f"\n### {metric_name} Statistics (degrees)\n"
    orientation_stats += df_to_markdown(stats) + "\n"

with open(OUTPUT_DIR / "palm_orientation_stats.md", "w") as f:
    f.write(orientation_stats)
print(orientation_stats)

# Plot boxplots for Left and Right hand Roll, Pitch, Yaw
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for r_idx, angle in enumerate(["Roll", "Pitch", "Yaw"]):
    for c_idx, hand in enumerate(["Left", "Right"]):
        ax = axes[r_idx, c_idx]
        col_name = f"{hand}_{angle}"
        sns.boxplot(
            data=df_orientation, x="User", y=col_name, ax=ax,
            palette=[USER_COLORS[u] for u in USERS],
            showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":8}
        )
        ax.set_title(f"{hand} Hand {angle}")
        ax.set_ylabel("Angle (degrees)")
        ax.set_xlabel("User")

plt.suptitle("Comparison of Palm Orientation (Roll, Pitch, Yaw) across Users", y=0.99)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "palm_orientation_comparison.png", dpi=300, bbox_inches='tight')
plt.close()


# ──────────────────────────────────────────────────────────────────────
# SECTION 3: Feature Distribution Analysis
# ──────────────────────────────────────────────────────────────────────
print("\n--- SECTION 3: Feature Distribution Analysis ---")
# We compile the samples into DataFrames
feat_dfs = []
for user, samples in feature_dist_samples.items():
    if not samples:
        continue
    temp_df = pd.DataFrame(samples)
    temp_df["User"] = USER_MAP[user]
    feat_dfs.append(temp_df)

df_features = pd.concat(feat_dfs, ignore_index=True)

# Select a subset of features to print stats for: Index Tip and Wrist
# x, y, and z separately
feat_dist_stats = ""
target_cols = [
    "right_Index Tip_x", "right_Index Tip_y", "right_Index Tip_z",
    "right_Wrist_x", "right_Wrist_y", "right_Wrist_z"
]

for col in target_cols:
    if col in df_features.columns:
        stats = df_features.groupby("User")[col].agg(["mean", "std", "min", "max"]).reset_index()
        feat_dist_stats += f"\n### Normalized Feature: {col} Statistics\n"
        feat_dist_stats += df_to_markdown(stats) + "\n"

with open(OUTPUT_DIR / "feature_distribution_stats.md", "w") as f:
    f.write(feat_dist_stats)
print(feat_dist_stats)

# Plot feature distribution for Right Index Tip and Wrist (most signs are right-hand dominant)
# Since signs are right-hand dominant in many datasets, right hand features show clear patterns
fig, axes = plt.subplots(3, 2, figsize=(16, 16))

# Right Index Tip x, y, z
for i, axis in enumerate(["x", "y", "z"]):
    col = f"right_Index Tip_{axis}"
    ax = axes[i, 0]
    # KDE / Density plot
    for user in USERS:
        user_label = USER_MAP[user]
        user_data = df_features[df_features["User"] == user_label][col].dropna()
        sns.kdeplot(user_data, ax=ax, label=user_label, color=USER_COLORS[user], fill=True, alpha=0.15)
    ax.set_title(f"Right Index Tip Normalized {axis.upper()}")
    ax.set_xlabel("Normalized coordinate")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)

# Right Wrist x, y, z
for i, axis in enumerate(["x", "y", "z"]):
    col = f"right_Wrist_{axis}"
    ax = axes[i, 1]
    for user in USERS:
        user_label = USER_MAP[user]
        user_data = df_features[df_features["User"] == user_label][col].dropna()
        sns.kdeplot(user_data, ax=ax, label=user_label, color=USER_COLORS[user], fill=True, alpha=0.15)
    ax.set_title(f"Right Wrist Normalized {axis.upper()}")
    ax.set_xlabel("Normalized coordinate")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)

plt.suptitle("Normalized Feature Distributions (KDE Density Plots) for Right Hand", y=0.99)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_distribution_kde.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot boxplots of these features for comparison
fig, axes = plt.subplots(3, 2, figsize=(16, 16))
for i, axis in enumerate(["x", "y", "z"]):
    col_idx = f"right_Index Tip_{axis}"
    ax = axes[i, 0]
    sns.boxplot(data=df_features, x="User", y=col_idx, ax=ax, palette=[USER_COLORS[u] for u in USERS])
    ax.set_title(f"Right Index Tip Normalized {axis.upper()}")
    ax.set_ylabel("Coordinate value")
    
    col_wrist = f"right_Wrist_{axis}"
    ax = axes[i, 1]
    sns.boxplot(data=df_features, x="User", y=col_wrist, ax=ax, palette=[USER_COLORS[u] for u in USERS])
    ax.set_title(f"Right Wrist Normalized {axis.upper()}")
    ax.set_ylabel("Coordinate value")

plt.suptitle("Boxplots of Normalized Feature Distributions for Right Hand", y=0.99)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_distribution_boxplots.png", dpi=300, bbox_inches='tight')
plt.close()


# ──────────────────────────────────────────────────────────────────────
# SECTION 4: Missing Frame Analysis
# ──────────────────────────────────────────────────────────────────────
print("\n--- SECTION 4: Missing Frame Analysis ---")
missing_stats_summary = df_missing.groupby("User").agg({
    "TotalFrames": ["sum", "mean"],
    "MissingFrames": ["sum", "mean", "max"],
    "PctMissing": ["mean", "max", "std"],
    "LongestConsecutiveMissing": ["mean", "max"]
}).reset_index()

# Flatten headers
missing_stats_summary.columns = [
    "User", "TotalFrames_Sum", "TotalFrames_Mean", "MissingFrames_Sum",
    "MissingFrames_Mean", "MissingFrames_Max", "PctMissing_Mean", "PctMissing_Max",
    "PctMissing_Std", "LongestConsecutiveMissing_Mean", "LongestConsecutiveMissing_Max"
]

markdown_missing = "### Missing Frame Aggregates per User\n"
markdown_missing += df_to_markdown(missing_stats_summary) + "\n"

with open(OUTPUT_DIR / "missing_frame_stats.md", "w") as f:
    f.write(markdown_missing)
print(markdown_missing)

# Plot boxplots of missing frame percentage and longest consecutive missing frames
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(
    data=df_missing, x="User", y="PctMissing", ax=axes[0],
    palette=[USER_COLORS[u] for u in USERS],
    showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":8}
)
axes[0].set_title("Percentage of Missing Frames per Sequence")
axes[0].set_ylabel("Percentage (%)")
axes[0].set_xlabel("User")

sns.boxplot(
    data=df_missing, x="User", y="LongestConsecutiveMissing", ax=axes[1],
    palette=[USER_COLORS[u] for u in USERS],
    showmeans=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":8}
)
axes[1].set_title("Longest Consecutive Missing Frame Segment")
axes[1].set_ylabel("Number of Frames")
axes[1].set_xlabel("User")

plt.suptitle("Missing Frame Analysis across Users", y=0.98)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "missing_frames_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nAnalysis code execution complete. All tables and plots are successfully generated and saved!")
