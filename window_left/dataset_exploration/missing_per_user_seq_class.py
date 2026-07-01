"""
Per-user, per-sequence-ID, per-class missing frame analysis.

For every annotated segment (sign and background gap) across all users,
computes:
  - % of frames where both hands are untracked  (missing)
  - longest consecutive missing-frame run
  - which recording and sequence the segment belongs to

Outputs:
  1. all_missing_segments_detail.csv   — one row per segment
  2. missing_per_user_seq_class.csv    — aggregated per (User, SeqID, Label)
  3. missing_per_user_class.csv        — aggregated per (User, Label)
"""
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path("c:/Shoab/Thesis/Experiments/dataset")
USERS = ["user1", "user2", "user3", "user5"]
USER_DISPLAY = {
    "user1": "User1",
    "user2": "User2",
    "user3": "User3",
    "user5": "User4",
}
OUTPUT_DIR = Path(__file__).resolve().parent


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def extract_rec_id(fname: str) -> str | None:
    m = re.search(r"(P\d+_S\d+_R\d+)", fname, re.IGNORECASE)
    return m.group(1) if m else None


def extract_seq_id(recording_id: str) -> str:
    """Extract the sequence identifier (e.g. 'S1') from 'P1_S1_R3'."""
    m = re.search(r"_(S\d+)_", recording_id)
    return m.group(1) if m else "unknown"


def load_segments(path: Path) -> list[dict]:
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["start", "end", "label"], engine="python")
    segs = []
    for _, row in df.iterrows():
        try:
            segs.append({
                "start": int(row["start"]),
                "end": int(row["end"]),
                "label": str(row["label"]).strip(),
            })
        except Exception:
            continue
    return segs


def longest_consecutive(mask: np.ndarray) -> int:
    max_c = cur = 0
    for v in mask:
        if v:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 0
    return max_c


def build_intervals(segment_defs: list[dict], n_frames: int,
                    bg_label: str = "background") -> list[dict]:
    """Build full timeline including background gaps between annotated signs."""
    intervals = []
    prev_end = 0
    for s in segment_defs:
        if s["start"] > prev_end:
            intervals.append({"start": prev_end, "end": s["start"] - 1,
                              "label": bg_label})
        intervals.append(s)
        prev_end = s["end"] + 1
    if prev_end < n_frames:
        intervals.append({"start": prev_end, "end": n_frames - 1,
                          "label": bg_label})
    return intervals


# ──────────────────────────────────────────────────────────────────────
# Main collection loop
# ──────────────────────────────────────────────────────────────────────
records: list[dict] = []

for user in USERS:
    display = USER_DISPLAY[user]
    user_leap_dir = DATASET_ROOT / user / "leap_data"
    user_seg_dir = DATASET_ROOT / user / "segmentation"

    if not user_leap_dir.exists() or not user_seg_dir.exists():
        print(f"[WARN] Skipping {user}: missing leap_data or segmentation dir")
        continue

    # Build path maps keyed by recording ID
    csv_map: dict[str, Path] = {}
    for f in sorted(user_leap_dir.glob("*.csv")):
        rid = extract_rec_id(f.stem)
        if rid:
            csv_map[rid] = f

    seg_map: dict[str, Path] = {}
    for f in sorted(user_seg_dir.glob("*.txt")):
        rid = extract_rec_id(f.stem)
        if rid:
            seg_map[rid] = f

    for rid in sorted(csv_map.keys()):
        if rid not in seg_map:
            continue

        df = pd.read_csv(csv_map[rid])
        segs = load_segments(seg_map[rid])
        n_frames = len(df)
        seq_id = extract_seq_id(rid)

        # Both-hands-missing mask
        left_ok = (df["left_confidence"] > 0) & (df["left_palm_x"] != 0)
        right_ok = (df["right_confidence"] > 0) & (df["right_palm_x"] != 0)
        missing = ~(left_ok | right_ok)

        intervals = build_intervals(segs, n_frames)

        for iv in intervals:
            s, e = iv["start"], min(iv["end"], n_frames - 1)
            seg_missing = missing.iloc[s : e + 1].to_numpy()
            seg_len = e - s + 1
            n_miss = int(seg_missing.sum())
            pct = (n_miss / seg_len * 100) if seg_len > 0 else 0.0
            lc = longest_consecutive(seg_missing)

            records.append({
                "User": display,
                "UserDir": user,
                "RecordingID": rid,
                "SeqID": seq_id,
                "Label": iv["label"],
                "SegStart": s,
                "SegEnd": e,
                "SegLen": seg_len,
                "MissingFrames": n_miss,
                "PctMissing": round(pct, 2),
                "LongestConsecutiveMissing": lc,
            })

    print(f"[OK] {display} ({user}): {sum(1 for r in records if r['UserDir'] == user)} segments collected")

df_all = pd.DataFrame(records)

# ──────────────────────────────────────────────────────────────────────
# Output 1: Segment-level detail
# ──────────────────────────────────────────────────────────────────────
detail_path = OUTPUT_DIR / "all_missing_segments_detail.csv"
df_all.to_csv(detail_path, index=False)
print(f"\nSaved segment detail -> {detail_path}  ({len(df_all)} rows)")

# ──────────────────────────────────────────────────────────────────────
# Output 2: Per (User, SeqID, Label) aggregate
# ──────────────────────────────────────────────────────────────────────
agg_usl = df_all.groupby(["User", "SeqID", "Label"]).agg(
    TotalSegments=("SegLen", "count"),
    TotalFrames=("SegLen", "sum"),
    TotalMissing=("MissingFrames", "sum"),
    MeanPctMissing=("PctMissing", "mean"),
    MaxPctMissing=("PctMissing", "max"),
    MeanLongestConsec=("LongestConsecutiveMissing", "mean"),
    MaxLongestConsec=("LongestConsecutiveMissing", "max"),
).reset_index()
agg_usl["OverallPctMissing"] = (
    agg_usl["TotalMissing"] / agg_usl["TotalFrames"] * 100
).round(2)
agg_usl = agg_usl.sort_values(
    ["User", "SeqID", "OverallPctMissing"], ascending=[True, True, False]
)

usl_path = OUTPUT_DIR / "missing_per_user_seq_class.csv"
agg_usl.to_csv(usl_path, index=False)
print(f"Saved per (User, SeqID, Label) -> {usl_path}  ({len(agg_usl)} rows)")

# ──────────────────────────────────────────────────────────────────────
# Output 3: Per (User, Label) aggregate
# ──────────────────────────────────────────────────────────────────────
agg_ul = df_all.groupby(["User", "Label"]).agg(
    TotalSegments=("SegLen", "count"),
    TotalFrames=("SegLen", "sum"),
    TotalMissing=("MissingFrames", "sum"),
    MeanPctMissing=("PctMissing", "mean"),
    MaxPctMissing=("PctMissing", "max"),
    MeanLongestConsec=("LongestConsecutiveMissing", "mean"),
    MaxLongestConsec=("LongestConsecutiveMissing", "max"),
).reset_index()
agg_ul["OverallPctMissing"] = (
    agg_ul["TotalMissing"] / agg_ul["TotalFrames"] * 100
).round(2)
agg_ul = agg_ul.sort_values(
    ["User", "OverallPctMissing"], ascending=[True, False]
)

ul_path = OUTPUT_DIR / "missing_per_user_class.csv"
agg_ul.to_csv(ul_path, index=False)
print(f"Saved per (User, Label) -> {ul_path}  ({len(agg_ul)} rows)")

# ──────────────────────────────────────────────────────────────────────
# Console summary
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TOP MISSING-FRAME HOTSPOTS  (OverallPctMissing > 0, per User × Seq × Class)")
print("=" * 80)
hotspots = agg_usl[agg_usl["OverallPctMissing"] > 0].sort_values(
    "OverallPctMissing", ascending=False
)
if hotspots.empty:
    print("  No segments with missing frames found.")
else:
    print(hotspots.head(30).to_string(index=False))

print("\n" + "=" * 80)
print("PER-USER × PER-CLASS SUMMARY  (top 20 by OverallPctMissing)")
print("=" * 80)
top_ul = agg_ul[agg_ul["OverallPctMissing"] > 0].sort_values(
    "OverallPctMissing", ascending=False
)
if top_ul.empty:
    print("  No missing frames at the user × class level.")
else:
    print(top_ul.head(20).to_string(index=False))

print("\nDone.")
