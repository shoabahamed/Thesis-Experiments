"""
Per-class missing frame analysis for User1.
For each sign segment (and background gaps), computes:
  - % of frames where both hands are untracked
  - longest consecutive missing-frame run
  - which recording the segment belongs to
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = Path("c:/Shoab/Thesis/Experiments/dataset")
USER = "user1"

user_leap_dir = DATASET_ROOT / USER / "leap_data"
user_seg_dir  = DATASET_ROOT / USER / "segmentation"

csv_files = sorted(user_leap_dir.glob("*.csv"))
seg_files = sorted(user_seg_dir.glob("*.txt"))

# Build a map from recording ID to paths
def extract_rec_id(fname):
    import re
    m = re.search(r"(P\d+_S\d+_R\d+)", fname, re.IGNORECASE)
    return m.group(1) if m else None

csv_map = {}
for f in csv_files:
    rid = extract_rec_id(f.stem)
    if rid:
        csv_map[rid] = f

seg_map = {}
for f in seg_files:
    rid = extract_rec_id(f.stem)
    if rid:
        seg_map[rid] = f

def load_segments(path):
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["start", "end", "label"], engine="python")
    segs = []
    for _, row in df.iterrows():
        try:
            segs.append({"start": int(row["start"]), "end": int(row["end"]),
                         "label": str(row["label"]).strip()})
        except:
            continue
    return segs

def longest_consecutive(mask):
    max_c = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            max_c = max(max_c, cur)
        else:
            cur = 0
    return max_c

# Collect per-segment records
records = []

for rid in sorted(csv_map.keys()):
    if rid not in seg_map:
        continue
    df = pd.read_csv(csv_map[rid])
    segs = load_segments(seg_map[rid])
    n_frames = len(df)

    left_ok  = (df["left_confidence"] > 0) & (df["left_palm_x"] != 0)
    right_ok = (df["right_confidence"] > 0) & (df["right_palm_x"] != 0)
    missing  = ~(left_ok | right_ok)  # True where BOTH hands lost

    # Build full interval list including background gaps
    intervals = []
    prev_end = 0
    for s in segs:
        if s["start"] > prev_end:
            intervals.append({"start": prev_end, "end": s["start"] - 1,
                              "label": "background"})
        intervals.append(s)
        prev_end = s["end"] + 1
    if prev_end < n_frames:
        intervals.append({"start": prev_end, "end": n_frames - 1,
                          "label": "background"})

    for iv in intervals:
        s, e = iv["start"], iv["end"]
        seg_missing = missing.iloc[s:e+1].to_numpy()
        seg_len = e - s + 1
        n_miss = int(seg_missing.sum())
        pct = (n_miss / seg_len * 100) if seg_len > 0 else 0.0
        lc = longest_consecutive(seg_missing)
        records.append({
            "RecordingID": rid,
            "Label": iv["label"],
            "SegStart": s,
            "SegEnd": e,
            "SegLen": seg_len,
            "MissingFrames": n_miss,
            "PctMissing": round(pct, 2),
            "LongestConsecutiveMissing": lc,
        })

df_rec = pd.DataFrame(records)

# ── Per-class aggregate ────────────────────────────────────────────
print("=" * 80)
print("PER-CLASS MISSING FRAME SUMMARY FOR USER1")
print("=" * 80)

agg = df_rec.groupby("Label").agg(
    TotalSegments=("SegLen", "count"),
    TotalFrames=("SegLen", "sum"),
    TotalMissing=("MissingFrames", "sum"),
    MeanPctMissing=("PctMissing", "mean"),
    MaxPctMissing=("PctMissing", "max"),
    MeanLongestConsec=("LongestConsecutiveMissing", "mean"),
    MaxLongestConsec=("LongestConsecutiveMissing", "max"),
).reset_index()
agg["OverallPctMissing"] = (agg["TotalMissing"] / agg["TotalFrames"] * 100).round(2)
agg = agg.sort_values("OverallPctMissing", ascending=False)

print(agg.to_string(index=False))

# ── Segments with any missing frames (detail table) ────────────────
print("\n" + "=" * 80)
print("INDIVIDUAL SEGMENTS WITH MISSING FRAMES (User1)")
print("=" * 80)

has_missing = df_rec[df_rec["MissingFrames"] > 0].sort_values(
    "PctMissing", ascending=False)

print(has_missing.to_string(index=False))

# ── Save to CSV ───────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent
agg.to_csv(out_dir / "user1_missing_per_class_summary.csv", index=False)
has_missing.to_csv(out_dir / "user1_missing_segments_detail.csv", index=False)
print(f"\nSaved summary  → {out_dir / 'user1_missing_per_class_summary.csv'}")
print(f"Saved detail   → {out_dir / 'user1_missing_segments_detail.csv'}")
