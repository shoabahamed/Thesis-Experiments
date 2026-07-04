"""
Data loading: CSV/TXT parsing, recording discovery, and segment extraction.

Provides:
  - load_segments           : parse annotation TXT file
  - load_leap_csv           : parse Leap CSV → (T, 138) feature matrix
  - find_user_recordings    : discover matching CSV+TXT pairs per user
  - build_intervals_with_background : insert background gaps between signs
  - extract_segments_for_recording  : build per-interval records for one recording
  - load_all_segments       : full pipeline → segments_by_user dict
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import BACKGROUND_LABEL, FEATURE_KEYS
from features import extract_features_from_row

RECORDING_ID_PATTERN = re.compile(r"(P\d+_S\d+_R\d+)", re.IGNORECASE)


# ──────────────────────────────────────────────────────────────────────
# File loaders
# ──────────────────────────────────────────────────────────────────────

def load_segments(path: Path) -> list[dict]:
    """Load segmentation file into a list of {start, end, label} dicts."""
    path = Path(path)
    if not path.exists():
        return []

    df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["start", "end", "label"], engine="python",
    )
    segments = []

    for _, row in df.iterrows():
        try:
            start = int(row["start"])
            end = int(row["end"])
            label = str(row["label"]).strip()
            if label:
                segments.append({"start": start, "end": end, "label": label})
        except Exception:
            continue

    return segments


def load_leap_csv(
    path: Path,
    return_dataframe: bool = False,
):
    """Load Leap CSV and return feature matrix (num_frames, 138)."""
    path = Path(path)
    if not path.exists():
        empty = np.zeros((0, len(FEATURE_KEYS)), dtype=np.float32)
        return (empty, pd.DataFrame()) if return_dataframe else empty

    df = pd.read_csv(path)
    if df.empty:
        empty = np.zeros((0, len(FEATURE_KEYS)), dtype=np.float32)
        return (empty, df) if return_dataframe else empty

    feature_rows = []
    for _, row in tqdm(
        df.iterrows(), total=len(df),
        desc=f"Extracting {path.name}", leave=False,
    ):
        feature_rows.append(extract_features_from_row(row))

    features = np.vstack(feature_rows).astype(np.float32)
    return (features, df) if return_dataframe else features


# ──────────────────────────────────────────────────────────────────────
# Recording discovery
# ──────────────────────────────────────────────────────────────────────

def extract_recording_id(filename: str) -> str | None:
    """Extract canonical recording ID from a CSV or TXT filename."""
    match = RECORDING_ID_PATTERN.search(filename)
    return match.group(1) if match else None


def find_user_recordings(dataset_root: Path) -> dict[str, list[dict]]:
    """Find matching (CSV, segmentation) pairs for each user directory."""
    dataset_root = Path(dataset_root)
    user_map: dict[str, list[dict]] = defaultdict(list)

    for user_dir in sorted(dataset_root.glob("user*")):
        if not user_dir.is_dir():
            continue

        leap_dir = user_dir / "leap_data"
        seg_dir = user_dir / "segmentation"
        if not leap_dir.exists() or not seg_dir.exists():
            continue

        seg_map = {}
        for seg_path in seg_dir.glob("*.txt"):
            recording_id = extract_recording_id(seg_path.name)
            if recording_id is not None:
                seg_map[recording_id] = seg_path

        for csv_path in leap_dir.glob("*.csv"):
            recording_id = extract_recording_id(csv_path.name)
            if recording_id is None:
                continue

            seg_path = seg_map.get(recording_id)
            if seg_path is not None:
                user_map[user_dir.name].append({
                    "recording_id": recording_id,
                    "csv_path": csv_path,
                    "seg_path": seg_path,
                })

    return user_map


# ──────────────────────────────────────────────────────────────────────
# Segment extraction
# ──────────────────────────────────────────────────────────────────────

def build_intervals_with_background(
    segment_defs: list[dict],
    num_frames: int,
    background_label: str = BACKGROUND_LABEL,
) -> list[dict]:
    """Build labeled intervals including background gaps between annotated signs."""
    if num_frames <= 0:
        return []

    cleaned = []
    for seg in segment_defs:
        try:
            start = max(0, int(seg["start"]))
            end = min(num_frames - 1, int(seg["end"]))
            label = str(seg["label"]).strip()
            if end >= start and label:
                cleaned.append({
                    "start": start, "end": end,
                    "label": label, "is_background": False,
                })
        except Exception:
            continue

    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    intervals = []
    prev_end = -1

    for seg in cleaned:
        if seg["start"] > prev_end + 1:
            intervals.append({
                "start": prev_end + 1,
                "end": seg["start"] - 1,
                "label": background_label,
                "is_background": True,
            })

        intervals.append(seg)
        prev_end = max(prev_end, seg["end"])

    if prev_end < num_frames - 1:
        intervals.append({
            "start": prev_end + 1,
            "end": num_frames - 1,
            "label": background_label,
            "is_background": True,
        })

    return intervals


def extract_segments_for_recording(
    csv_path: Path,
    seg_path: Path,
) -> list[dict]:
    """Return per-interval records for one recording, including background gaps."""
    features, raw_df = load_leap_csv(csv_path, return_dataframe=True)
    segment_defs = load_segments(seg_path)

    if features.shape[0] == 0:
        return []

    interval_defs = build_intervals_with_background(
        segment_defs, num_frames=features.shape[0],
    )
    if not interval_defs:
        return []

    confidence_cols = [
        col for col in ["left_confidence", "right_confidence"]
        if col in raw_df.columns
    ]
    records = []

    for seg in interval_defs:
        start = max(0, int(seg["start"]))
        end = min(features.shape[0] - 1, int(seg["end"]))
        if end < start:
            continue

        segment_array = features[start : end + 1]
        if segment_array.size == 0:
            continue

        segment_confidence = None
        if confidence_cols:
            conf_slice = raw_df.loc[start:end, confidence_cols].to_numpy(
                dtype=np.float32,
            )
            if conf_slice.size > 0:
                segment_confidence = np.nanmean(conf_slice, axis=1)

        records.append({
            "segment": segment_array.astype(np.float32),
            "label": seg["label"],
            "confidence": segment_confidence,
            "segment_span": (start, end),
            "recording_features": features,
            "is_background": bool(seg.get("is_background", False)),
        })

    return records


# ──────────────────────────────────────────────────────────────────────
# High-level loader
# ──────────────────────────────────────────────────────────────────────

def load_all_segments(
    dataset_root: Path,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Discover all recordings and extract segments.

    Returns
    -------
    user_recordings : dict[user_name → list of recording metadata]
    segments_by_user : dict[user_name → list of segment dicts]
    """
    user_recordings = find_user_recordings(dataset_root)
    if not user_recordings:
        raise RuntimeError(
            "No matching Leap CSV and segmentation TXT pairs were found."
        )

    print("Matched recordings per user:")
    for user_name, recs in user_recordings.items():
        print(f"  {user_name}: {len(recs)} recordings")

    segments_by_user: dict[str, list[dict]] = defaultdict(list)
    for user_name, recordings in user_recordings.items():
        for rec in tqdm(recordings, desc=f"Extracting segments for {user_name}"):
            rec_segments = extract_segments_for_recording(
                rec["csv_path"], rec["seg_path"],
            )
            for item in rec_segments:
                item["recording_id"] = rec["recording_id"]
            segments_by_user[user_name].extend(rec_segments)

    total = sum(len(lst) for lst in segments_by_user.values())
    print(f"\nTotal raw segments (sign + background): {total}")

    return user_recordings, segments_by_user
