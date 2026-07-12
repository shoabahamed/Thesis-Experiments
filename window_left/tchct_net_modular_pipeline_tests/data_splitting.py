"""
Data splitting: user-based train/val/test splits, segment filtering, and label encoding.

Provides:
  - build_recording_catalog : aggregate interval records into recording-level samples
  - split_dev_recordings    : split dev recordings into train/val (recording-level)
  - filter_segments         : quality filtering on segments
  - build_label_encoding    : create label_to_id / id_to_label from training segments
  - prepare_split           : full pipeline for a given (dev_users, test_user) combination
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.preprocessing import LabelEncoder

from config import (
    BACKGROUND_LABEL,
    DEV_VAL_RATIO,
    DEV_VAL_SEED,
    SEED,
)


# ──────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────

def dedupe_consecutive_labels(labels: list[str]) -> list[str]:
    """Collapse consecutive duplicate labels while preserving order."""
    if not labels:
        return []
    out = [labels[0]]
    for label in labels[1:]:
        if label != out[-1]:
            out.append(label)
    return out


def near_zero_frame_ratio(segment: np.ndarray, eps: float = 1e-6) -> float:
    """Compute fraction of frames whose L2 norm is almost zero."""
    if segment.size == 0:
        return 1.0
    frame_norm = np.linalg.norm(segment, axis=1)
    return float(np.mean(frame_norm < eps))


# ──────────────────────────────────────────────────────────────────────
# Recording catalog
# ──────────────────────────────────────────────────────────────────────

def build_recording_catalog(
    segments_by_user: dict[str, list[dict]],
    background_label: str = BACKGROUND_LABEL,
) -> list[dict]:
    """Aggregate interval-level records into recording-level continuous samples."""
    rec_map: dict[tuple[str, str], dict] = {}

    for user_name, items in segments_by_user.items():
        for item in items:
            recording_id = str(item.get("recording_id", "unknown"))
            key = (user_name, recording_id)
            if key not in rec_map:
                rec_map[key] = {
                    "user": user_name,
                    "recording_id": recording_id,
                    "V": item["recording_features"].astype(np.float32),
                    "intervals": [],
                }

            start, end = item["segment_span"]
            rec_map[key]["intervals"].append({
                "start": int(start),
                "end": int(end),
                "label": str(item["label"]),
                "is_background": bool(item.get("is_background", False)),
            })

    catalog = []
    for rec in rec_map.values():
        intervals = sorted(rec["intervals"], key=lambda x: (x["start"], x["end"]))
        segmentation_regions = [
            {
                "label": str(seg["label"]),
                "start_frame": int(seg["start"]),
                "end_frame": int(seg["end"]),
            }
            for seg in intervals
            if seg["label"] != background_label
        ]
        gt_labels = [seg["label"] for seg in segmentation_regions]
        gt_labels = dedupe_consecutive_labels(gt_labels)
        missing_ratio = near_zero_frame_ratio(rec["V"])
        catalog.append({
            "user": rec["user"],
            "recording_id": rec["recording_id"],
            "V": rec["V"],
            "ground_truth": gt_labels,
            "segmentation_regions": segmentation_regions,
            "missing_ratio": float(missing_ratio),
            "num_frames": int(rec["V"].shape[0]),
        })

    return catalog


# ──────────────────────────────────────────────────────────────────────
# Dev split
# ──────────────────────────────────────────────────────────────────────

def split_dev_recordings(
    catalog: list[dict],
    dev_users: list[str],
    val_ratio: float = DEV_VAL_RATIO,
    seed: int = DEV_VAL_SEED,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Split development recordings into train/val keys per user (recording-level)."""
    train_keys = set()
    val_keys = set()

    for user in dev_users:
        user_recs = [
            rec for rec in catalog
            if rec["user"] == user and len(rec.get("ground_truth", [])) > 0
        ]
        if len(user_recs) == 0:
            print(f"Warning: No development recordings with non-empty ground truth found for user {user}.")
            continue

        # Use a user-specific seed derived deterministically from the main seed to ensure reproducibility
        user_offset = sum(ord(c) for c in user)
        user_seed = seed + user_offset
        rng = random.Random(user_seed)
        
        # Group recordings by sequence identifier
        # Assuming recording_id format is like P1_S1_R1, we group by everything except the last part
        seq_groups = defaultdict(list)
        for rec in user_recs:
            rec_id = rec["recording_id"]
            # Extract sequence prefix, e.g., 'P1_S1' from 'P1_S1_R1'
            # If the format varies, this safely drops the last repetition chunk
            parts = rec_id.split("_")
            seq_prefix = "_".join(parts[:-1]) if len(parts) > 1 else rec_id
            seq_groups[seq_prefix].append(rec)
        
        train_recs = []
        val_recs = []

        for seq_prefix, group_recs in seq_groups.items():
            # Copy and shuffle to pick a random repetition for this sequence
            group_copy = list(group_recs)
            rng.shuffle(group_copy)
            
            # Exactly one repetition goes to validation, the rest go to training
            val_recs.append(group_copy[0])
            train_recs.extend(group_copy[1:])

        for rec in train_recs:
            train_keys.add((rec["user"], rec["recording_id"]))
        for rec in val_recs:
            val_keys.add((rec["user"], rec["recording_id"]))

    if len(train_keys) == 0 and len(val_keys) == 0:
        raise RuntimeError(
            "No development recordings with non-empty ground truth were found for any dev users."
        )

    return train_keys, val_keys


# ──────────────────────────────────────────────────────────────────────
# Segment filtering
# ──────────────────────────────────────────────────────────────────────

def print_filter_report(pool_name: str, total: int, removed_reasons: Counter) -> None:
    """Log how many segments were dropped from a pool and why, with percentages."""
    dropped = sum(removed_reasons.values())
    kept = total - dropped
    pct = (lambda n: f"{n}/{total} ({100.0 * n / total:.1f}%)") if total > 0 else (lambda n: f"{n}/0 (0.0%)")

    print(f"  [{pool_name}] total={total}  kept={pct(kept)}  dropped={pct(dropped)}")
    if dropped:
        for reason, count in removed_reasons.most_common():
            print(f"      - {reason:<15}: {pct(count)}")


def filter_segments(
    segments: list[dict],
    min_len: int = 10,
    max_zero_ratio: float = 0.40,
    min_confidence: float = 0.1,
) -> tuple[list[dict], Counter]:
    """Return filtered segments and a reason counter for removed segments."""
    kept = []
    removed_reasons = Counter()

    for item in segments:
        segment = item["segment"]
        conf = item.get("confidence", None)
        reasons = []

        if len(segment) < min_len:
            reasons.append("too_short")
        if near_zero_frame_ratio(segment) > max_zero_ratio:
            reasons.append("missing_data")
        if conf is not None and len(conf) > 0:
            avg_conf = float(np.nanmean(conf))
            if np.isnan(avg_conf) or avg_conf < min_confidence:
                reasons.append("low_confidence")

        if reasons:
            for reason in reasons:
                removed_reasons[reason] += 1
            continue

        kept.append(item)

    return kept, removed_reasons


# ──────────────────────────────────────────────────────────────────────
# Label encoding
# ──────────────────────────────────────────────────────────────────────

def build_label_encoding(
    train_segments: list[dict],
    background_label: str = BACKGROUND_LABEL,
) -> tuple[dict[str, int], dict[int, str], int, int]:
    """Create label_to_id / id_to_label mappings from training segments.

    Returns
    -------
    label_to_id  : str → int
    id_to_label  : int → str
    background_id: int
    num_classes  : int
    """
    sign_labels = sorted({
        item["label"]
        for item in train_segments
        if not item.get("is_background", False)
    })
    all_labels = [background_label] + sign_labels

    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    label_to_id = {
        label: int(idx)
        for idx, label in enumerate(label_encoder.classes_)
    }
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    background_id = int(label_to_id[background_label])
    num_classes = len(label_to_id)

    print("Label mapping (string -> ID):")
    for label, idx in sorted(label_to_id.items(), key=lambda x: x[1]):
        print(f"  {label!r} -> {idx}")
    print(f"\nTotal classes: {num_classes}")
    print(f"Background class ID: {background_id}")

    return label_to_id, id_to_label, background_id, num_classes


# ──────────────────────────────────────────────────────────────────────
# Full split pipeline
# ──────────────────────────────────────────────────────────────────────

def prepare_split(
    segments_by_user: dict[str, list[dict]],
    dev_users: list[str],
    test_user: str,
    val_ratio: float = DEV_VAL_RATIO,
    val_seed: int = DEV_VAL_SEED,
    exclude_train_seq: str = "",
) -> dict:
    """Build train/val/test segments + label encoding + WER catalogs.

    Returns a dict containing all the pieces needed for training:
    {
        "train_segments", "val_segments", "test_segments",
        "label_to_id", "id_to_label", "background_id", "num_classes",
        "recording_catalog",
        "test_wer_catalog", "dev_val_wer_catalog", "dev_train_wer_catalog",
    }
    """
    available_users = sorted(segments_by_user.keys())

    for u in dev_users:
        if u not in available_users:
            raise RuntimeError(
                f"Dev user '{u}' not found. Available: {available_users}"
            )
    if test_user not in available_users:
        raise RuntimeError(
            f"Test user '{test_user}' not found. Available: {available_users}"
        )

    # Build recording catalog
    recording_catalog = build_recording_catalog(segments_by_user)
    if len(recording_catalog) == 0:
        raise RuntimeError("No recording-level continuous samples were found.")

    # Recording-level dev split
    dev_train_keys, dev_val_keys = split_dev_recordings(
        catalog=recording_catalog,
        dev_users=dev_users,
        val_ratio=val_ratio,
        seed=val_seed,
    )

    # Pool segments into dev / test
    dev_pool: list[dict] = []
    test_pool: list[dict] = []

    for user_name, user_segs in segments_by_user.items():
        for item in user_segs:
            item_with_user = dict(item)
            item_with_user["user"] = user_name
            if user_name in dev_users:
                dev_pool.append(item_with_user)
            elif user_name == test_user:
                test_pool.append(item_with_user)

    # Filter
    dev_filtered, dev_removed = filter_segments(dev_pool)
    test_filtered, test_removed = filter_segments(test_pool)

    # Route dev segments by recording key
    train_segments: list[dict] = []
    val_segments: list[dict] = []

    for item in dev_filtered:
        key = (item.get("user", "unknown"), str(item.get("recording_id", "unknown")))
        if key in dev_train_keys:
            if exclude_train_seq and f"_{exclude_train_seq}_" in key[1]:
                continue
            train_segments.append(item)
        elif key in dev_val_keys:
            val_segments.append(item)

    test_segments = list(test_filtered)

    print(f"\n{'='*60}")
    print(f"SPLIT: test_user={test_user}, dev_users={dev_users}")
    print(f"{'='*60}")
    print("Segment filtering (reasons: too_short, missing_data, low_confidence):")
    print_filter_report("dev pool", len(dev_pool), dev_removed)
    print_filter_report("test pool", len(test_pool), test_removed)
    print(f"Train segments (filtered): {len(train_segments)}")
    print(f"Val segments (filtered)  : {len(val_segments)}")
    print(f"Test segments (filtered) : {len(test_segments)}")

    if len(train_segments) == 0:
        raise RuntimeError("Training segment pool is empty.")
    if len(val_segments) == 0:
        raise RuntimeError("Validation segment pool is empty.")
    if len(test_segments) == 0:
        raise RuntimeError("Test segment pool is empty.")

    # Label encoding from training data
    label_to_id, id_to_label, background_id, num_classes = build_label_encoding(
        train_segments,
    )

    # WER catalogs
    test_wer_catalog = [
        rec for rec in recording_catalog
        if rec["user"] == test_user and len(rec.get("ground_truth", [])) > 0
    ]
    dev_val_wer_catalog = [
        rec for rec in recording_catalog
        if (rec["user"], rec["recording_id"]) in dev_val_keys
        and len(rec.get("ground_truth", [])) > 0
    ]
    dev_train_wer_catalog = [
        rec for rec in recording_catalog
        if (rec["user"], rec["recording_id"]) in dev_train_keys
        and len(rec.get("ground_truth", [])) > 0
        and not (exclude_train_seq and f"_{exclude_train_seq}_" in rec["recording_id"])
    ]

    print(f"WER catalog sizes -> train: {len(dev_train_wer_catalog)}, "
          f"val: {len(dev_val_wer_catalog)}, test: {len(test_wer_catalog)}")

    return {
        "train_segments": train_segments,
        "val_segments": val_segments,
        "test_segments": test_segments,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "background_id": background_id,
        "num_classes": num_classes,
        "recording_catalog": recording_catalog,
        "test_wer_catalog": test_wer_catalog,
        "dev_val_wer_catalog": dev_val_wer_catalog,
        "dev_train_wer_catalog": dev_train_wer_catalog,
    }
