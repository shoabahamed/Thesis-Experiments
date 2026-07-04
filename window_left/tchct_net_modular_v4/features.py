"""
Feature extraction and normalization for Leap Motion skeleton data.

Provides:
  - extract_features_from_row : CSV row → 132-D raw feature vector
  - palm_reference_normalize_frame : per-frame palm-reference normalization
  - palm_reference_normalize_sequence : vectorised version for (T, D) arrays
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    FEATURE_KEYS,
    HANDS,
    HAND_POSITION_TRIPLETS,
    PALM_TRIPLETS,
    USER_SCALE,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _safe_float(value, default: float = 0.0) -> float:
    """Convert values to float safely; return default on invalid input."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


# ──────────────────────────────────────────────────────────────────────
# Raw feature extraction
# ──────────────────────────────────────────────────────────────────────

def extract_features_from_row(row: pd.Series) -> np.ndarray:
    """Convert one Leap CSV row to one 132-D raw feature frame (no normalization)."""
    values = np.asarray(
        [_safe_float(row.get(key, 0.0)) for key in FEATURE_KEYS],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


# ──────────────────────────────────────────────────────────────────────
# Palm-reference normalization
# ──────────────────────────────────────────────────────────────────────

def palm_reference_normalize_frame(frame: np.ndarray, user: str | None = None) -> np.ndarray:
    """Normalize one frame by subtracting each hand's palm from that hand's coordinates,
    and optionally divide by hand size (USER_SCALE) for the user.

    Input shape: (132,)
    Output shape: (132,)
    """
    out = np.asarray(frame, dtype=np.float32).copy()
    if out.shape[0] != len(FEATURE_KEYS):
        raise ValueError(
            f"Expected frame with {len(FEATURE_KEYS)} features, got {out.shape[0]}"
        )

    # Resolve scale factor based on user
    scale = 1.0
    if user is not None:
        scale = USER_SCALE.get(user, 1.0)

    for hand in HANDS:
        px, py, pz = PALM_TRIPLETS[hand]
        palm = out[[px, py, pz]].copy()
        if not np.all(np.isfinite(palm)):
            palm = np.zeros((3,), dtype=np.float32)

        for ix, iy, iz in HAND_POSITION_TRIPLETS[hand]:
            out[ix] = (out[ix] - palm[0]) / scale
            out[iy] = (out[iy] - palm[1]) / scale
            out[iz] = (out[iz] - palm[2]) / scale

        # Keep palm channels for stable shape and explicit origin anchoring.
        out[px] = 0.0
        out[py] = 0.0
        out[pz] = 0.0

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def palm_reference_normalize_sequence(sequence: np.ndarray, user: str | None = None) -> np.ndarray:
    """Apply palm-reference normalization frame-wise.

    Input shape: (T, D)
    Output shape: (T, D)
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise ValueError(f"Expected sequence with shape (T, D), got {seq.shape}")
    if seq.shape[0] == 0:
        return seq.astype(np.float32, copy=False)

    normalized = np.empty_like(seq, dtype=np.float32)
    for i in range(seq.shape[0]):
        normalized[i] = palm_reference_normalize_frame(seq[i], user=user)
    return normalized
