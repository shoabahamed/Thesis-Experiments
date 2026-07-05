"""
Post-logit disambiguation heuristics for the THCT-Net streaming decoder.

Two independent, additive corrections computed from raw Leap Motion sensor
columns that are otherwise discarded before reaching the model:

  Hook A (background-vs-sign, every frame) : compute_motion_energy
  Hook B (sign-vs-sign, once per emission) : compute_turning_angle_histogram
                                              + disambiguate_region_label

Zero imports from model.py / trainer.py — this module only consumes raw
sensor arrays and probability vectors, so it is unit-testable in isolation
and safe to import from data_loading.py without pulling in torch/model deps.
"""
from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Raw auxiliary sensor columns (parallel channel to config.FEATURE_KEYS,
# never normalized, never fed to the model)
# ──────────────────────────────────────────────────────────────────────
RAW_AUX_KEYS: list[str] = []
for _hand in ["left", "right"]:
    for _field in ["palm_vx", "palm_vy", "palm_vz", "confidence", "grab_strength", "pinch_strength"]:
        RAW_AUX_KEYS.append(f"{_hand}_{_field}")

# Column indices within a RAW_AUX_KEYS-ordered row, per hand.
_AUX_INDEX = {key: idx for idx, key in enumerate(RAW_AUX_KEYS)}


def _hand_slice(hand: str) -> dict[str, int]:
    return {
        "vx": _AUX_INDEX[f"{hand}_palm_vx"],
        "vy": _AUX_INDEX[f"{hand}_palm_vy"],
        "vz": _AUX_INDEX[f"{hand}_palm_vz"],
        "conf": _AUX_INDEX[f"{hand}_confidence"],
        "grab": _AUX_INDEX[f"{hand}_grab_strength"],
        "pinch": _AUX_INDEX[f"{hand}_pinch_strength"],
    }


_LEFT = _hand_slice("left")
_RIGHT = _hand_slice("right")


# ──────────────────────────────────────────────────────────────────────
# Hook A — motion energy (background-vs-sign)
# ──────────────────────────────────────────────────────────────────────

def compute_motion_energy(
    aux_row: np.ndarray,
    prev_aux_row: np.ndarray | None,
    conf_thresh: float,
    grab_w: float,
    pinch_w: float,
) -> float:
    """
    One frame's motion energy, per hand, combined via max().

    aux_row / prev_aux_row : (12,) arrays ordered per RAW_AUX_KEYS, for this
    frame and the previous frame (prev_aux_row is None on the very first
    frame of a recording).
    """
    energies = []
    for hand_idx in (_LEFT, _RIGHT):
        conf = float(aux_row[hand_idx["conf"]])
        if conf < conf_thresh:
            energies.append(0.0)
            continue

        v_mag = float(np.sqrt(
            aux_row[hand_idx["vx"]] ** 2
            + aux_row[hand_idx["vy"]] ** 2
            + aux_row[hand_idx["vz"]] ** 2
        ))

        if prev_aux_row is None:
            d_grab = 0.0
            d_pinch = 0.0
        else:
            d_grab = abs(float(aux_row[hand_idx["grab"]]) - float(prev_aux_row[hand_idx["grab"]]))
            d_pinch = abs(float(aux_row[hand_idx["pinch"]]) - float(prev_aux_row[hand_idx["pinch"]]))

        energies.append(v_mag + grab_w * d_grab + pinch_w * d_pinch)

    return max(energies)


def disambiguate_background(
    is_near_miss: bool,
    energy_now: float,
    theta_high: float | None,
) -> bool:
    """
    Returns True if a narrowly-failing background gate should be rescued
    to IN_SIGN because independent motion-energy evidence says real motion
    is happening right now. Only ever rescues a near-miss — the caller is
    responsible for ensuring this is invoked only when the gate already
    failed by a narrow margin.
    """
    if theta_high is None:
        return False
    return bool(is_near_miss and energy_now > theta_high)


# ──────────────────────────────────────────────────────────────────────
# Hook B — turning-angle histogram (sign-vs-sign)
# ──────────────────────────────────────────────────────────────────────

def _resample(trace: np.ndarray, steps: int) -> np.ndarray:
    """Linearly interpolate (T, 3) velocity trace to exactly `steps` points."""
    t_len = trace.shape[0]
    if t_len == 0:
        return np.zeros((steps, 3), dtype=np.float32)
    if t_len == 1:
        return np.repeat(trace, steps, axis=0)

    src_idx = np.linspace(0.0, t_len - 1, num=t_len)
    dst_idx = np.linspace(0.0, t_len - 1, num=steps)
    out = np.empty((steps, 3), dtype=np.float32)
    for axis in range(3):
        out[:, axis] = np.interp(dst_idx, src_idx, trace[:, axis])
    return out


def _hand_histogram(
    velocity_trace: np.ndarray,  # (T_region, 3)
    n_bins: int,
    resample_steps: int,
    motion_eps: float,
) -> np.ndarray:
    resampled = _resample(velocity_trace, resample_steps)  # (resample_steps, 3)

    speeds = np.linalg.norm(resampled, axis=1)
    valid = speeds >= motion_eps
    directions = np.zeros_like(resampled)
    nonzero = speeds > 0
    directions[nonzero] = resampled[nonzero] / speeds[nonzero, None]

    angles = []
    prev_dir = None
    for i in range(resample_steps):
        if not valid[i]:
            prev_dir = None
            continue
        if prev_dir is not None:
            dot = float(np.clip(np.dot(prev_dir, directions[i]), -1.0, 1.0))
            angles.append(np.arccos(dot))
        prev_dir = directions[i]

    if len(angles) < 1:
        return np.zeros(n_bins, dtype=np.float32)

    hist, _ = np.histogram(angles, bins=n_bins, range=(0.0, np.pi))
    total = hist.sum()
    if total == 0:
        return np.zeros(n_bins, dtype=np.float32)
    return (hist / total).astype(np.float32)


def compute_turning_angle_histogram(
    raw_frames: np.ndarray,  # (T_region, 12) ordered per RAW_AUX_KEYS
    n_bins: int,
    resample_steps: int,
    motion_eps: float,
) -> np.ndarray:
    """
    Returns a (2 * n_bins,) L1-normalized vector: left-hand histogram
    concatenated with right-hand histogram, fixed order regardless of
    which hand is dominant for this sign.
    """
    left_v = raw_frames[:, [_LEFT["vx"], _LEFT["vy"], _LEFT["vz"]]]
    right_v = raw_frames[:, [_RIGHT["vx"], _RIGHT["vy"], _RIGHT["vz"]]]

    left_hist = _hand_histogram(left_v, n_bins, resample_steps, motion_eps)
    right_hist = _hand_histogram(right_v, n_bins, resample_steps, motion_eps)

    return np.concatenate([left_hist, right_hist])


def build_class_templates(
    labeled_segments: list[dict],  # each has "raw_aux" (T,12) + "label"
    class_order: list[str],
    n_bins: int,
    resample_steps: int,
    motion_eps: float,
) -> np.ndarray:
    """
    Returns (num_sign_classes, 2*n_bins) template bank, one row per entry
    of class_order, in that exact order.

    labeled_segments: only SIGN segments (exclude background) from
    dev_users' TRAINING split — never touch the held-out test user here.
    """
    by_class: dict[str, list[np.ndarray]] = {c: [] for c in class_order}
    for seg in labeled_segments:
        label = seg["label"]
        if label not in by_class:
            continue
        hist = compute_turning_angle_histogram(
            seg["raw_aux"], n_bins=n_bins, resample_steps=resample_steps, motion_eps=motion_eps,
        )
        by_class[label].append(hist)

    templates = np.zeros((len(class_order), 2 * n_bins), dtype=np.float32)
    for i, label in enumerate(class_order):
        instances = by_class[label]
        if instances:
            templates[i] = np.mean(np.stack(instances, axis=0), axis=0)
    return templates


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def disambiguate_region_label(
    region_probs: np.ndarray,     # (T_region, num_classes)
    region_raw_aux: np.ndarray,   # (T_region, 12)
    templates: np.ndarray,        # (num_sign_classes, 2*n_bins)
    class_order: list[str],
    background_idx: int,
    fallback_label: str,
    tau_margin: float,
    lam: float,
    top_k: int,
    n_bins: int,
    resample_steps: int,
    motion_eps: float,
) -> tuple[str, bool]:
    """
    Called once, at emission time, for a single completed IN_SIGN region.

    Returns (final_label, triggered) where `triggered` is True iff the
    margin check fired and Hook B's blended decision was actually used
    (as opposed to the plain majority-vote / fallback_label passing
    through unchanged).
    """
    mean_probs = region_probs.mean(axis=0)

    # Only ever re-rank sign classes — background must not reach here
    # (Hook A / the existing gate already handles background before a
    # region is considered IN_SIGN).
    sign_indices = [i for i in range(mean_probs.shape[0]) if i != background_idx]
    if len(sign_indices) < 2:
        return fallback_label, False

    order = sorted(sign_indices, key=lambda i: mean_probs[i], reverse=True)
    top1, top2 = order[0], order[1]
    margin = float(mean_probs[top1] - mean_probs[top2])

    if margin >= tau_margin:
        return fallback_label, False

    candidates = order[:top_k]
    hist = compute_turning_angle_histogram(
        region_raw_aux, n_bins=n_bins, resample_steps=resample_steps, motion_eps=motion_eps,
    )

    best_label = fallback_label
    best_score = -np.inf
    for idx in candidates:
        label = class_order[idx]
        sim = _cosine_sim(hist, templates[idx])
        blended = lam * sim + (1.0 - lam) * float(mean_probs[idx])
        if blended > best_score:
            best_score = blended
            best_label = label

    return best_label, (best_label != fallback_label)
