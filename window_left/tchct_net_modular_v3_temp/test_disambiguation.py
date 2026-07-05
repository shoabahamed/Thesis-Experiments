"""Unit tests for disambiguation.py — run in isolation, no model/decoder deps."""
from __future__ import annotations

import numpy as np

from disambiguation import (
    RAW_AUX_KEYS,
    compute_motion_energy,
    compute_turning_angle_histogram,
    disambiguate_background,
    disambiguate_region_label,
    build_class_templates,
)


def _row(**kwargs) -> np.ndarray:
    row = np.zeros(len(RAW_AUX_KEYS), dtype=np.float32)
    for key, val in kwargs.items():
        row[RAW_AUX_KEYS.index(key)] = val
    return row


def _make_left_velocity_trace(vectors: list[tuple[float, float, float]]) -> np.ndarray:
    """(T,12) raw_aux array with only left-hand velocity set, full confidence."""
    T = len(vectors)
    frames = np.zeros((T, len(RAW_AUX_KEYS)), dtype=np.float32)
    for t, (vx, vy, vz) in enumerate(vectors):
        frames[t, RAW_AUX_KEYS.index("left_palm_vx")] = vx
        frames[t, RAW_AUX_KEYS.index("left_palm_vy")] = vy
        frames[t, RAW_AUX_KEYS.index("left_palm_vz")] = vz
        frames[t, RAW_AUX_KEYS.index("left_confidence")] = 1.0
        frames[t, RAW_AUX_KEYS.index("right_confidence")] = 1.0
    return frames


def test_motion_energy_zero_on_low_confidence():
    row = _row(left_palm_vx=10.0, left_confidence=0.05, right_confidence=0.05)
    e = compute_motion_energy(row, None, conf_thresh=0.3, grab_w=0.5, pinch_w=0.5)
    assert e == 0.0


def test_motion_energy_uses_velocity_and_deltas():
    prev = _row(left_confidence=1.0, right_confidence=1.0, left_grab_strength=0.2)
    now = _row(left_palm_vx=3.0, left_palm_vy=4.0, left_confidence=1.0,
               right_confidence=1.0, left_grab_strength=0.7)
    e = compute_motion_energy(now, prev, conf_thresh=0.3, grab_w=0.5, pinch_w=0.5)
    # v_mag = 5.0, d_grab = 0.5 -> 5.0 + 0.5*0.5 = 5.25
    assert abs(e - 5.25) < 1e-5


def test_motion_energy_first_frame_no_crash():
    row = _row(left_palm_vx=1.0, left_confidence=1.0, right_confidence=1.0)
    e = compute_motion_energy(row, None, conf_thresh=0.3, grab_w=0.5, pinch_w=0.5)
    assert e == 1.0


def test_disambiguate_background_rescue_logic():
    assert disambiguate_background(is_near_miss=True, energy_now=5.0, theta_high=1.0) is True
    assert disambiguate_background(is_near_miss=False, energy_now=5.0, theta_high=1.0) is False
    assert disambiguate_background(is_near_miss=True, energy_now=0.5, theta_high=1.0) is False
    assert disambiguate_background(is_near_miss=True, energy_now=5.0, theta_high=None) is False


def test_turning_angle_histogram_rotation_invariant():
    # A simple L-shaped path: move +x, then +y (a 90-degree turn).
    vectors = [(1, 0, 0)] * 10 + [(0, 1, 0)] * 10
    frames = _make_left_velocity_trace(vectors)
    hist_a = compute_turning_angle_histogram(frames, n_bins=10, resample_steps=20, motion_eps=0.1)

    # Rotate every velocity vector by the same rotation matrix (90 deg about z).
    theta = np.pi / 3
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    rotated_vectors = [tuple(R @ np.array(v)) for v in vectors]
    frames_rot = _make_left_velocity_trace(rotated_vectors)
    hist_b = compute_turning_angle_histogram(frames_rot, n_bins=10, resample_steps=20, motion_eps=0.1)

    assert np.allclose(hist_a, hist_b, atol=1e-4)


def test_turning_angle_histogram_scale_invariant_in_angle_bins():
    vectors = [(1, 0, 0)] * 10 + [(0, 1, 0)] * 10
    frames = _make_left_velocity_trace(vectors)
    hist_a = compute_turning_angle_histogram(frames, n_bins=10, resample_steps=20, motion_eps=0.1)

    scaled_vectors = [(3 * vx, 3 * vy, 3 * vz) for vx, vy, vz in vectors]
    frames_scaled = _make_left_velocity_trace(scaled_vectors)
    hist_b = compute_turning_angle_histogram(frames_scaled, n_bins=10, resample_steps=20, motion_eps=0.1)

    assert np.allclose(hist_a, hist_b, atol=1e-4)


def test_turning_angle_histogram_motionless_hand_returns_zero_half():
    vectors = [(0, 0, 0)] * 20
    frames = _make_left_velocity_trace(vectors)  # right hand also zero
    hist = compute_turning_angle_histogram(frames, n_bins=5, resample_steps=20, motion_eps=0.1)
    assert np.all(hist == 0.0)
    assert hist.shape == (10,)


def test_no_nan_or_domain_errors_on_degenerate_input():
    frames = np.random.randn(3, len(RAW_AUX_KEYS)).astype(np.float32) * 1e-9
    hist = compute_turning_angle_histogram(frames, n_bins=10, resample_steps=20, motion_eps=5.0)
    assert not np.any(np.isnan(hist))


def test_disambiguate_region_label_no_trigger_when_confident():
    class_order = ["WHAT", "WHICH", "background"]
    templates = np.eye(3, dtype=np.float32)[:, :4]
    templates = np.random.RandomState(0).rand(3, 4).astype(np.float32)
    region_probs = np.tile(np.array([0.05, 0.9, 0.05], dtype=np.float32), (5, 1))
    region_raw_aux = np.zeros((5, len(RAW_AUX_KEYS)), dtype=np.float32)

    label, triggered = disambiguate_region_label(
        region_probs, region_raw_aux, templates, class_order,
        background_idx=2, fallback_label="WHICH",
        tau_margin=0.15, lam=0.4, top_k=3, n_bins=2, resample_steps=20, motion_eps=0.1,
    )
    assert label == "WHICH"
    assert triggered is False


def test_disambiguate_region_label_triggers_and_reranks_on_ambiguous_margin():
    class_order = ["WHAT", "WHICH", "background"]
    # WHICH template matches the query histogram far better than WHAT's.
    templates = np.array([
        [0.0, 1.0],   # WHAT
        [1.0, 0.0],   # WHICH
        [0.0, 0.0],   # background (unused)
    ], dtype=np.float32)
    # mean_probs: WHAT slightly ahead of WHICH -> ambiguous (margin < tau)
    region_probs = np.tile(np.array([0.5, 0.45, 0.05], dtype=np.float32), (5, 1))

    # Build raw_aux whose turning-angle histogram is dominated by bin 0
    # (a straight-line motion => histogram ~ [1, 0] for the moving hand),
    # matching the WHICH template.
    vectors = [(1, 0, 0)] * 20
    region_raw_aux = _make_left_velocity_trace(vectors)

    label, triggered = disambiguate_region_label(
        region_probs, region_raw_aux, templates, class_order,
        background_idx=2, fallback_label="WHAT",
        tau_margin=0.15, lam=0.9, top_k=2, n_bins=1, resample_steps=20, motion_eps=0.1,
    )
    assert triggered is True
    assert label == "WHICH"


def test_build_class_templates_averages_per_class():
    class_order = ["A", "B"]
    seg1 = {"label": "A", "raw_aux": _make_left_velocity_trace([(1, 0, 0)] * 20)}
    seg2 = {"label": "A", "raw_aux": _make_left_velocity_trace([(1, 0, 0)] * 20)}
    seg3 = {"label": "B", "raw_aux": _make_left_velocity_trace([(0, 0, 0)] * 20)}

    templates = build_class_templates(
        [seg1, seg2, seg3], class_order, n_bins=2, resample_steps=20, motion_eps=0.1,
    )
    assert templates.shape == (2, 4)
    # class B has an all-motionless left hand -> zero histogram half
    assert np.all(templates[1] == 0.0)
