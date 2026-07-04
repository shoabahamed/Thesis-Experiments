"""
Data augmentation pipeline for Leap Motion continuous sign language recognition skeleton data.

Provides:
  - SignLanguageAugmentationPipeline : main augmentation composer
  - random_rotate_3d                 : rotate coordinates around palm
  - random_scale_uniform             : uniformly scale coordinates around palm
  - add_coordinate_noise             : add Gaussian noise to coordinates
  - random_frame_dropout             : randomly drop frames and interpolate
  - verify_augmented_sample          : validation checks for shape, range, bone lengths, etc.
"""
from __future__ import annotations

import numpy as np
import torch

from config import (
    FEATURE_INDEX,
    HANDS,
    FINGERS,
    BONES,
    HAND_POSITION_TRIPLETS,
    PALM_TRIPLETS,
)


def get_rotation_matrix(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    """Compute 3D rotation matrix for given Euler angles (in radians)."""
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx]
    ], dtype=np.float32)

    cy, sy = np.cos(angle_y), np.sin(angle_y)
    Ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy]
    ], dtype=np.float32)

    cz, sz = np.cos(angle_z), np.sin(angle_z)
    Rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return Rz @ (Ry @ Rx)


def random_rotate_3d(sequence: np.ndarray, max_degrees: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate the active hand coordinates around the palm reference point (0, 0, 0) in 3D.

    Each hand is rotated independently around the X, Y, and Z axes.
    """
    max_rad = np.deg2rad(max_degrees)
    out = sequence.copy()
    T = out.shape[0]

    for hand in HANDS:
        flat_idx = []
        for ix, iy, iz in HAND_POSITION_TRIPLETS[hand]:
            flat_idx.extend([ix, iy, iz])

        # Skip if hand is completely inactive (all zeros in sequence)
        if np.all(out[:, flat_idx] == 0.0):
            continue

        angles = rng.uniform(-max_rad, max_rad, size=3)
        R = get_rotation_matrix(angles[0], angles[1], angles[2])

        triplets = HAND_POSITION_TRIPLETS[hand]
        coords = np.zeros((T, len(triplets), 3), dtype=np.float32)
        for j, (ix, iy, iz) in enumerate(triplets):
            coords[:, j, 0] = out[:, ix]
            coords[:, j, 1] = out[:, iy]
            coords[:, j, 2] = out[:, iz]

        # Rotate batch
        rotated_coords = np.matmul(coords, R.T)

        for j, (ix, iy, iz) in enumerate(triplets):
            out[:, ix] = rotated_coords[:, j, 0]
            out[:, iy] = rotated_coords[:, j, 1]
            out[:, iz] = rotated_coords[:, j, 2]

    return out


def random_scale_uniform(
    sequence: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Uniformly scale all joint coordinates of active hands around the palm reference point."""
    out = sequence.copy()

    for hand in HANDS:
        flat_idx = []
        for ix, iy, iz in HAND_POSITION_TRIPLETS[hand]:
            flat_idx.extend([ix, iy, iz])

        if np.all(out[:, flat_idx] == 0.0):
            continue

        out[:, flat_idx] *= scale_factor

    return out


def _detect_sequence_scale(sequence: np.ndarray) -> float:
    """Detect if sequence is normalized by hand size and return the scale factor.
    If normalized, returns the estimated scale (e.g. USER_SCALE around 90).
    If not normalized, returns 1.0.
    """
    bone_lengths = compute_bone_lengths(sequence)
    if not bone_lengths:
        return 1.0
    # Get mean bone length of active hands
    all_lens = []
    for hand_lens in bone_lengths.values():
        all_lens.append(np.mean(hand_lens))
    if not all_lens:
        return 1.0
    mean_len = float(np.mean(all_lens))
    # If mean bone length is small (e.g. < 2.0), it's normalized.
    # The raw bone lengths are typically around 15.0 - 50.0 mm (mean ~30.0 mm).
    # We can compute the ratio of the expected average raw bone length (~30.0) to mean_len.
    if mean_len < 2.0:
        return 30.0 / max(1e-5, mean_len)
    return 1.0


def add_coordinate_noise(
    sequence: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add independent zero-mean Gaussian noise to coordinates of active hands."""
    out = sequence.copy()
    T = out.shape[0]

    # Detect sequence scale and adjust noise standard deviation accordingly
    seq_scale = _detect_sequence_scale(sequence)
    adjusted_noise_std = noise_std / seq_scale

    for hand in HANDS:
        flat_idx = []
        for ix, iy, iz in HAND_POSITION_TRIPLETS[hand]:
            flat_idx.extend([ix, iy, iz])

        if np.all(out[:, flat_idx] == 0.0):
            continue

        noise = rng.normal(0.0, adjusted_noise_std, size=(T, len(flat_idx))).astype(np.float32)
        out[:, flat_idx] += noise

    return out


def random_frame_dropout(
    sequence: np.ndarray,
    drop_prob: float,
    max_consecutive: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly drop a percentage of frames and fill via linear interpolation (or duplication at boundaries)."""
    T, D = sequence.shape
    if T <= 2:
        return sequence.copy()

    # Determine frames to drop while avoiding consecutive long segments
    dropped = np.zeros(T, dtype=bool)
    consecutive_count = 0
    for t in range(T):
        if consecutive_count >= max_consecutive:
            consecutive_count = 0
            continue
        if rng.random() < drop_prob:
            dropped[t] = True
            consecutive_count += 1
        else:
            consecutive_count = 0

    # Keep at least a few valid frames to prevent complete loss
    if np.all(dropped):
        dropped[rng.choice(T, size=min(T, 5), replace=False)] = False

    if not np.any(dropped):
        return sequence.copy()

    out = sequence.copy()
    kept_indices = np.where(~dropped)[0]

    for t in range(T):
        if not dropped[t]:
            continue

        preceding = kept_indices[kept_indices < t]
        succeeding = kept_indices[kept_indices > t]

        if len(preceding) > 0 and len(succeeding) > 0:
            t_prev = preceding[-1]
            t_next = succeeding[0]
            weight = (t - t_prev) / (t_next - t_prev)
            out[t] = out[t_prev] + weight * (out[t_next] - out[t_prev])
        elif len(preceding) > 0:
            # End of sequence boundary
            t_prev = preceding[-1]
            out[t] = out[t_prev]
        elif len(succeeding) > 0:
            # Start of sequence boundary
            t_next = succeeding[0]
            out[t] = out[t_next]

    return out


def compute_bone_lengths(sequence: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate lengths of finger bones across all frames of the sequence."""
    lengths = {}
    T = sequence.shape[0]

    for hand in HANDS:
        flat_idx = []
        for ix, iy, iz in HAND_POSITION_TRIPLETS[hand]:
            flat_idx.extend([ix, iy, iz])

        if np.all(sequence[:, flat_idx] == 0.0):
            continue

        hand_lengths = []
        for finger in FINGERS:
            idx_meta = [FEATURE_INDEX[f"{hand}_{finger}_metacarpal_sx"],
                        FEATURE_INDEX[f"{hand}_{finger}_metacarpal_sy"],
                        FEATURE_INDEX[f"{hand}_{finger}_metacarpal_sz"]]
            idx_prox = [FEATURE_INDEX[f"{hand}_{finger}_proximal_sx"],
                        FEATURE_INDEX[f"{hand}_{finger}_proximal_sy"],
                        FEATURE_INDEX[f"{hand}_{finger}_proximal_sz"]]
            idx_inter = [FEATURE_INDEX[f"{hand}_{finger}_intermediate_sx"],
                         FEATURE_INDEX[f"{hand}_{finger}_intermediate_sy"],
                         FEATURE_INDEX[f"{hand}_{finger}_intermediate_sz"]]
            idx_dist = [FEATURE_INDEX[f"{hand}_{finger}_distal_sx"],
                        FEATURE_INDEX[f"{hand}_{finger}_distal_sy"],
                        FEATURE_INDEX[f"{hand}_{finger}_distal_sz"]]

            # Metacarpal -> Proximal
            b1 = sequence[:, idx_meta] - sequence[:, idx_prox]
            hand_lengths.append(np.linalg.norm(b1, axis=1))

            # Proximal -> Intermediate
            b2 = sequence[:, idx_prox] - sequence[:, idx_inter]
            hand_lengths.append(np.linalg.norm(b2, axis=1))

            # Intermediate -> Distal
            b3 = sequence[:, idx_inter] - sequence[:, idx_dist]
            hand_lengths.append(np.linalg.norm(b3, axis=1))

        lengths[hand] = np.stack(hand_lengths, axis=1)  # (T, 15)

    return lengths


def verify_augmented_sample(
    original: np.ndarray,
    augmented: np.ndarray,
    scale_factor: float,
) -> bool:
    """Verify that the augmented sample preserves required constraints.

    Returns True if valid, False otherwise.
    """
    # 1. Shape is unchanged
    if original.shape != augmented.shape:
        return False

    # 2. No NaN or Inf
    if not np.all(np.isfinite(augmented)):
        return False

    # 3. Sequence length is unchanged
    if original.shape[0] != augmented.shape[0]:
        return False

    # Detect original sequence scale to adjust absolute thresholds
    seq_scale = _detect_sequence_scale(original)
    thr_diff = 15.0 / seq_scale
    thr_limit = 50.0 / seq_scale

    # 4. Bone lengths remain approximately constant (modulo scaling)
    orig_bones = compute_bone_lengths(original)
    aug_bones = compute_bone_lengths(augmented)

    for hand in orig_bones:
        if hand not in aug_bones:
            continue
        o_b = orig_bones[hand]
        a_b = aug_bones[hand]

        # Expected bone lengths after scaling
        expected_a_b = o_b * scale_factor

        # Difference should be minimal (mostly minor interpolation/noise effects)
        diff = np.abs(a_b - expected_a_b)
        mean_orig = np.mean(o_b)
        if mean_orig > 1e-5:
            rel_diff = diff / mean_orig
            # Maximum 25% relative change or scaled absolute change threshold
            if np.any(rel_diff > 0.25) and np.any(diff > thr_diff):
                return False

    # 5. Coordinate ranges remain within expected limits
    orig_max, orig_min = np.max(original), np.min(original)
    aug_max, aug_min = np.max(augmented), np.min(augmented)

    limit_factor = max(1.2, scale_factor * 1.2)
    if aug_max > orig_max * limit_factor + thr_limit or aug_min < orig_min * limit_factor - thr_limit:
        return False

    # 6. Temporal continuity (no unrealistic jumps)
    orig_diffs = np.linalg.norm(np.diff(original, axis=0), axis=1)
    aug_diffs = np.linalg.norm(np.diff(augmented, axis=0), axis=1)

    if len(orig_diffs) > 0:
        max_orig_diff = np.max(orig_diffs)
        max_aug_diff = np.max(aug_diffs)
        if max_aug_diff > max_orig_diff * limit_factor + thr_limit:
            return False

    return True


class SignLanguageAugmentationPipeline:
    """Pipeline to compose modular data augmentations with configurable probabilities."""

    def __init__(
        self,
        rotation_prob: float = 0.5,
        rotation_range: float = 8.0,
        scaling_prob: float = 0.5,
        scaling_range: tuple[float, float] = (0.95, 1.05),
        noise_prob: float = 0.5,
        noise_std: float = 2.0,
        dropout_prob: float = 0.5,
        dropout_rate: float = 0.08,
        max_consecutive_dropout: int = 2,
        seed: int | None = None,
    ):
        self.rotation_prob = rotation_prob
        self.rotation_range = rotation_range
        self.scaling_prob = scaling_prob
        self.scaling_range = scaling_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.dropout_rate = dropout_rate
        self.max_consecutive_dropout = max_consecutive_dropout

        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        """Allow setting seed for reproducibility."""
        self.rng = np.random.default_rng(seed)

    def __call__(self, sequence: np.ndarray) -> np.ndarray:
        """Apply composed augmentations to the given sequence (T, 132)."""
        if sequence.ndim != 2 or sequence.shape[1] != 132:
            raise ValueError(f"Expected sequence with shape (T, 132), got {sequence.shape}")

        original = sequence.copy()
        augmented = sequence.copy()

        scale_factor = 1.0

        # 1. Random Frame Dropout (run before noise/scaling/rotation)
        if self.rng.random() < self.dropout_prob:
            augmented = random_frame_dropout(
                augmented,
                drop_prob=self.dropout_rate,
                max_consecutive=self.max_consecutive_dropout,
                rng=self.rng,
            )

        # 2. Small Uniform Scaling
        if self.rng.random() < self.scaling_prob:
            scale_factor = self.rng.uniform(self.scaling_range[0], self.scaling_range[1])
            augmented = random_scale_uniform(augmented, scale_factor)

        # 3. Small 3D Rotation
        if self.rng.random() < self.rotation_prob:
            augmented = random_rotate_3d(augmented, self.rotation_range, self.rng)

        # 4. Gaussian Coordinate Noise
        if self.rng.random() < self.noise_prob:
            augmented = add_coordinate_noise(augmented, self.noise_std, self.rng)

        # Validation checks
        if not verify_augmented_sample(original, augmented, scale_factor):
            # Fallback to original sequence on validation failure
            return original

        return augmented
