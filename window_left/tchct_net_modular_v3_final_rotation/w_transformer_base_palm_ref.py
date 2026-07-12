"""
Fixed 30 FPS double-threaded Leap Motion + webcam capture with a separate
inference process for sign language thesis.

Difference vs `single_thread_fixed_fps_inference.py`:
  - Webcam frames are captured on a background thread.
  - The main loop blocks until a *fresh* frame arrives (never consumes stale buffered frames).

Each tick (at 30 FPS) the **main process**:
  1. Reads one fresh webcam frame (bufferless capture)
  2. Grabs the latest Leap Motion hand tracking data
  3. Sends both to the inference process via a multiprocessing Queue

The **inference process** runs continuously:
  - Maintains a fixed-size rolling buffer of recent frames
  - On each new frame: adds it to the buffer, runs prediction using
    the full buffer window (newest frame + last N), then drops the
    oldest frame
"""

import os
import leap
import cv2
import time
import sys
import threading
import queue
import multiprocessing as mp
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


# ──────────────────────────── constants ────────────────────────────
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
BONES = ["metacarpal", "proximal", "intermediate", "distal"]

TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS          # ~0.033 s per tick
DEFAULT_DURATION = 30                        # seconds

# ST-GCN online inference uses notebook-parity left-window size.
STGCN_WINDOW_SIZE = 30

# Inference buffer: how many past frames (including the newest) the
# inference process keeps in its sliding window.
BUFFER_SIZE = STGCN_WINDOW_SIZE

# Minimum number of frames in the buffer before prediction runs.
MIN_BUFFER_FOR_PREDICTION = STGCN_WINDOW_SIZE

# ──────────────────────── model config ─────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_MODELS_ROOT)
STGCN_CHECKPOINT_PATH = os.path.join(_MODELS_ROOT, "models_updated", "trained_models", "transformer_users3.pt")
SENTENCE_OUTPUT_PATH = os.path.join(_MODELS_ROOT, "temp", "live_inference_sentence.txt")

NUM_CLASSES = 21
# Notebook model uses 132 features: palm(3) + wrist(3) + finger bones(60) per hand.
INPUT_DIM = 132
# THCT-Net skeleton constants
T_FRAMES     = 30   # window length
M_ENTITIES   = 2    # left hand, right hand
V_PER_ENT    = 22   # joints per hand  (palm + wrist + 5 fingers × 4 bones)
C_IN         = 3    # x, y, z
NUM_JOINTS   = V_PER_ENT * M_ENTITIES  # 44  (used after early entity-fusion)

# Decoder defaults — must match config.py / decoder.py (SimplifiedBagDecoder),
# NOT the old majority-vote-over-a-window OnlineCausalDecoder this file used
# to carry. bg_exit_frames default kept at 1 (original immediate-exit
# behavior) while still in the experiment phase; bg_exit_frames=5 was
# validated across user1/2/3/5 to lower WER/FPR — pass it explicitly (e.g.
# via checkpoint config/metadata) to use it.
BAG_SIZE = 5
BAG_AGGREGATION = "mean"
CONFIDENCE_THRESHOLD = 0.35
SIGN_BG_MARGIN = 0.10
MIN_SIGN_MS = 500
MIN_SIGN_FRAMES = max(1, round(MIN_SIGN_MS / 1000 * TARGET_FPS))
BG_EXIT_FRAMES = 1

# Sentence-boundary heuristic (demo-only, not part of the sign decoder):
# clear the in-progress sentence after this many consecutive background
# frames (post-bag voted_label), i.e. a longer pause than a normal
# inter-sign gap.
SENTENCE_BREAK_FRAMES = 45   # ~1.5s at 30 FPS

BACKGROUND_LABEL = "background"

# Label mapping  – update with your actual class names
IDX_TO_LABEL = {
    0: "AUGUST",
    1: "BIG",
    2: "BIRD",
    3: "BOAT",
    4: "COME",
    5: "DRIVER",
    6: "FARMING",
    7: "FEBRUARY",
    8: "GO",
    9: "GREETINGS",
    10: "OUR",
    11: "READ",
    12: "SMALL",
    13: "TIGER",
    14: "TRAIN",
    15: "UGLY",
    16: "VAN",
    17: "WHAT",
    18: "WHICH",
    19: "WRITE",
    20: "background",
}

# ── 132 raw-coordinate feature keys (notebook-parity) ──
# Per hand: palm(3) + wrist(3) + 5 fingers×4 bones×3 coords = 66
# Both hands -> 132.
HANDS = ["left", "right"]
CARTESIAN_AXES = ["x", "y", "z"]
START_AXES = ["sx", "sy", "sz"]

FEATURE_KEYS = []

# Palm and wrist coordinates per hand.
for hand in HANDS:
    for part in ["palm", "wrist"]:
        for axis in CARTESIAN_AXES:
            FEATURE_KEYS.append(f"{hand}_{part}_{axis}")

# Finger bone start-joint coordinates per hand.
for hand in HANDS:
    for finger in FINGERS:
        for bone in BONES:
            for axis in START_AXES:
                FEATURE_KEYS.append(f"{hand}_{finger}_{bone}_{axis}")

assert len(FEATURE_KEYS) == INPUT_DIM, f"Expected {INPUT_DIM} features, got {len(FEATURE_KEYS)}"

FEATURE_INDEX = {key: idx for idx, key in enumerate(FEATURE_KEYS)}
PALM_TRIPLETS: dict[str, tuple[int, int, int]] = {}
HAND_POSITION_TRIPLETS: dict[str, list[tuple[int, int, int]]] = {}

for hand in HANDS:
    PALM_TRIPLETS[hand] = tuple(FEATURE_INDEX[f"{hand}_palm_{axis}"] for axis in CARTESIAN_AXES)

    triplets: list[tuple[int, int, int]] = []
    triplets.append(tuple(FEATURE_INDEX[f"{hand}_wrist_{axis}"] for axis in CARTESIAN_AXES))
    for finger in FINGERS:
        for bone in BONES:
            triplets.append(tuple(FEATURE_INDEX[f"{hand}_{finger}_{bone}_{axis}"] for axis in START_AXES))
    HAND_POSITION_TRIPLETS[hand] = triplets


# ──────────────────────────── helpers ──────────────────────────────
def zero_hand_dict(prefix):
    """Return a dict with all hand fields set to 0."""
    d = {}

    # palm
    for k in ["x", "y", "z", "vx", "vy", "vz", "nx", "ny", "nz", "dx", "dy", "dz", "width"]:
        d[f"{prefix}_palm_{k}"] = 0.0

    # arm
    for k in ["wrist_x", "wrist_y", "wrist_z", "elbow_x", "elbow_y", "elbow_z"]:
        d[f"{prefix}_{k}"] = 0.0

    # fingers & bones
    for f in FINGERS:
        for b in BONES:
            for k in ["sx", "sy", "sz", "ex", "ey", "ez", "dx", "dy", "dz", "width"]:
                d[f"{prefix}_{f}_{b}_{k}"] = 0.0

    return d


def extract_hand_row(event):
    """Extract a flat dict of hand data from a TrackingEvent (or None → zeros)."""
    row = {}

    # Defaults – both hands zeroed
    row.update(zero_hand_dict("left"))
    row.update(zero_hand_dict("right"))
    for side in ["left", "right"]:
        row[f"{side}_confidence"] = 0.0
        row[f"{side}_grab_strength"] = 0.0
        row[f"{side}_pinch_strength"] = 0.0

    row["leap_timestamp"] = 0
    row["leap_frame_id"] = 0

    if event is None:
        return row

    row["leap_timestamp"] = event.timestamp
    row["leap_frame_id"] = event.tracking_frame_id

    for hand in event.hands:
        side = "left" if str(hand.type).endswith("Left") else "right"

        row[f"{side}_confidence"] = hand.confidence
        row[f"{side}_grab_strength"] = hand.grab_strength
        row[f"{side}_pinch_strength"] = hand.pinch_strength

        # palm
        row[f"{side}_palm_x"] = hand.palm.position.x
        row[f"{side}_palm_y"] = hand.palm.position.y
        row[f"{side}_palm_z"] = hand.palm.position.z

        row[f"{side}_palm_vx"] = hand.palm.velocity.x
        row[f"{side}_palm_vy"] = hand.palm.velocity.y
        row[f"{side}_palm_vz"] = hand.palm.velocity.z

        row[f"{side}_palm_nx"] = hand.palm.normal.x
        row[f"{side}_palm_ny"] = hand.palm.normal.y
        row[f"{side}_palm_nz"] = hand.palm.normal.z

        row[f"{side}_palm_dx"] = hand.palm.direction.x
        row[f"{side}_palm_dy"] = hand.palm.direction.y
        row[f"{side}_palm_dz"] = hand.palm.direction.z

        row[f"{side}_palm_width"] = hand.palm.width

        # arm  (LeapC: prev = elbow, next = wrist)
        row[f"{side}_wrist_x"] = hand.arm.next_joint.x
        row[f"{side}_wrist_y"] = hand.arm.next_joint.y
        row[f"{side}_wrist_z"] = hand.arm.next_joint.z

        row[f"{side}_elbow_x"] = hand.arm.prev_joint.x
        row[f"{side}_elbow_y"] = hand.arm.prev_joint.y
        row[f"{side}_elbow_z"] = hand.arm.prev_joint.z

        # digits & bones
        for fi, digit in enumerate(hand.digits):
            finger = FINGERS[fi]
            for bi, bone in enumerate(digit.bones):
                bone_name = BONES[bi]

                sx, sy, sz = bone.prev_joint.x, bone.prev_joint.y, bone.prev_joint.z
                ex, ey, ez = bone.next_joint.x, bone.next_joint.y, bone.next_joint.z

                row[f"{side}_{finger}_{bone_name}_sx"] = sx
                row[f"{side}_{finger}_{bone_name}_sy"] = sy
                row[f"{side}_{finger}_{bone_name}_sz"] = sz
                row[f"{side}_{finger}_{bone_name}_ex"] = ex
                row[f"{side}_{finger}_{bone_name}_ey"] = ey
                row[f"{side}_{finger}_{bone_name}_ez"] = ez
                row[f"{side}_{finger}_{bone_name}_dx"] = ex - sx
                row[f"{side}_{finger}_{bone_name}_dy"] = ey - sy
                row[f"{side}_{finger}_{bone_name}_dz"] = ez - sz
                row[f"{side}_{finger}_{bone_name}_width"] = bone.width

    return row


# ───────────────────── THCT-Net Model ─────────────────────

# The raw 132-D feature vector is NOT laid out as [hand0's 66][hand1's 66]:
# FEATURE_KEYS interleaves both hands' palm/wrist first (12 values), then all
# of hand0's finger bones, then all of hand1's finger bones. This permutation
# gathers each hand's 22 slots (palm, wrist, 20 finger bones) into a
# contiguous 66-value block, in HANDS order — matches model.py exactly.
_HAND_FEATURE_PERM: list[int] = []
for _hand in HANDS:
    _HAND_FEATURE_PERM.extend(PALM_TRIPLETS[_hand])
    for _triplet in HAND_POSITION_TRIPLETS[_hand]:
        _HAND_FEATURE_PERM.extend(_triplet)

assert len(_HAND_FEATURE_PERM) == INPUT_DIM, (
    f"Expected {INPUT_DIM} permuted feature indices, got {len(_HAND_FEATURE_PERM)}"
)


def _gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    GroupNorm factory matching model.py — behaves identically at train and
    inference time regardless of batch size (unlike BatchNorm, which falls
    back to running_mean/running_var at inference and can mismatch
    training-time behavior for batch_size=1 streaming).
    """
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _to_skeleton_tensor(x: Tensor) -> Tensor:
    """
    x : (B, T, 132)  — palm-normalised flat features
    returns (B, 3, T, 22, 2)

    Applies _HAND_FEATURE_PERM before reshaping so each hand's 22 joints end
    up contiguous — matches model.py's _to_skeleton_tensor exactly.
    """
    B, T, D = x.shape
    x = x[:, :, _HAND_FEATURE_PERM]
    # (B, T, 132) → (B, T, M=2, V=22, C=3)
    x = x.reshape(B, T, M_ENTITIES, V_PER_ENT, C_IN)
    # → (B, C=3, T, V=22, M=2)
    x = x.permute(0, 4, 1, 3, 2).contiguous()
    return x


class _ISATABlock(nn.Module):
    """
    Multi-Head Self-Attention with trainable regularisation:
        score = α · tanh(QKᵀ / √Cβ) + A
    where A ∈ R^{U×U} is a learnable bias matrix and α is a scalar.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Identity()

        # Learnable regularisation (eq. 6)
        self.alpha = nn.Parameter(torch.ones(1))
        self.A     = nn.Parameter(torch.zeros(num_heads, 1, 1))

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, U, d_model)"""
        B, U, d = x.shape
        H, Dh   = self.num_heads, self.head_dim

        residual = x
        x_norm   = self.norm1(x)

        Q = self.q_proj(x_norm).reshape(B, U, H, Dh).transpose(1, 2)
        K = self.k_proj(x_norm).reshape(B, U, H, Dh).transpose(1, 2)
        V = x_norm.reshape(B, U, H, Dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
        scores = self.alpha * torch.tanh(scores) + self.A
        attn   = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, U, d) + residual

        out = out + self.ffn(self.norm2(out))
        return out


class TransformerStream(nn.Module):
    """
    Input  : (B, C=3, T=30, V=22, M=2)
    Tokens : 3D sliding window Tw×Vw×Mw  →  U tokens of dim d_model
    Blocks : L × ISATABlock
    Output : (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        d_model:     int   = 128,
        num_heads:   int   = 4,
        num_layers:  int   = 4,
        Tw:          int   = 5,
        Vw:          int   = 2,
        Mw:          int   = 1,
        dropout:     float = 0.1,
        window_size: int   = 30,
    ) -> None:
        super().__init__()
        self.Tw, self.Vw, self.Mw = Tw, Vw, Mw
        self.d_model = d_model

        # BatchNorm3d → GroupNorm: identical behavior at B=1 inference.
        self.token_embed = nn.Sequential(
            nn.Conv3d(C_IN, d_model,
                      kernel_size=(Tw, Vw, Mw),
                      stride=(Tw, Vw, Mw), bias=False),
            _gn(d_model),
            nn.GELU(),
        )

        self.nT = window_size // Tw
        self.nV = V_PER_ENT   // Vw
        self.nM = M_ENTITIES  // Mw
        self.num_tokens = self.nT * self.nV * self.nM

        self.pos_enc = nn.Parameter(
            torch.zeros(1, self.num_tokens, d_model)
        )
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        self.blocks = nn.ModuleList([
            _ISATABlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.temporal_agg = nn.Conv3d(
            d_model, d_model,
            kernel_size=(min(5, self.nT), 1, 1),
            padding=(min(5, self.nT) // 2, 0, 0),
        )
        self.bn_agg = _gn(d_model)

        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, 3, T, V, M)"""
        B = x.size(0)

        tokens = self.token_embed(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_enc

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = (tokens
                  .transpose(1, 2)
                  .reshape(B, self.d_model, self.nT, self.nV, self.nM))
        tokens = F.gelu(self.bn_agg(self.temporal_agg(tokens)))

        out = self.gap(tokens).flatten(1)
        return self.head(out)


class _CNNBranch(nn.Module):
    """
    Single branch (raw S or motion M).
    Input  : (B, C=3, T=30, V_total=44)
    """

    def __init__(self, base_ch: int = 64) -> None:
        super().__init__()

        # BatchNorm2d → GroupNorm throughout this branch.
        self.enc1 = nn.Sequential(
            nn.Conv2d(C_IN, base_ch, kernel_size=(1, 1), bias=False),
            _gn(base_ch), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=(3, 1),
                      padding=(1, 0), bias=False),
            _gn(base_ch), nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(NUM_JOINTS, base_ch, kernel_size=(3, 3),
                      padding=(1, 1), stride=(2, 2), bias=False),
            _gn(base_ch), nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=(3, 3),
                      padding=(1, 1), stride=(2, 2), bias=False),
            _gn(base_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, 3, T, V*M=44)"""
        x = self.enc1(x)
        x = self.enc2(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.enc3(x)
        x = self.enc4(x)
        return x


class _ResidualFusion(nn.Module):
    """
    Fuses concatenated dual-branch features via asymmetric 1×7 / 7×1 convs
    + residual shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int = 128) -> None:
        super().__init__()
        # BatchNorm2d → GroupNorm on both the main path and the shortcut.
        self.path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 7),
                      padding=(0, 3), bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(7, 1),
                      padding=(3, 0), bias=False),
            _gn(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                _gn(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.path(x) + self.shortcut(x))


class CNNStream(nn.Module):
    """
    Full CNN stream.
    Input  : (B, C=3, T=30, V=22, M=2)
    Output : (B, num_classes)
    """

    def __init__(self, num_classes: int, base_ch: int = 64) -> None:
        super().__init__()
        self.branch_S = _CNNBranch(base_ch)
        self.branch_M = _CNNBranch(base_ch)

        self.fusion = _ResidualFusion(base_ch * 2, out_ch=128)

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(128, 256)
        self.drop = nn.Dropout(0.3)
        self.fc2  = nn.Linear(256, num_classes)

    @staticmethod
    def _temporal_diff(x: Tensor) -> Tensor:
        """Frame-to-frame difference: M_t = S_{t+1} − S_t  (eq. 7)"""
        diff = x[:, :, 1:, :] - x[:, :, :-1, :]
        return F.pad(diff, (0, 0, 0, 1))

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, 3, T, V=22, M=2)"""
        B, C, T, V, M = x.shape

        x_flat = x.reshape(B, C, T, V * M)
        x_motion = self._temporal_diff(x_flat)

        feat_S = self.branch_S(x_flat)
        feat_M = self.branch_M(x_motion)

        if feat_S.shape != feat_M.shape:
            feat_M = F.interpolate(feat_M, size=feat_S.shape[2:])

        fused = torch.cat([feat_S, feat_M], dim=1)
        fused = self.fusion(fused)

        out = self.gap(fused).flatten(1)
        out = self.drop(F.relu(self.fc1(out)))
        return self.fc2(out)


class THCTNet(nn.Module):
    """
    Two-stream Hybrid CNN-Transformer Network — Leap Motion edition.

    Drop-in replacement for any model in your pipeline.
    Call signature:  logits = model(sequences, lengths)
                     sequences : (B, 30, 132)  palm-normalised flat features
                     lengths   : (B,)          accepted, ignored (fixed-length windows)
    """

    def __init__(
        self,
        num_classes: int,
        d_model:     int   = 128,
        num_heads:   int   = 4,
        num_layers:  int   = 4,
        base_ch:     int   = 64,
        window_size: int   = 30,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size

        self.cnn_stream  = CNNStream(num_classes, base_ch)
        self.trans_stream = TransformerStream(
            num_classes, d_model, num_heads, num_layers, window_size=window_size
        )

        # Learnable fusion weight in (0,1) via sigmoid; starts at 0.5
        self._raw_w = nn.Parameter(torch.zeros(1))

    def _fusion_weight(self) -> Tensor:
        return torch.sigmoid(self._raw_w)

    def forward(self, sequences: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        sequences : (B, T=30, D=132)   ← exactly what your DataLoader yields
        lengths   : (B,)               ← accepted for API compatibility, not used
        returns   : (B, num_classes)
        """
        x = _to_skeleton_tensor(sequences)

        logits_cnn  = self.cnn_stream(x)
        logits_trans = self.trans_stream(x)

        w   = self._fusion_weight().to(x.device)
        return w * logits_cnn + (1.0 - w) * logits_trans

    def forward_streams(self, sequences: Tensor):
        x = _to_skeleton_tensor(sequences)
        return self.cnn_stream(x), self.trans_stream(x)


def minmax_normalize_window(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-window min-max normalization, same as notebook training path."""
    w_min = window.min(axis=0, keepdims=True)
    w_max = window.max(axis=0, keepdims=True)
    denom = w_max - w_min
    denom = np.where(denom < eps, 1.0, denom)
    return ((window - w_min) / denom).astype(np.float32)


def palm_reference_normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize one frame by subtracting each hand's palm from that hand's positions."""
    # (Kept for compatibility, though sequence normalization is preferred)
    out = np.asarray(frame, dtype=np.float32).copy()
    if out.shape[0] != INPUT_DIM:
        raise ValueError(f"Expected frame with {INPUT_DIM} features, got {out.shape[0]}")

    for hand in HANDS:
        px, py, pz = PALM_TRIPLETS[hand]
        palm = out[[px, py, pz]].copy()
        if not np.all(np.isfinite(palm)):
            palm = np.zeros((3,), dtype=np.float32)

        for ix, iy, iz in HAND_POSITION_TRIPLETS[hand]:
            out[ix] -= palm[0]
            out[iy] -= palm[1]
            out[iz] -= palm[2]

        out[px] = 0.0
        out[py] = 0.0
        out[pz] = 0.0

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def palm_reference_normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Apply palm-reference normalization to a sequence (T, D) optimally."""
    seq = np.asarray(sequence, dtype=np.float32).copy()
    if seq.ndim != 2:
        raise ValueError(f"Expected sequence with shape (T, D), got {seq.shape}")
    if seq.shape[0] == 0:
        return seq

    for hand in HANDS:
        px, py, pz = PALM_TRIPLETS[hand]
        palm = seq[:, [px, py, pz]].copy()
        
        bad_mask = ~np.all(np.isfinite(palm), axis=1)
        palm[bad_mask] = 0.0

        x_idx = [t[0] for t in HAND_POSITION_TRIPLETS[hand]]
        y_idx = [t[1] for t in HAND_POSITION_TRIPLETS[hand]]
        z_idx = [t[2] for t in HAND_POSITION_TRIPLETS[hand]]
        
        seq[:, x_idx] -= palm[:, 0:1]
        seq[:, y_idx] -= palm[:, 1:2]
        seq[:, z_idx] -= palm[:, 2:3]

        seq[:, px] = 0.0
        seq[:, py] = 0.0
        seq[:, pz] = 0.0

    return np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_first(payload: dict, keys: list[str], required: bool = True):
    for key in keys:
        if key in payload:
            return payload[key]
    if required:
        raise KeyError(f"Missing required key. Tried: {keys}")
    return None


def _parse_id_to_label(raw_mapping, num_classes: int) -> dict[int, str]:
    # Supports either {id: label} or {label: id} dictionaries.
    if isinstance(raw_mapping, dict):
        parsed = {}

        parsed_as_id_to_label = {}
        try:
            for k, v in raw_mapping.items():
                parsed_as_id_to_label[int(k)] = str(v)
        except Exception:
            parsed_as_id_to_label = {}

        if parsed_as_id_to_label:
            parsed = parsed_as_id_to_label
        else:
            for maybe_label, maybe_idx in raw_mapping.items():
                try:
                    parsed[int(maybe_idx)] = str(maybe_label)
                except Exception:
                    continue
    elif isinstance(raw_mapping, list):
        parsed = {i: str(v) for i, v in enumerate(raw_mapping)}
    else:
        parsed = {}

    for cls_id in range(num_classes):
        parsed.setdefault(cls_id, f"sign_{cls_id}")
    return parsed


def _strip_module_prefix_if_needed(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _label_to_token(label: str) -> str:
    """Convert model class label to a readable token for sentence building."""
    return str(label).strip().replace("_", " ")


def _tokens_to_sentence(tokens: list[str], with_period: bool = False) -> str:
    """Join emitted tokens into a readable sentence."""
    sentence = " ".join(t for t in tokens if t).strip()
    if not sentence:
        return ""
    sentence = sentence[0].upper() + sentence[1:]
    if with_period and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _infer_thctnet_arch_from_state_dict(state_dict: dict) -> dict[str, int]:
    """Infer key THCT-Net dimensions from checkpoint tensors when metadata is absent."""
    inferred: dict[str, int] = {}

    # Try CNN stream head
    if "cnn_stream.fc2.weight" in state_dict and isinstance(state_dict["cnn_stream.fc2.weight"], torch.Tensor):
        inferred["num_classes"] = int(state_dict["cnn_stream.fc2.weight"].shape[0])
    # Fallback: try Transformer stream head
    elif "trans_stream.head.weight" in state_dict and isinstance(state_dict["trans_stream.head.weight"], torch.Tensor):
        inferred["num_classes"] = int(state_dict["trans_stream.head.weight"].shape[0])

    return inferred


def load_stgcn_runtime_checkpoint(path: str) -> dict:
    """Load ST-GCN runtime checkpoint from notebook-style payload."""
    if not str(path).strip():
        raise ValueError("STGCN_CHECKPOINT_PATH is empty. Set it to your trained ST-GCN checkpoint file.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"ST-GCN checkpoint not found at '{path}'.")

    payload = torch.load(path, map_location="cpu", weights_only=False)

    config = {}
    metadata = {}
    model_state = None
    raw_label_map = IDX_TO_LABEL
    has_explicit_label_map = False

    if isinstance(payload, dict):
        config = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
        model_state = _extract_first(payload, ["model_state_dict", "state_dict", "model"], required=False)
        raw_label_map = payload.get(
            "id_to_label",
            payload.get(
                "label_to_id",
                config.get(
                    "id_to_label",
                    config.get(
                        "label_to_id",
                        metadata.get("id_to_label", metadata.get("label_to_id", IDX_TO_LABEL)),
                    ),
                ),
            ),
        )
        has_explicit_label_map = any(
            key in payload for key in ("id_to_label", "label_to_id")
        ) or any(
            key in config for key in ("id_to_label", "label_to_id")
        ) or any(
            key in metadata for key in ("id_to_label", "label_to_id")
        )
        if model_state is None and all(isinstance(v, torch.Tensor) for v in payload.values()):
            model_state = payload

    if not isinstance(model_state, dict):
        raise ValueError("Could not find a valid model state_dict in the checkpoint.")

    model_state = _strip_module_prefix_if_needed(model_state)
    inferred = _infer_thctnet_arch_from_state_dict(model_state)

    if isinstance(payload, dict):
        num_classes = int(payload.get("num_classes", config.get("num_classes", metadata.get("num_classes", inferred.get("num_classes", NUM_CLASSES)))))
    else:
        num_classes = int(inferred.get("num_classes", NUM_CLASSES))
    id_to_label = _parse_id_to_label(raw_label_map, num_classes=num_classes)

    if not has_explicit_label_map:
        print(
            "[Inference][Warning] Checkpoint has no id_to_label/label_to_id metadata. "
            "Falling back to built-in IDX_TO_LABEL; verify label order matches training LabelEncoder classes_."
        )

    return {
        "model_state_dict": model_state,
        "input_dim": int(config.get("input_dim", metadata.get("input_dim", INPUT_DIM))),
        "num_classes": num_classes,
        "d_model": int(config.get("d_model", metadata.get("d_model", 128))),
        "num_heads": int(config.get("num_heads", metadata.get("num_heads", 4))),
        "num_layers": int(config.get("num_layers", metadata.get("num_layers", 4))),
        "base_ch": int(config.get("base_ch", metadata.get("base_ch", 64))),
        "window_size": int(config.get("window_size", metadata.get("window_size", STGCN_WINDOW_SIZE))),
        "id_to_label": id_to_label,
        "bag_size": max(1, int(config.get("bag_size", metadata.get("bag_size", BAG_SIZE)))),
        "aggregation": str(config.get("aggregation", metadata.get("aggregation", BAG_AGGREGATION))),
        "confidence_threshold": float(config.get("confidence_threshold", metadata.get("confidence_threshold", CONFIDENCE_THRESHOLD))),
        "sign_bg_margin": float(config.get("sign_bg_margin", metadata.get("sign_bg_margin", SIGN_BG_MARGIN))),
        "min_sign_frames": max(1, int(config.get("min_sign_frames", metadata.get("min_sign_frames", MIN_SIGN_FRAMES)))),
        "bg_exit_frames": max(1, int(config.get("bg_exit_frames", metadata.get("bg_exit_frames", BG_EXIT_FRAMES)))),
        "background_label": str(config.get("background_label", BACKGROUND_LABEL)),
    }


# ───────────────── Streaming decoder (ported from decoder.py) ──────────────
# CAUTION: this is the core inference engine — the same bag-aggregated
# hysteresis decoder (_BagAggregator, SimplifiedBagDecoder) validated in
# visualize_decoder.ipynb. Any changes here directly affect live WER.
# Kept as a standalone copy here (not imported) so this demo file has no
# runtime dependency on decoder.py beyond matching its logic.

class _BagAggregator:
    """
    Causal sliding bag over raw logits.

    Why logits and not probs:
        Averaging in logit space is equivalent to a product-of-experts,
        which is sharper and more discriminative than averaging softmax probs.
        Converting to probs happens once after aggregation.

    Modes
    -----
    mean      : arithmetic mean of per-window probs after softmax
    max       : element-wise max of per-window probs
    attention : recency-weighted mean, most recent window weighted highest
    """

    def __init__(self, bag_size: int, aggregation: str, num_classes: int):
        self.bag_size    = max(1, int(bag_size))
        self.aggregation = aggregation
        self.num_classes = num_classes
        self._buffer     = deque(maxlen=self.bag_size)

    def update(self, logits: np.ndarray):
        """Push one logit vector, return aggregated probs (None until bag is full)."""
        self._buffer.append(logits.copy())

        if len(self._buffer) < self.bag_size:
            return None

        bag         = np.stack(self._buffer, axis=0)           # (bag_size, C)
        bag_shifted = bag - bag.max(axis=-1, keepdims=True)
        exp_bag     = np.exp(bag_shifted)
        probs       = exp_bag / exp_bag.sum(axis=-1, keepdims=True)  # (bag_size, C)

        if self.aggregation == "mean":
            return probs.mean(axis=0)
        if self.aggregation == "max":
            return probs.max(axis=0)
        if self.aggregation == "attention":
            weights  = np.linspace(0.5, 1.0, len(self._buffer))
            weights /= weights.sum()
            return (probs * weights[:, np.newaxis]).sum(axis=0)

        raise ValueError(f"Unknown aggregation mode: {self.aggregation}")

    def reset(self):
        self._buffer.clear()


class SimplifiedBagDecoder:
    """
    Causal streaming decoder using bag-aggregated logits.

    States
    ------
    SEEKING : waiting for a sign to begin
    IN_SIGN : inside an active sign region, accumulating votes

    Emission
    --------
    Fires at the TRAILING edge once bg_exit_frames consecutive background
    frames have been seen (default 1 = original immediate-exit behavior;
    bg_exit_frames=5 was validated across user1/2/3/5 to lower WER/FPR).
    Emits the majority label observed across the entire region. Discards
    regions shorter than min_sign_frames (noise / glitches).
    """

    def __init__(
        self,
        id_to_label: dict,
        background_label: str,
        bag_size: int               = BAG_SIZE,
        aggregation: str            = BAG_AGGREGATION,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        sign_bg_margin: float       = SIGN_BG_MARGIN,
        min_sign_frames: int        = MIN_SIGN_FRAMES,
        bg_exit_frames: int         = BG_EXIT_FRAMES,
    ):
        self.id_to_label          = id_to_label
        self.background_label     = background_label
        self.confidence_threshold = float(confidence_threshold)
        self.sign_bg_margin       = float(sign_bg_margin)
        self.min_sign_frames      = max(1, int(min_sign_frames))
        self.bg_exit_frames       = max(1, int(bg_exit_frames))

        self.background_id = next(
            (k for k, v in id_to_label.items() if v == background_label), None
        )

        self._bag = _BagAggregator(bag_size, aggregation, len(id_to_label))

        # Hysteresis state
        self.state              = "SEEKING"
        self.region_votes       = Counter()
        self.sign_frames        = 0
        self.region_start_frame = None      # frame where current IN_SIGN region began
        self.bg_streak          = 0         # consecutive background frames seen while IN_SIGN

    # ------------------------------------------------------------------

    def _gate(self, agg_probs: np.ndarray):
        """Apply confidence gate to aggregated probabilities."""
        pred_id    = int(np.argmax(agg_probs))
        pred_label = self.id_to_label.get(pred_id, f"sign_{pred_id}")
        pred_conf  = float(agg_probs[pred_id])
        bg_conf    = (
            float(agg_probs[self.background_id])
            if self.background_id is not None else 0.0
        )

        is_sign = (
            pred_label != self.background_label
            and pred_conf  >= self.confidence_threshold
            and (pred_conf - bg_conf) >= self.sign_bg_margin
        )

        voted_label   = pred_label if is_sign else self.background_label
        is_background = not is_sign

        return voted_label, is_background, pred_label, pred_conf, bg_conf, agg_probs

    # ------------------------------------------------------------------

    def update(self, logits: np.ndarray, frame_index: int) -> dict:
        """
        Process one frame.

        Parameters
        ----------
        logits      : (C,) raw logits from model — NOT softmaxed (the bag
                      aggregator does its own softmax internally).
        frame_index : int current frame index, needed for emit_region tracking
        """
        pre_bag_logits = logits.copy()
        agg_probs      = self._bag.update(logits)

        # Bag not full yet — stay in SEEKING, emit nothing
        if agg_probs is None:
            raw_probs = np.exp(logits - logits.max())
            raw_probs /= raw_probs.sum()
            return {
                "raw_label":      self.id_to_label.get(int(np.argmax(logits)), "?"),
                "raw_conf":       float(raw_probs.max()),
                "bg_conf":        0.0,
                "gated_label":    self.background_label,
                "voted_label":    self.background_label,
                "state":          self.state,
                "emitted_label":  None,
                "emit_region":    None,
                "pre_bag_logits": pre_bag_logits,
                "post_bag_probs": None,
            }

        voted_label, is_background, pred_label, pred_conf, bg_conf, agg_probs = \
            self._gate(agg_probs)

        emitted_label = None
        emit_region   = None

        if self.state == "SEEKING":
            if not is_background:
                self.state              = "IN_SIGN"
                self.region_votes[voted_label] += 1
                self.sign_frames        = 1
                self.region_start_frame = frame_index
                self.bg_streak          = 0

        elif self.state == "IN_SIGN":
            if not is_background:
                self.bg_streak = 0
                self.region_votes[voted_label] += 1
                self.sign_frames += 1
            else:
                # Background frame — only exit after bg_exit_frames consecutive
                # background frames (default 1 = immediate exit, matches original).
                self.bg_streak += 1
                if self.bg_streak >= self.bg_exit_frames:
                    if self.sign_frames >= self.min_sign_frames:
                        emitted_label = self.region_votes.most_common(1)[0][0]
                        emit_region   = (
                            self.region_start_frame,
                            frame_index,
                            emitted_label,
                        )
                    # else: region too short → discard silently

                    self.state              = "SEEKING"
                    self.region_votes       = Counter()
                    self.sign_frames        = 0
                    self.region_start_frame = None
                    self.bg_streak          = 0
                # else: still within the background grace period — stay IN_SIGN,
                # this frame is not counted as a vote

        return {
            "raw_label":      pred_label,
            "raw_conf":       pred_conf,
            "bg_conf":        bg_conf,
            "gated_label":    voted_label,
            "voted_label":    voted_label,
            "state":          self.state,
            "emitted_label":  emitted_label,
            "emit_region":    emit_region,
            "pre_bag_logits": pre_bag_logits,
            "post_bag_probs": agg_probs,
        }

    # ------------------------------------------------------------------

    def flush(self):
        """Call once at stream end; emits any sign region still open."""
        emitted     = None
        emit_region = None

        if self.state == "IN_SIGN" and self.sign_frames >= self.min_sign_frames:
            emitted     = self.region_votes.most_common(1)[0][0]
            emit_region = (self.region_start_frame, None, emitted)

        self.state              = "SEEKING"
        self.region_votes       = Counter()
        self.sign_frames        = 0
        self.region_start_frame = None
        self.bg_streak          = 0
        self._bag.reset()

        return emitted, emit_region


# ──────────── feature extraction & normalization ───────────────────
import operator

# Pre-compile the dictionary key getter using a fast C-backend lookup.
# This eliminates Python loop overhead for dictionary key extraction.
_fast_feature_getter = operator.itemgetter(*FEATURE_KEYS)

def extract_leap_features(hand_data):
    """Extract notebook-parity 132 raw coordinate features optimally."""
    # Since `extract_hand_row` guarantees all keys via `zero_hand_dict`,
    # we can bypass the slow Python list-comprehension and `.get(k, 0)` entirely.
    return np.array(_fast_feature_getter(hand_data), dtype=np.float32)


def normalize_sequence(seq):
    """Per-sequence min-max normalization to [0, 1], matching GestureDataset."""
    seq_min = seq.min(axis=0)
    seq_max = seq.max(axis=0)
    seq_range = seq_max - seq_min
    seq_range[seq_range < 1e-8] = 1.0  # avoid div-by-zero
    return (seq - seq_min) / seq_range


# ──────────────────────── Leap listener ────────────────────────────
class LatestFrameListener(leap.Listener):
    """Caches the most recent tracking event so the main loop can grab it."""

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._latest_event = None
        self.connected = False
        self.device_serial = None

    def on_connection_event(self, event):
        self.connected = True
        print("Connected to Leap Motion")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        self.device_serial = info.serial
        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        with self._lock:
            self._latest_event = event

    def get_latest(self):
        """Return the most recent tracking event (may be None if none received yet)."""
        with self._lock:
            return self._latest_event


# ──────────────────────── camera helpers ───────────────────────────
def find_working_camera():
    """
    Find a working camera.  Tries camera index 1 first (common on Windows
    when an IR camera occupies index 0), then falls back to index 0.

    Returns:
        cv2.VideoCapture or None
    """
    if sys.platform == "win32":
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("Camera 1 opened successfully")
                return cap
            cap.release()

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            print("Camera 0 opened successfully")
            return cap
        cap.release()

    return None


class BufferlessVideoCapture:
    """
    Background camera reader that keeps only the latest frame.

    - Reader thread continuously calls `cap.read()`
    - A 1-slot queue stores only the most recent frame
    - `read()` blocks until a frame is available and *consumes* it
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._reader, name="BufferlessVideoCaptureReader")
        self._thread.daemon = True
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Keep only most recent unconsumed frame
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout_s=None):
        """
        Block until a frame is available, then consume it.

        Returns:
            (ret, frame)
        """
        if timeout_s is None:
            frame = self._q.get()
            if frame is None:
                return False, None
            return True, frame

        try:
            frame = self._q.get(timeout=timeout_s)
        except queue.Empty:
            return False, None
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        self._stop.set()
        # Unblock any waiting consumer
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)
        self._cap.release()

    def get(self, prop_id):
        return self._cap.get(prop_id)

    def set(self, prop_id, value):
        return self._cap.set(prop_id, value)


# ──────────────────── inference process ────────────────────────────
def inference_process(frame_queue: mp.Queue, buffer_size: int, stop_event, ready_event):
    """
    Consumer process: receives (frame_number, hand_data, cam_frame) from
    the main loop, maintains a sliding buffer of Leap features, and runs
    THCT-Net inference.

    Camera frames are received but NOT used for model inference (RGB is
    stored/forwarded only; no MediaPipe processing for now).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = load_stgcn_runtime_checkpoint(STGCN_CHECKPOINT_PATH)

    if runtime["window_size"] != buffer_size:
        raise ValueError(
            f"Buffer/window mismatch: buffer_size={buffer_size}, "
            f"checkpoint_window_size={runtime['window_size']}."
        )

    model = THCTNet(
        num_classes=runtime["num_classes"],
        d_model=runtime["d_model"],
        num_heads=runtime["num_heads"],
        num_layers=runtime["num_layers"],
        base_ch=runtime["base_ch"],
        window_size=runtime["window_size"],
    ).to(device)

    model.load_state_dict(runtime["model_state_dict"], strict=True)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Inference] Device: {device}  |  Parameters: {total_params:,}")
    print(f"[Inference] Loaded THCT-Net checkpoint from: {STGCN_CHECKPOINT_PATH}")
    print(f"[Inference] Live sentence output file: {SENTENCE_OUTPUT_PATH}")
    print(
        "[Inference] Decoder config: "
        f"bag_size={runtime['bag_size']}, aggregation={runtime['aggregation']}, "
        f"conf_thr={runtime['confidence_threshold']:.2f}, "
        f"margin={runtime['sign_bg_margin']:.2f}, "
        f"min_sign_frames={runtime['min_sign_frames']}, "
        f"bg_exit_frames={runtime['bg_exit_frames']}, "
        f"background='{runtime['background_label']}'"
    )

    decoder = SimplifiedBagDecoder(
        id_to_label=runtime["id_to_label"],
        background_label=runtime["background_label"],
        bag_size=runtime["bag_size"],
        aggregation=runtime["aggregation"],
        confidence_threshold=runtime["confidence_threshold"],
        sign_bg_margin=runtime["sign_bg_margin"],
        min_sign_frames=runtime["min_sign_frames"],
        bg_exit_frames=runtime["bg_exit_frames"],
    )

    os.makedirs(os.path.dirname(SENTENCE_OUTPUT_PATH), exist_ok=True)
    sentence_tokens: list[str] = []
    all_emitted_tokens: list[str] = []
    last_emitted_token = ""
    sentence_bg_streak = 0

    def write_sentence_output(last_event: str):
        current_sentence = _tokens_to_sentence(sentence_tokens, with_period=False)
        accumulated_sentence = _tokens_to_sentence(all_emitted_tokens, with_period=False)
        with open(SENTENCE_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write("Live THCT-Net sentence output\n")
            f.write(f"updated_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"last_event: {last_event}\n\n")

            f.write("Last emitted token:\n")
            f.write((last_emitted_token if last_emitted_token else "(none)") + "\n\n")

            f.write("Accumulated sentence (this run):\n")
            f.write((accumulated_sentence if accumulated_sentence else "(empty)") + "\n\n")

            f.write("Current sentence (in progress):\n")
            f.write((current_sentence if current_sentence else "(empty)") + "\n\n")

    write_sentence_output("initialized")

    # TorchScript JIT tracing completely removes Python execution overhead for the model's graph convolutions
    # allowing inference to execute far faster natively in C++/CUDA.
    dummy_x = torch.zeros(1, buffer_size, runtime["input_dim"], device=device)
    dummy_len = torch.tensor([buffer_size], dtype=torch.long, device=device)
    
    with torch.no_grad():
        model = torch.jit.trace(model, (dummy_x, dummy_len))
        model(dummy_x, dummy_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print("[Inference] Warm-up & JIT compilation complete")

    ready_event.set()

    buffer = deque(maxlen=buffer_size)
    prediction_count = 0
    frames_received = 0
    
    # Enable CuDNN benchmark for optimized convolutions
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    # TRICK: To avoid torch.roll() copying memory entirely, we allocate a 2x sized buffer.
    # By writing each new frame into both `[idx]` and `[idx + buffer_size]`, 
    # any full chronologically sequential window is just a 0-copy view!
    double_buffer = torch.zeros((1, buffer_size * 2, runtime["input_dim"]), dtype=torch.float32, device=device)
    lengths_tensor = torch.tensor([buffer_size], dtype=torch.long, device=device)
    write_idx = 0
    
    # Pre-compute GPU indices for ultra-fast native device palm normalization
    # This prevents the need to run python normalization and pushes it instantly to VRAM
    gpu_idxs = {}
    for hand in HANDS:
        gpu_idxs[f"{hand}_palm"] = torch.tensor(PALM_TRIPLETS[hand], device=device)
        gpu_idxs[f"{hand}_x"] = torch.tensor([t[0] for t in HAND_POSITION_TRIPLETS[hand]], device=device)
        gpu_idxs[f"{hand}_y"] = torch.tensor([t[1] for t in HAND_POSITION_TRIPLETS[hand]], device=device)
        gpu_idxs[f"{hand}_z"] = torch.tensor([t[2] for t in HAND_POSITION_TRIPLETS[hand]], device=device)
    
    print(f"[Inference] Process started  (buffer_size={buffer_size}, "
          f"min_for_prediction={MIN_BUFFER_FOR_PREDICTION})")

    while True:
        try:
            item = frame_queue.get(timeout=0.1)
        except Exception:
            if stop_event.is_set():
                break
            continue

        if item is None:
            break

        frame_number, hand_data, cam_frame = item
        _ = cam_frame
        frames_received += 1

        features = extract_leap_features(hand_data)
        if features.shape[0] != runtime["input_dim"]:
            raise ValueError(
                f"Feature size mismatch: extracted {features.shape[0]} dims, "
                f"but model expects {runtime['input_dim']} dims."
            )
            
        # Push raw features instantly to GPU
        features_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        
        # Native GPU Palm Normalization (Zero CPU/Numpy overhead)
        for hand in HANDS:
            palm = features_tensor[gpu_idxs[f"{hand}_palm"]]
            valid = torch.isfinite(palm).all()
            palm_val = palm if valid else torch.zeros(3, device=device)
            
            features_tensor[gpu_idxs[f"{hand}_x"]] -= palm_val[0]
            features_tensor[gpu_idxs[f"{hand}_y"]] -= palm_val[1]
            features_tensor[gpu_idxs[f"{hand}_z"]] -= palm_val[2]
            features_tensor[gpu_idxs[f"{hand}_palm"]] = 0.0
            
        # Double-buffer ring write (Eliminates torch.roll and .cat memory copies)
        double_buffer[0, write_idx, :] = features_tensor
        double_buffer[0, write_idx + buffer_size, :] = features_tensor
        
        # O(1) 0-copy chronological rolling view! Memory stays completely stationary
        start_idx = write_idx + 1
        window_view = double_buffer[:, start_idx : start_idx + buffer_size, :]
        
        # Modulo advance
        write_idx = (write_idx + 1) % buffer_size

        buffer.append({
            "frame_number": frame_number,
        })

        newest_frame = buffer[-1]["frame_number"]
        if len(buffer) >= MIN_BUFFER_FOR_PREDICTION:
            # torch.inference_mode() is an extreme optimization over torch.no_grad()
            with torch.inference_mode():
                # We safely pass the chronological view directly to the TorchScript model.
                # NOTE: raw logits, NOT softmaxed — SimplifiedBagDecoder's internal
                # _BagAggregator does its own softmax after bag-averaging in logit space.
                logits = model(window_view, lengths_tensor)
                logits_np = logits[0].detach().cpu().numpy().astype(np.float32)

            decoded = decoder.update(logits_np, frame_index=newest_frame)

            sentence_updated = False
            if decoded["voted_label"] == runtime["background_label"]:
                sentence_bg_streak += 1
            else:
                sentence_bg_streak = 0

            emitted = decoded["emitted_label"]
            if emitted is not None:
                token = _label_to_token(emitted)
                if token and (not sentence_tokens or sentence_tokens[-1] != token):
                    sentence_tokens.append(token)
                    all_emitted_tokens.append(token)
                    last_emitted_token = token
                    sentence_updated = True

            # Treat a longer pause (SENTENCE_BREAK_FRAMES, distinct from the
            # decoder's own bg_exit_frames sign-boundary hysteresis) as an
            # end-of-sentence boundary.
            if sentence_bg_streak >= SENTENCE_BREAK_FRAMES and sentence_tokens:
                sentence_tokens.clear()
                sentence_bg_streak = 0
                sentence_updated = True

            if sentence_updated:
                write_sentence_output("sentence_updated")

            prediction_count += 1
            oldest_frame = buffer[0]["frame_number"]
            if emitted is not None:
                print(
                    f"[Inference] #{prediction_count:>4d} | "
                    f"window {oldest_frame:>4d}-{newest_frame:>4d} | "
                    f"EMIT={emitted} | raw={decoded['raw_label']} ({decoded['raw_conf']:.2%})"
                )
            else:
                print(
                    f"[Inference] #{prediction_count:>4d} | "
                    f"window {oldest_frame:>4d}-{newest_frame:>4d} | "
                    f"raw={decoded['raw_label']} ({decoded['raw_conf']:.2%}) | "
                    f"gated={decoded['gated_label']} | voted={decoded['voted_label']}"
                )
        else:
            print(f"[Inference] Buffering... frame {newest_frame:>4d} | "
                  f"buf {len(buffer):>3d}/{MIN_BUFFER_FOR_PREDICTION} needed")

    print(f"[Inference] Process shutting down. "
          f"Total predictions: {prediction_count} | "
          f"Total frames received: {frames_received}")
    write_sentence_output("process_stopped")


# ──────────────────────── main loop ────────────────────────────────
def main():
    """Double-threaded, fixed-30-FPS capture loop with inference process (fresh webcam frames)."""
    print("=" * 60)
    print("  Fixed 30 FPS  –  Leap Motion + Webcam  →  Inference")
    print("  (double-threaded bufferless webcam)")
    print("=" * 60)
    print()

    duration = DEFAULT_DURATION
    fps = TARGET_FPS
    total_frames = duration * fps

    if not str(STGCN_CHECKPOINT_PATH).strip():
        print("Error: STGCN_CHECKPOINT_PATH is empty.")
        print("Set STGCN_CHECKPOINT_PATH at the top of this file, then rerun.")
        return
    if not os.path.isfile(STGCN_CHECKPOINT_PATH):
        print(f"Error: ST-GCN checkpoint not found at: {STGCN_CHECKPOINT_PATH}")
        return

    print("Initializing Leap Motion...")
    listener = LatestFrameListener()
    connection = leap.Connection()
    connection.add_listener(listener)

    print("Initializing camera...")
    raw_cap = find_working_camera()
    if raw_cap is None or not raw_cap.isOpened():
        print("Error: Could not open any camera")
        print("  1. Check that the camera is connected and not used by another app")
        print("  2. Check camera permissions")
        connection.remove_listener(listener)
        return

    print("Warming up camera...")
    for _ in range(15):
        raw_cap.read()

    cap = BufferlessVideoCapture(raw_cap)

    frame_queue = mp.Queue(maxsize=60)
    stop_event = mp.Event()
    ready_event = mp.Event()
    inf_proc = mp.Process(
        target=inference_process,
        args=(frame_queue, BUFFER_SIZE, stop_event, ready_event),
        daemon=True,
    )
    inf_proc.start()

    print("Waiting for inference model to load...")
    ready_event.wait(timeout=30.0)
    if not ready_event.is_set():
        print("Error: Inference process did not become ready in time.")
        inf_proc.terminate()
        cap.release()
        connection.remove_listener(listener)
        return
    print("Inference model ready!\n")

    try:
        with connection.open():
            connection.set_tracking_mode(leap.TrackingMode.Desktop)
            print("Leap connection established. Waiting for device...")
            time.sleep(1.0)

            print(f"\nRecording: {duration}s  @  {fps} FPS  ({total_frames} frames)")
            print(f"  Inference buffer : {BUFFER_SIZE} frames\n")

            for c in [3, 2, 1]:
                print(f"  Starting in {c}...")
                time.sleep(1.0)
            print("  GO!\n")

            recording_start = time.perf_counter()
            frame_number = 0
            missed_cam_frames = 0
            dropped_queue_frames = 0
            last_printed_second = -1

            while frame_number < total_frames:
                tick_start = time.perf_counter()

                elapsed_since_start = tick_start - recording_start
                current_second = int(elapsed_since_start)
                if current_second != last_printed_second:
                    remaining = duration - current_second
                    print(f"  [t]  {current_second}s / {duration}s elapsed  "
                          f"({remaining}s remaining)", flush=True)
                    last_printed_second = current_second

                # 1) Read one fresh webcam frame (blocks until a new frame arrives if empty)
                ret, cam_frame = cap.read()

                # 2) Grab latest Leap tracking data
                tracking_event = listener.get_latest()
                hand_data = extract_hand_row(tracking_event)
                hand_data["frame_number"] = frame_number
                hand_data["system_time"] = tick_start

                # 3) Send to inference process
                if ret and cam_frame is not None:
                    try:
                        frame_queue.put_nowait((frame_number, hand_data, cam_frame))
                    except Exception:
                        dropped_queue_frames += 1
                else:
                    missed_cam_frames += 1

                frame_number += 1

                # 4) Sleep the remainder of the tick to hold 30 FPS
                elapsed = time.perf_counter() - tick_start
                sleep_time = FRAME_INTERVAL - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            total_elapsed = time.perf_counter() - recording_start
            actual_fps = frame_number / total_elapsed if total_elapsed > 0 else 0

            print("\nCapture complete!")
            print(f"  Frames captured      : {frame_number}")
            print(f"  Missed cam frames    : {missed_cam_frames}")
            print(f"  Dropped queue frames : {dropped_queue_frames}")
            print(f"  Sent to inference    : {frame_number - missed_cam_frames - dropped_queue_frames}")
            print(f"  Actual duration      : {total_elapsed:.3f}s")
            print(f"  Effective FPS        : {actual_fps:.2f}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        stop_event.set()
        try:
            frame_queue.put(None, timeout=2.0)
        except Exception:
            pass
        inf_proc.join(timeout=5.0)
        if inf_proc.is_alive():
            inf_proc.terminate()

        cap.release()
        cv2.destroyAllWindows()
        connection.remove_listener(listener)


if __name__ == "__main__":
    mp.freeze_support()                     # needed on Windows
    main()

