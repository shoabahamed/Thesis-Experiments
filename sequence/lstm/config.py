"""
Central configuration for the LSTM sign-language recognition pipeline.

All hyperparameters, paths, device setup, feature definitions, and
reproducibility seeding live here. Every other module imports from config.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # sequence/lstm -> sequence -> Experiments
DATASET_ROOT = PROJECT_ROOT / "dataset"

# ──────────────────────────────────────────────────────────────────────
# Label / split constants
# ──────────────────────────────────────────────────────────────────────
BACKGROUND_LABEL = "background"

# All users available in the dataset (used by leave-one-out).
ALL_USERS = ["user1", "user2", "user3", "user5"]

# Legacy single-split defaults (override via CLI / main).
DEFAULT_DEV_USERS = ["user1", "user2", "user5"]
DEFAULT_TEST_USER = "user3"
DEV_VAL_RATIO = 0.12
DEV_VAL_SEED = SEED + 202

# ──────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────────────────────────────
BATCH_SIZE = 4
MODEL_NAME = "lstm_fullseq"
NORMALIZATION_NAME = "palm_ref"
EPOCHS = 50
LEARNING_RATE = 3e-4

# ──────────────────────────────────────────────────────────────────────
# LSTM architecture
# ──────────────────────────────────────────────────────────────────────
FEAT_DIM = 128
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2

# ──────────────────────────────────────────────────────────────────────
# Streaming / online decoder
# ──────────────────────────────────────────────────────────────────────
STREAM_MODE = "lstm_online"
WER_EXAMPLE_PRINT_COUNT = 5

LEAP_FPS = 30
MIN_SIGN_MS = 500
MIN_SIGN_FRAMES = max(1, int(round((MIN_SIGN_MS / 1000.0) * LEAP_FPS)))

BAG_SIZE = 5
BAG_AGGREGATION = "mean"
CONFIDENCE_THRESHOLD = 0.35
SIGN_BG_MARGIN = 0.10

# ──────────────────────────────────────────────────────────────────────
# Feature definition (132 raw Leap features)
# ──────────────────────────────────────────────────────────────────────
HANDS = ["left", "right"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
BONES = ["metacarpal", "proximal", "intermediate", "distal"]
CARTESIAN_AXES = ["x", "y", "z"]
START_AXES = ["sx", "sy", "sz"]

FEATURE_KEYS: list[str] = []

# Palm and wrist coordinates per hand.
for _hand in HANDS:
    for _part in ["palm", "wrist"]:
        for _axis in CARTESIAN_AXES:
            FEATURE_KEYS.append(f"{_hand}_{_part}_{_axis}")

# Finger bone start-joint coordinates per hand.
for _hand in HANDS:
    for _finger in FINGERS:
        for _bone in BONES:
            for _axis in START_AXES:
                FEATURE_KEYS.append(f"{_hand}_{_finger}_{_bone}_{_axis}")

assert len(FEATURE_KEYS) == 132, f"Expected 132 features, got {len(FEATURE_KEYS)}"

INPUT_DIM = len(FEATURE_KEYS)

FEATURE_INDEX = {key: idx for idx, key in enumerate(FEATURE_KEYS)}

PALM_TRIPLETS: dict[str, tuple[int, int, int]] = {}
HAND_POSITION_TRIPLETS: dict[str, list[tuple[int, int, int]]] = {}

for _hand in HANDS:
    PALM_TRIPLETS[_hand] = tuple(
        FEATURE_INDEX[f"{_hand}_palm_{axis}"] for axis in CARTESIAN_AXES
    )

    _triplets: list[tuple[int, int, int]] = []
    _triplets.append(
        tuple(FEATURE_INDEX[f"{_hand}_wrist_{axis}"] for axis in CARTESIAN_AXES)
    )
    for _finger in FINGERS:
        for _bone in BONES:
            _triplets.append(
                tuple(FEATURE_INDEX[f"{_hand}_{_finger}_{_bone}_{axis}"] for axis in START_AXES)
            )
    HAND_POSITION_TRIPLETS[_hand] = _triplets
