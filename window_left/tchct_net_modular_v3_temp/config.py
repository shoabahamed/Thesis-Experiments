"""
Central configuration for the THCT-Net sign-language recognition pipeline.

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
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # sequence/tchct_net -> sequence -> Experiments
DATASET_ROOT = PROJECT_ROOT / "dataset"

# ──────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────
MODULE_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATHS_BY_USER = {
    "user1": "C:/Shoab/Thesis/Experiments/window_left/tchct_net_modular_temp/trained_models/20260630T231117Z_thct_net_val-0.9945_b419c550.pt",
    "user2": "C:/Shoab/Thesis/Experiments/window_left/tchct_net_modular_temp/trained_models/20260701T021606Z_thct_net_val-0.9938_78d33c3a.pt",
    "user3": "C:/Shoab/Thesis/Experiments/window_left/tchct_net_modular_temp/trained_models/20260701T052352Z_thct_net_val-0.9888_28a7a419.pt",
    "user5": "C:/Shoab/Thesis/Experiments/window_left/tchct_net_modular_temp/trained_models/20260701T083817Z_thct_net_val-0.9634_a66d6c82.pt",
}

# ──────────────────────────────────────────────────────────────────────
# Label / split constants
# ──────────────────────────────────────────────────────────────────────
BACKGROUND_LABEL = "background"

# All users available in the dataset (used by leave-one-out).
ALL_USERS = ["user1", "user2", "user3", "user5"]

# Default users for single-split mode (override via CLI / main).
DEFAULT_DEV_USERS = ["user1", "user2", "user5"]
DEFAULT_TEST_USER = "user3"
DEV_VAL_RATIO = 0.12
DEV_VAL_SEED = SEED + 202

# ──────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────────────────────────────
BATCH_SIZE = 4
GLOSS_BALANCED_GLOSSES_PER_BATCH = 4
GLOSS_BALANCED_SAMPLES_PER_GLOSS = 6
MODEL_NAME = "thct_net"
NORMALIZATION_NAME = "palm_ref"
EPOCHS = 7
LEARNING_RATE = 3e-4

# ──────────────────────────────────────────────────────────────────────
# Data Augmentation Configuration
# ──────────────────────────────────────────────────────────────────────
USE_AUGMENTATION = False
AUGMENT_ROTATION_PROB = 0.5
AUGMENT_ROTATION_RANGE = 8.0     # degrees (±8 deg)
AUGMENT_SCALING_PROB = 0.5
AUGMENT_SCALING_RANGE = (0.95, 1.05)
AUGMENT_NOISE_PROB = 0.5
AUGMENT_NOISE_STD = 2.0          # mm (since relative coordinates are in mm)
AUGMENT_DROPOUT_PROB = 0.2
AUGMENT_DROPOUT_RATE = 0.05


# ──────────────────────────────────────────────────────────────────────
# THCT-Net architecture
# ──────────────────────────────────────────────────────────────────────
D_MODEL = 128          # Transformer hidden dimension
NUM_HEADS = 4          # Attention heads
NUM_TRANSFORMER_LAYERS = 4   # Number of ISATA blocks
BASE_CH = 64           # CNN base channels
DROPOUT = 0.1          # Dropout rate (used in Transformer stream)

WINDOW_SIZE = 30
STRIDE = 1

# ──────────────────────────────────────────────────────────────────────
# Streaming / online decoder
# ──────────────────────────────────────────────────────────────────────
STREAM_MODE = "thct_net_batch"
WER_EXAMPLE_PRINT_COUNT = 5

LEAP_FPS = 30
MIN_SIGN_MS = 500
MIN_SIGN_FRAMES = max(1, int(round((MIN_SIGN_MS / 1000.0) * LEAP_FPS)))

BAG_SIZE = 5
BAG_AGGREGATION = "mean"
CONFIDENCE_THRESHOLD = 0.35
SIGN_BG_MARGIN = 0.10

ONLINE_WINDOW_SIZE = WINDOW_SIZE
ONLINE_STRIDE = 1

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
                tuple(FEATURE_INDEX[f"{_hand}_{_finger}_{_bone}_{_axis}"] for _axis in START_AXES)
            )
    HAND_POSITION_TRIPLETS[_hand] = _triplets

# ──────────────────────────────────────────────────────────────────────
# Disambiguation (post-logit, additive, read-only heuristics)
# ──────────────────────────────────────────────────────────────────────
# Master switch — must default True only after per-fold calibration
# and the flag-off regression check have both been validated.
USE_DISAMBIGUATION = False

# Hook A: motion energy (background-vs-sign rescue)
ENERGY_CONF_THRESH = 0.3     # ignore a hand's velocity if its confidence < this
ENERGY_GRAB_WEIGHT = 0.5     # weight (alpha) on |delta grab_strength|
ENERGY_PINCH_WEIGHT = 0.5    # weight (beta) on |delta pinch_strength|
BG_MARGIN_RESCUE_EPS = 0.03  # how close to SIGN_BG_MARGIN counts as "near-miss"
# theta_high is intentionally NOT a config constant — it's calibrated per
# LOUO fold (from dev_users only, never the held-out test user) and stored
# in that fold's templates.npz metadata (see build_templates.py).

# Hook B: turning-angle histogram (sign-vs-sign refinement)
HIST_N_BINS = 10            # bins over [0, pi] turning angle range
HIST_RESAMPLE_STEPS = 20    # fixed-length resampling before angle computation
HIST_MOTION_EPS = 5.0       # mm/frame-equivalent; below this, no direction defined
DISAMBIG_TAU_MARGIN = 0.15  # region-level top1-top2 margin below which Hook B fires
DISAMBIG_LAMBDA = 0.4       # blend weight: final = lam*cos_sim + (1-lam)*softmax
DISAMBIG_TOP_K = 3          # only re-rank the top-k candidate labels, not all classes

TEMPLATE_DIR = "trained_models/disambiguation"  # per-fold template banks live here
