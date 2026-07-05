# `tchct_net_modular_v3` — Repository Context Document

Use this as a system/context prompt for another AI working on this codebase.

---

## 1. High-Level Purpose

This is a **modular PyTorch pipeline for continuous sign-language recognition** from **Leap Motion hand-skeleton data**. It implements **THCT-Net** (Two-stream Hybrid CNN-Transformer Network), adapted for a custom multi-user dataset.

The system has two evaluation modes:

1. **Window-level classification** — train/evaluate on fixed 30-frame windows labeled by gloss (sign class).
2. **Streaming / online recognition** — run a sliding-window model + bag-aggregated hysteresis decoder over full recordings, then compute **WER** (Word Error Rate) and **SHREC'21 streaming metrics** (Detection Rate, FPR, Jaccard).

**v3-specific change:** All `BatchNorm` layers were replaced with `GroupNorm` so inference behaves identically at `batch_size=1` (streaming) and larger training batches. README notes this improved training stability and results, especially with missing-frame augmentation.

---

## 2. Project Location & Dependencies

```
c:\Shoab\Thesis\Experiments\window_left\tchct_net_modular_v3\
```

- Part of a thesis experiment suite under `Experiments/window_left/`
- Sibling folders: `tchct_net_modular`, `v2`, `v3_part2`, `v4`, `v5` (iterations of the same architecture)
- **Dataset root (default):** `PROJECT_ROOT / "dataset"` → `c:\Shoab\Thesis\Experiments\dataset` (via `config.py`: `parents[2]` from module file)
- **Entry point:** `main.py`
- **Stack:** Python, PyTorch, NumPy, Pandas, scikit-learn, matplotlib, tqdm

---

## 3. Dataset Format & Users

### Directory layout (expected)

```
dataset/
  user1/
    leap_data/       ← CSV files (Leap Motion skeleton per frame)
    segmentation/    ← TXT files (frame-level gloss annotations)
  user2/
  user3/
  user5/
```

### File pairing

- Recording IDs follow pattern `P\d+_S\d+_R\d+` (e.g. `P1_S1_R1`)
- Each CSV is matched to a TXT with the same recording ID

### Segmentation TXT format

Whitespace-separated, no header:

```
start_frame  end_frame  gloss_label
```

### Users

- **Available users:** `user1`, `user2`, `user3`, `user5` (no `user4`)
- **Default single-split:** train on `user1, user2, user5`, test on `user3`
- **LOUO mode:** each user becomes test user once; all others are dev users

---

## 4. Input Features (132-D per frame)

Defined in `config.py` → `FEATURE_KEYS` (132 total):

| Component | Count |
|-----------|-------|
| Left/right palm xyz | 6 |
| Left/right wrist xyz | 6 |
| Per hand: 5 fingers × 4 bones × 3 start-joint coords (sx,sy,sz) | 120 |

**Normalization:** `palm_reference_normalize_sequence` (`features.py`)

- For each hand: subtract that hand's palm position from all joint coordinates
- Palm channels set to 0 (origin anchor)
- Applied per-window at train time and per-window during streaming inference

---

## 5. Module Map

| File | Role |
|------|------|
| `config.py` | All hyperparameters, paths, seeds, feature schema, decoder params |
| `main.py` | CLI orchestrator: load data → split → train → evaluate → save |
| `data_loading.py` | Parse CSV/TXT, discover recordings, extract segments + background gaps |
| `data_splitting.py` | User-based splits, filtering, label encoding, WER catalogs |
| `dataset.py` | Window datasets, gloss-balanced sampler, window generation |
| `features.py` | Raw feature extraction + palm-reference normalization |
| `augmentations.py` | Training augmentations (rotation, scale, noise, frame dropout) |
| `model.py` | THCT-Net architecture (CNN + Transformer streams, GroupNorm) |
| `decoder.py` | Online streaming decoder (bag aggregator + hysteresis state machine) |
| `trainer.py` | Training loop, early stopping, LR scheduling |
| `evaluation.py` | Window accuracy, WER eval, SHREC metrics, result persistence |
| `metrics_original.py` | Buggy Duo Streamers baseline metrics (for comparison) |
| `metrics_corrected.py` | Correct SHREC'21 protocol implementation |
| `utils.py` | WER, FocalLoss, checkpoint save/load, plots, TeeLogger |
| `benchmark.py` | Standalone latency/MACs benchmark (not part of main pipeline) |

---

## 6. End-to-End Pipeline Flow

```
load_all_segments(dataset_root)
    ↓
prepare_split(dev_users, test_user)     ← user-level holdout + recording-level val split
    ↓
build_windows_from_segments()           ← 30-frame left-aligned windows, stride=1
    ↓
LeapSignDataset + GlossBalancedBatchSampler
    ↓
THCTNet training (CrossEntropy or FocalLoss)
    ↓
Window-level accuracy report + confusion matrices
    ↓
stream_model_online() on full recordings  ← WER evaluation
    ↓
SHREC'21 streaming metrics (original + corrected)
    ↓
Save checkpoint + training curves + WER results (parquet + npz)
```

---

## 7. Data Splitting Logic (`data_splitting.py`)

### User split

- **Dev users** → train + val segments
- **Test user** → held out entirely for test

### Validation split (within dev users)

- **Recording-level**, not random frame split
- Recordings grouped by sequence prefix (e.g. `P1_S1` from `P1_S1_R1`)
- **Exactly one repetition per sequence → validation**; rest → training
- Deterministic per-user seed: `DEV_VAL_SEED + sum(ord(c) for c in user)`

### Segment filtering (`filter_segments`)

Removes segments if:

- Length < 10 frames
- >40% frames have near-zero L2 norm (missing Leap data)
- Average hand confidence < 0.1

### Background handling

- Gaps between annotated signs are auto-labeled `"background"`
- Label encoding: background is always class 0; sign glosses sorted alphabetically

### Val/test label filtering

- Windows whose gloss was never seen in training are dropped

---

## 8. Window Generation (`dataset.py`)

Training uses **segment-contained windows**, not arbitrary sliding windows over full videos:

- For each annotated interval `[tb, te]`, generate all left-aligned windows of size 30 fully inside the interval
- Stride = 1
- Each window inherits the segment's gloss label
- No padding — windows that would extend outside the interval are skipped

**Effective batch size:** `GLOSS_BALANCED_GLOSSES_PER_BATCH × GLOSS_BALANCED_SAMPLES_PER_GLOSS` = 4 × 6 = **24 windows/batch**

`GlossBalancedBatchSampler`: each batch samples M unique glosses, K windows each — balances rare sign classes.

> **Note:** `FullSequenceDataset` exists for full-recording training but is **not used** by `main.py` in v3. The active path is window-based `LeapSignDataset`.

---

## 9. Model Architecture (`model.py` — THCTNet)

### Input

- `(B, T=30, D=132)` flat feature windows
- Reshaped to skeleton tensor: `(B, 3, T, 22, 2)` — 3 coords, 22 joints/hand, 2 hands

### Stream 1: CNNStream

- Two parallel branches:
  - **S branch:** raw coordinates
  - **M branch:** frame-to-frame temporal difference (motion)
- Each branch: 2D convs over `(time, joints)` with GroupNorm
- Concatenate → `_ResidualFusion` (1×7 + 7×1 convs) → GAP → FC → `(B, num_classes)`

### Stream 2: TransformerStream

- 3D conv token embedding: patches of `(Tw=5, Vw=2, Mw=1)` → 6×11×2 = **132 tokens**
- Learnable positional encoding
- 4 × `_ISATABlock` (ISATA self-attention from THCT-Net paper, eq. 6)
  - Non-causal (full attention over all tokens)
  - Custom attention: `α * tanh(QK^T/√d) + A` with learnable α and per-head bias A
- Temporal aggregation conv3d → GAP → linear head

### Fusion

- Learnable sigmoid weight `w`: `output = w * logits_cnn + (1-w) * logits_trans`
- Starts at w=0.5

### Key design choice (v3)

**GroupNorm everywhere instead of BatchNorm** — critical for streaming inference at B=1 without running-stat mismatch.

---

## 10. Training (`trainer.py`)

| Setting | Default |
|---------|---------|
| Optimizer | Adam, lr=3e-4 |
| Scheduler | ReduceLROnPlateau on val macro-F1 |
| Early stopping | patience=15 on val macro-F1 |
| Grad clip | norm=1.0 |
| Epochs | 7 |
| Model selection | Best val **macro-F1** (not accuracy — background dominates) |
| Loss | CrossEntropyLoss (optional FocalLoss with class weights) |

---

## 11. Streaming Decoder (`decoder.py`) — Critical Path

This is the **online inference engine**. Changes here directly affect WER.

### `stream_model_online(V, model, normalize_fn, ...)`

1. Maintain a deque of the last 30 frames
2. Every frame (stride=1): when window is full, normalize window → model forward → get logits
3. Feed logits to `SimplifiedBagDecoder`

### `_BagAggregator`

- Causal sliding bag of size 5 over raw logits
- Averages softmax probs (logit-space averaging ≈ product-of-experts)
- Returns `None` until bag is full (first 4 frames skipped)

### `SimplifiedBagDecoder` — hysteresis state machine

**States:** `SEEKING` ↔ `IN_SIGN`

**Gating rule** (must pass all):

- Predicted label ≠ background
- Confidence ≥ 0.35 (`CONFIDENCE_THRESHOLD`)
- Confidence − background_conf ≥ 0.10 (`SIGN_BG_MARGIN`)

**Emission:** at trailing edge (transition back to background)

- Emit majority-voted label from the region
- Discard regions shorter than 15 frames (~500ms at 30 FPS)

**Output per frame:** raw logits, post-bag probs, state, emitted label, emit region `(start, end, label)`

> **Important inconsistency:** `decoder.py` header comments say "causal" and "strictly causal", but the **model itself is non-causal within each 30-frame window** (full self-attention). Streaming is "online" in the sense that each window only uses past+current frames up to the window end, processed independently.

---

## 12. Evaluation (`evaluation.py`)

### Window-level

- Per-split accuracy + sklearn classification report
- Confusion matrices saved for Val/Test

### WER (primary sequence metric)

- Runs decoder on full recordings from WER catalog
- GT: deduplicated consecutive sign labels from segmentation
- Pred: emitted sign labels from decoder
- WER = Levenshtein distance / GT length

### SHREC'21 streaming metrics

Two implementations run in parallel:

- `metrics_original.py` — buggy Duo Streamers baseline (for reproducing old numbers)
- `metrics_corrected.py` — faithful to official SHREC'21 MATLAB script

Metrics: Detection Rate, FPR, Misclassification, Jaccard Index

### Result persistence (`save_split_results`)

Per split:

- `{slug}_metadata.parquet` — scalar WER fields per recording
- `{slug}_arrays.npz` — per-frame logits, probs, labels keyed by recording

---

## 13. Key Hyperparameters (`config.py`)

```python
SEED = 42
DEVICE = cuda if available else cpu

# Data
WINDOW_SIZE = 30          # frames (~1s at 30 FPS)
STRIDE = 1
BATCH_SIZE = 4            # val/test loader only; train uses gloss-balanced sampler
DEV_VAL_RATIO = 0.12      # (used in split logic via one-rep-per-sequence rule)

# Model
D_MODEL = 128
NUM_HEADS = 4
NUM_TRANSFORMER_LAYERS = 4
BASE_CH = 64
DROPOUT = 0.1

# Decoder
BAG_SIZE = 5
BAG_AGGREGATION = "mean"
CONFIDENCE_THRESHOLD = 0.35
SIGN_BG_MARGIN = 0.10
MIN_SIGN_MS = 500         → MIN_SIGN_FRAMES = 15

# Training
EPOCHS = 7
LEARNING_RATE = 3e-4
USE_AUGMENTATION = False  # enable via --augment CLI flag
```

---

## 14. CLI Usage (`main.py`)

```bash
# Default: single split, test user3
python main.py

# Leave-one-out over all users
python main.py --louo

# Custom dataset path
python main.py --dataset-root C:/Shoab/Thesis/Experiments/window_left/dataset

# Enable augmentation
python main.py --louo --augment --dropout-prob 0.2 --dropout-rate 0.05

# Skip training, load pre-trained checkpoints from config
python main.py --louo --from-checkpoint

# Quick smoke test (2 recordings/user, 2 epochs)
python main.py --test-mode

# Focal loss
python main.py --focal-loss --focal-gamma 2.0
```

### Outputs

```
trained_models/
  logs/run_{mode}_{timestamp}.log
  {timestamp}_thct_net_val-{acc}_{uid}.pt
  {timestamp}_thct_net_val-{acc}_{uid}.pt.json
  plots/curves_test_{user}_{timestamp}.png
  plots/confusion_matrices/{user}/
  results/{split}_metadata.parquet
  results/{split}_arrays.npz
```

---

## 15. Augmentations (`augmentations.py`)

Applied only during training (if `--augment`):

| Augmentation | Default | Notes |
|-------------|---------|-------|
| Frame dropout + linear interpolation | prob=0.2, rate=5% | Simulates missing Leap frames |
| Uniform scaling around palm | prob=0.5, range [0.95, 1.05] | |
| 3D rotation around palm | prob=0.5, ±8° | Per hand independently |
| Gaussian coordinate noise | prob=0.5, std=2mm | |

Validation via `verify_augmented_sample`: checks shape, NaN, bone length preservation, coordinate ranges. Falls back to original on failure.

---

## 16. Known Quirks & Gotchas

1. **`FullSequenceDataset` is unused** in the main pipeline — windows are the training unit.
2. **Model is non-causal within windows** but used in a causal sliding-window streaming setup.
3. **`CHECKPOINT_PATHS_BY_USER`** in config points to `tchct_net_modular_temp/` (older folder), not v3's own checkpoints.
4. **`decoder.py` comments** warn: "Any changes here will directly affect WER results."
5. **Background class dominates** window counts — macro-F1 is the model selection metric for a reason.
6. **Val split is one recording per sequence**, not a random 12% — `DEV_VAL_RATIO` is passed but the actual logic uses the one-rep-per-sequence rule in `split_dev_recordings`.
7. **Test windows filtered** to training-seen labels only; unseen glosses in test user's data are excluded from window eval but may still appear in WER if in GT (handled by label mapping gaps).

---

## 17. Typical Results (from saved checkpoints, LOUO run 2026-07-03)

| Test User | Val Acc | Val F1 | Test WER |
|-----------|---------|--------|----------|
| user1 | ~0.996 | ~0.993 | (see logs) |
| user2 | ~0.995 | ~0.992 | (see logs) |
| user3 | ~0.995 | ~0.991 | (see logs) |
| user5 | 0.997 | 0.993 | 0.052 |

Example checkpoint metadata (`user5` fold):

```json
{
  "test_user": "user5",
  "dev_users": ["user1", "user2", "user3"],
  "best_val_acc": 0.9966,
  "best_val_f1": 0.9934,
  "test_mean_wer": 0.0521
}
```

---

## 18. Data Structure Reference

### Segment dict (from `data_loading.py`)

```python
{
  "segment": np.ndarray,           # (T_seg, 132) — frames for this interval
  "label": str,                    # gloss or "background"
  "confidence": np.ndarray | None, # per-frame Leap confidence
  "segment_span": (start, end),    # frame indices in full recording
  "recording_features": np.ndarray,# (T_full, 132) — entire recording
  "is_background": bool,
  "recording_id": str,             # added by load_all_segments
  "user": str,                     # added by prepare_split
}
```

### WER catalog entry (recording-level)

```python
{
  "user": str,
  "recording_id": str,
  "V": np.ndarray,                 # (T, 132) full recording features
  "ground_truth": list[str],       # deduplicated sign gloss sequence
  "segmentation_regions": list,    # non-background frame intervals
  "missing_ratio": float,
  "num_frames": int,
}
```

### Model forward signature

```python
logits = model(sequences, lengths)
# sequences: (B, 30, 132)
# lengths: (B,) — accepted but NOT used by THCTNet
# logits: (B, num_classes)
```

---

## 19. Extension Points (where you'd typically modify things)

| Goal | Where to look |
|------|---------------|
| Change window size | `config.WINDOW_SIZE`, model init, decoder |
| New normalization | `features.py`, update `NORMALIZE_FN` in `main.py` |
| New augmentation | `augmentations.py`, CLI args in `main.py` |
| Different split strategy | `data_splitting.py` |
| Architecture changes | `model.py` |
| Decoder tuning | `config.py` thresholds + `decoder.py` |
| New metrics | `evaluation.py`, `metrics_corrected.py` |
| Full-sequence training | Wire up `FullSequenceDataset` in `main.py` |

---

## 20. One-Paragraph Summary

**`tchct_net_modular_v3`** is a thesis experiment implementing **THCT-Net** for **continuous sign language recognition** from **Leap Motion skeleton CSVs** with **frame-level gloss annotations**. It loads multi-user recordings, splits by user (LOUO or single holdout), generates **30-frame labeled windows** for training with a **gloss-balanced sampler**, trains a **dual CNN+Transformer model** (GroupNorm, palm-reference normalized 132-D features), evaluates **window accuracy**, then runs **streaming inference** via a **bag-aggregated hysteresis decoder** to compute **WER** and **SHREC'21 metrics**. The pipeline is fully CLI-driven through `main.py`, with modular files for each concern, and v3 specifically fixes BatchNorm→GroupNorm for stable batch-1 streaming inference.
