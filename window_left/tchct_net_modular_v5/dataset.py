"""
PyTorch dataset and collation for full-sequence THCT-Net training.

Provides:
    - FullSequenceDataset     : one complete recording per item with per-frame labels
  - collate_full_sequences  : custom collate for variable-length sequences
"""
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    GLOSS_BALANCED_GLOSSES_PER_BATCH,
    GLOSS_BALANCED_SAMPLES_PER_GLOSS,
    SEED,
)



class FullSequenceDataset(Dataset):
    """One complete recording per item with frame-level integer labels."""

    def __init__(
        self,
        segments: list[dict],
        label_to_id: dict[str, int],
        normalize_fn=None,
        background_id: int = 0,
        augment_pipeline=None,
    ):
        self.label_to_id = label_to_id
        self.normalize_fn = normalize_fn
        self.background_id = background_id
        self.augment_pipeline = augment_pipeline
        self.samples: list[dict] = []

        by_recording: dict[str, list[dict]] = defaultdict(list)
        for item in segments:
            rec_id = str(item.get("recording_id", "unknown"))
            by_recording[rec_id].append(item)

        for rec_id, items in sorted(by_recording.items()):
            video = items[0].get("recording_features", None)
            if video is None:
                continue

            video = np.asarray(video, dtype=np.float32)
            t_len = int(video.shape[0])
            labels = np.full(t_len, background_id, dtype=np.int64)

            sign_items = [
                it for it in items
                if not it.get("is_background", False)
            ]
            sign_items.sort(key=lambda it: it["segment_span"][0])

            for seg in sign_items:
                start, end = seg["segment_span"]
                sign_id = label_to_id[seg["label"]]
                labels[start : end + 1] = sign_id

            ground_truth = [seg["label"] for seg in sign_items]

            self.samples.append({
                "recording_id": rec_id,
                "user": items[0].get("user", "unknown"),
                "video": video,
                "labels": labels,
                "length": t_len,
                "ground_truth": ground_truth,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video = sample["video"]
        if self.normalize_fn is not None:
            video = self.normalize_fn(video)
        if self.augment_pipeline is not None:
            video = self.augment_pipeline(video)
        video_t = torch.tensor(video, dtype=torch.float32)
        labels_t = torch.tensor(sample["labels"], dtype=torch.long)
        length_t = torch.tensor(sample["length"], dtype=torch.long)
        return video_t, labels_t, length_t


def collate_full_sequences(batch):
    """Pad variable-length sequences for batching."""
    videos, labels, lengths = zip(*batch)
    lengths = torch.stack(lengths)
    max_len = int(lengths.max().item())

    bsz = len(batch)
    feat_dim = videos[0].shape[1]
    padded_videos = torch.zeros(bsz, max_len, feat_dim, dtype=torch.float32)
    padded_labels = torch.full((bsz, max_len), -1, dtype=torch.long)

    for i, (video, label, length) in enumerate(zip(videos, labels, lengths)):
        seq_len = int(length.item())
        padded_videos[i, :seq_len] = video[:seq_len]
        padded_labels[i, :seq_len] = label[:seq_len]

    return padded_videos, padded_labels, lengths


# ──────────────────────────────────────────────────────────────────────
# Window-based Dataset and Samplers for Non-Causal THCT-Net
# ──────────────────────────────────────────────────────────────────────

def extract_left_window_from_video(video: np.ndarray, start: int, window_size: int) -> np.ndarray | None:
    """Extract a fixed-size left-aligned window with valid frames only (no padding)."""
    end = start + window_size
    if start < 0 or end > len(video):
        return None
    return video[start:end].astype(np.float32, copy=False)


def generate_windows_from_interval(item: dict, window_size: int = 30, stride: int = 1):
    """Generate left-aligned windows fully contained in [tb, te] (no padding)."""
    windows, labels = [], []

    label = item["label"]
    tb, te = item["segment_span"]
    video = item.get("recording_features", None)

    if video is None or len(video) == 0:
        # Fallback to local segment-only context if full recording is unavailable.
        video = item["segment"]
        tb, te = 0, len(video) - 1

    video = np.asarray(video, dtype=np.float32)

    start_min = int(tb)
    start_max = int(te) - window_size + 1

    for start in range(start_min, start_max + 1, stride):
        win = extract_left_window_from_video(video, start, window_size)
        if win is None:
            continue
        windows.append(win)
        labels.append(label)

    return windows, labels


def build_windows_from_segments(
    segments: list[dict],
    label_to_id: dict[str, int],
    window_size: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a window dataset from segments."""
    X, y = [], []
    for item in segments:
        windows, labels = generate_windows_from_interval(item, window_size, stride)
        X.extend(windows)
        y.extend([label_to_id[lbl] for lbl in labels if lbl in label_to_id])

    if len(X) == 0:
        from config import INPUT_DIM
        return np.zeros((0, window_size, INPUT_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y


class LeapSignDataset(Dataset):
    """Dataset returning (sequence, label, length) tensors.

    Input arrays:
        X: (N, W, D) where D=138 raw features
        y: (N,)
    __getitem__ output:
        sequence: (W, D), label: scalar, length: scalar
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, normalize_fn=None, augment_pipeline=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.normalize_fn = normalize_fn
        self.augment_pipeline = augment_pipeline
        self.lengths = np.full((len(self.X),), self.X.shape[1] if len(self.X) > 0 else 0, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sequence_np = self.X[idx]
        if self.normalize_fn is not None:
            sequence_np = self.normalize_fn(sequence_np)
        if self.augment_pipeline is not None:
            sequence_np = self.augment_pipeline(sequence_np)
        sequence = torch.tensor(sequence_np, dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        return sequence, label, length


def collate_batch(batch):
    """Stack a list of samples into one batch."""
    sequences, labels, lengths = zip(*batch)
    return torch.stack(sequences), torch.stack(labels), torch.stack(lengths)


class GlossBalancedBatchSampler:
    """Paper-style sampler: each batch contains M glosses and K samples per gloss."""

    def __init__(
        self,
        labels: np.ndarray,
        glosses_per_batch: int = GLOSS_BALANCED_GLOSSES_PER_BATCH,
        samples_per_gloss: int = GLOSS_BALANCED_SAMPLES_PER_GLOSS,
        seed: int = SEED,
    ):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.glosses_per_batch = int(glosses_per_batch)
        self.samples_per_gloss = int(samples_per_gloss)
        self.seed = int(seed)
        self._epoch = 0

        if len(self.labels) == 0:
            raise ValueError("Cannot build GlossBalancedBatchSampler with empty labels.")
        if self.glosses_per_batch <= 0 or self.samples_per_gloss <= 0:
            raise ValueError("glosses_per_batch and samples_per_gloss must be positive integers.")

        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, cls in enumerate(self.labels.tolist()):
            self.class_to_indices[int(cls)].append(idx)

        self.classes = sorted(self.class_to_indices.keys())
        if len(self.classes) == 0:
            raise ValueError("No classes available for batch sampling.")

        self.batch_size = self.glosses_per_batch * self.samples_per_gloss
        self.num_batches = max(1, int(np.ceil(len(self.labels) / max(1, self.batch_size))))

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)

        for _ in range(self.num_batches):
            if len(self.classes) >= self.glosses_per_batch:
                selected_classes = rng.sample(self.classes, self.glosses_per_batch)
            else:
                selected_classes = [rng.choice(self.classes) for _ in range(self.glosses_per_batch)]

            batch_indices = []
            for cls in selected_classes:
                cls_indices = self.class_to_indices[cls]
                if len(cls_indices) >= self.samples_per_gloss:
                    chosen = rng.sample(cls_indices, self.samples_per_gloss)
                else:
                    chosen = [rng.choice(cls_indices) for _ in range(self.samples_per_gloss)]
                batch_indices.extend(chosen)

            rng.shuffle(batch_indices)
            yield batch_indices

