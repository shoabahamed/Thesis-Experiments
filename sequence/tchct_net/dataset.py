"""
PyTorch dataset and collation for full-sequence THCT-Net training.

Provides:
  - FullSequenceDataset     : one complete recording per item with frame-level labels
  - collate_full_sequences  : custom collate for variable-length sequences
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


class FullSequenceDataset(Dataset):
    """One complete recording per item with frame-level integer labels."""

    def __init__(
        self,
        segments: list[dict],
        label_to_id: dict[str, int],
        normalize_fn=None,
        background_id: int = 0,
    ):
        self.label_to_id = label_to_id
        self.normalize_fn = normalize_fn
        self.background_id = background_id
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
