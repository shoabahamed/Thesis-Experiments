"""
Utility functions: class weights, WER computation, model save/load, plotting.

Provides:
  - compute_class_weights : inverse-frequency class weighting
  - compute_wer           : word error rate via edit distance
  - remove_consecutive_duplicates / remove_background : label cleaning
  - save_unique_model     : save model checkpoint with metadata
  - plot_training_curves  : matplotlib loss/accuracy plots
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import (
    BACKGROUND_LABEL,
    DEVICE,
    MODEL_NAME,
    NORMALIZATION_NAME,
    PROJECT_ROOT,
)


# ──────────────────────────────────────────────────────────────────────
# Class weights
# ──────────────────────────────────────────────────────────────────────

def compute_class_weights(
    train_dataset,
    num_classes: int,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from training dataset.

    Returns a (num_classes,) tensor on DEVICE.
    """
    frame_counts = Counter()
    for sample in train_dataset.samples:
        frame_counts.update(sample["labels"].tolist())

    epsilon = 1e-6
    raw_weights = np.array(
        [1.0 / (frame_counts.get(c, 0) + epsilon) for c in range(num_classes)],
        dtype=np.float32,
    )
    class_weights = raw_weights * (num_classes / raw_weights.sum())
    return torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Supports per-class weights (alpha) and ignore_index for padding.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int = -1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target != self.ignore_index
        if not mask.any():
            return torch.tensor(0.0, device=input.device, requires_grad=True)

        input_m = input[mask]
        target_m = target[mask]

        log_p = torch.nn.functional.log_softmax(input_m, dim=-1)
        p = torch.exp(log_p)

        log_p_true = log_p.gather(1, target_m.unsqueeze(1)).squeeze(1)
        p_true = p.gather(1, target_m.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - p_true) ** self.gamma
        loss = -focal_weight * log_p_true

        if self.weight is not None:
            alpha = self.weight[target_m]
            loss = loss * alpha

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ──────────────────────────────────────────────────────────────────────
# WER helpers
# ──────────────────────────────────────────────────────────────────────

def remove_consecutive_duplicates(labels: list[str]) -> list[str]:
    """Collapse consecutive duplicate labels while preserving order."""
    if not labels:
        return []
    out = [labels[0]]
    for lbl in labels[1:]:
        if lbl != out[-1]:
            out.append(lbl)
    return out


def remove_background(
    labels: list[str],
    background_label: str = BACKGROUND_LABEL,
) -> list[str]:
    """Remove background labels from a label sequence."""
    return [lbl for lbl in labels if lbl != background_label]


def compute_wer(pred: list[str], gt: list[str]) -> float:
    """Compute Word Error Rate using Levenshtein edit distance."""
    n, m = len(gt), len(pred)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost     = 0 if gt[i - 1] == pred[j - 1] else 1
            dp[i, j] = min(
                dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost,
            )

    return float(dp[n, m] / n)


# ──────────────────────────────────────────────────────────────────────
# Model save / load
# ──────────────────────────────────────────────────────────────────────

def save_unique_model(
    model_obj: nn.Module,
    best_val_acc: float | None = None,
    save_dir: str = "trained_models",
    model_name: str = MODEL_NAME,
    info: dict | None = None,
) -> str:
    """Save model checkpoint with unique filename and metadata JSON."""
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_id = uuid.uuid4().hex[:8]

    val_str = f"{best_val_acc:.4f}" if best_val_acc is not None else "nan"
    filename = f"{ts}_{model_name}_val-{val_str}_{short_id}.pt"
    path = os.path.join(save_dir, filename)

    metadata = {
        "model_name": model_name,
        "saved_at_utc": ts,
        "uid": short_id,
        "best_val_acc": best_val_acc,
        "normalization": NORMALIZATION_NAME,
    }

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        metadata["git_sha"] = git_sha
    except Exception:
        pass

    if info:
        metadata.update(info)

    payload = {"model_state_dict": model_obj.state_dict(), "metadata": metadata}
    torch.save(payload, path)

    try:
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        pass

    print(f"Saved model checkpoint: {path}")
    return path


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_training_curves(train_result: dict, save_path: str | None = None) -> None:
    """Plot training loss, accuracy, and macro-F1 curves."""
    history_train_loss = train_result["history_train_loss"]
    history_train_batch_acc = train_result["history_train_batch_acc"]
    history_val_acc = train_result["history_val_acc"]
    history_train_f1 = train_result.get("history_train_f1", [])
    history_val_f1 = train_result.get("history_val_f1", [])

    epochs_axis = np.arange(1, len(history_train_loss) + 1)

    plt.figure(figsize=(18, 4))

    # ── Loss ──
    plt.subplot(1, 3, 1)
    plt.plot(epochs_axis, history_train_loss, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    # ── Accuracy ──
    plt.subplot(1, 3, 2)
    if len(history_train_batch_acc) == len(epochs_axis):
        plt.plot(
            epochs_axis, history_train_batch_acc,
            marker="x", linestyle="--", label="Train (batch)",
        )
    plt.plot(epochs_axis, history_val_acc, marker="o", label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ── Macro F1 ──
    plt.subplot(1, 3, 3)
    if len(history_train_f1) == len(epochs_axis):
        plt.plot(
            epochs_axis, history_train_f1,
            marker="x", linestyle="--", label="Train (batch)",
        )
    if len(history_val_f1) == len(epochs_axis):
        plt.plot(epochs_axis, history_val_f1, marker="o", label="Validation")
    plt.title("Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Console Logging redirection (Tee)
# ──────────────────────────────────────────────────────────────────────

class TeeLogger:
    """Redirects sys.stdout and sys.stderr to both a file and standard output."""

    def __init__(self, filepath: str):
        log_dir = os.path.dirname(filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, data):
        self.stdout.write(data)
        # Skip writing tqdm progress bars (which use \r) to the log file
        if "\r" in data:
            return
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __getattr__(self, attr):
        return getattr(self.stdout, attr)

    def close(self):
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr
        self.file.close()

