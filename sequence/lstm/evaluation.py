"""
Evaluation: frame-level metrics and streaming WER evaluation harness.

Provides:
  - evaluate_lstm_full : frame-level accuracy + classification report
  - evaluate_lstm_wer  : streaming WER evaluation over a catalog of recordings
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from config import (
    BAG_AGGREGATION,
    BAG_SIZE,
    BACKGROUND_LABEL,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    LEAP_FPS,
    MIN_SIGN_FRAMES,
    MIN_SIGN_MS,
    SIGN_BG_MARGIN,
    STREAM_MODE,
)
from decoder import stream_lstm_online
from utils import compute_wer


# ──────────────────────────────────────────────────────────────────────
# Frame-level evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_lstm_full(
    model_obj: nn.Module,
    loader: DataLoader,
    num_classes: int,
    split_name: str = "",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Frame-level accuracy evaluation."""
    model_obj.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for sequences, labels, lengths in loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)

            logits = model_obj(sequences, lengths)
            preds = torch.argmax(logits, dim=2)

            for i in range(sequences.size(0)):
                seq_len = int(lengths[i].item())
                all_preds.extend(preds[i, :seq_len].cpu().numpy().tolist())
                all_targets.extend(labels[i, :seq_len].cpu().numpy().tolist())

    if len(all_targets) == 0:
        return 0.0, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    acc = accuracy_score(all_targets, all_preds)
    return (
        float(acc),
        np.asarray(all_preds, dtype=np.int64),
        np.asarray(all_targets, dtype=np.int64),
    )


def print_frame_level_report(
    model_obj: nn.Module,
    loaders: dict[str, DataLoader],
    id_to_label: dict[int, str],
    num_classes: int,
) -> pd.DataFrame:
    """Run frame-level evaluation on multiple splits and print reports."""
    summary_rows = []
    for split_name, loader in loaders.items():
        acc, preds, targets = evaluate_lstm_full(
            model_obj, loader, num_classes, split_name,
        )
        print(f"\n{'='*10} {split_name.upper()} FRAME-LEVEL REPORT {'='*10}")
        print(f"Accuracy: {acc:.4f}")
        target_names = [id_to_label[i] for i in range(num_classes)]
        print(
            classification_report(
                targets,
                preds,
                labels=list(range(num_classes)),
                target_names=target_names,
                zero_division=0,
            )
        )
        summary_rows.append({"split": split_name, "frame_accuracy": acc})

    return pd.DataFrame(summary_rows)


# ──────────────────────────────────────────────────────────────────────
# WER evaluation
# ──────────────────────────────────────────────────────────────────────

def _print_wer_summary(
    df: pd.DataFrame,
    split_name: str,
    normalization_name: str,
    stream_mode: str,
    print_examples: int,
) -> None:
    header = f"{'='*10} {split_name.upper()} WER SUMMARY {'='*10}"
    print(f"\n{header}")
    print(f"Normalization  : {normalization_name}")
    print(f"Stream mode    : {stream_mode}")
    print(f"Bag size       : {BAG_SIZE}  ({BAG_SIZE/LEAP_FPS*1000:.0f}ms)")
    print(f"Aggregation    : {BAG_AGGREGATION}")
    print(f"Conf threshold : {CONFIDENCE_THRESHOLD}")
    print(f"Min sign frames: {MIN_SIGN_FRAMES}  ({MIN_SIGN_MS}ms)")
    print(f"Total sequences evaluated: {len(df)}")

    if df.empty:
        return

    print(f"Mean WER:   {df['wer'].mean():.4f}")
    print(f"Median WER: {df['wer'].median():.4f}")
    print(f"Std WER:    {df['wer'].std(ddof=0):.4f}")

    n_show = min(print_examples, len(df))
    print(f"\nShowing {n_show} example sequence(s):")
    for _, row in df.head(n_show).iterrows():
        print(f"  [{int(row['sample_idx'])}] {row['user']} | {row['recording_id']}")
        print(
            f"    Frames       : {int(row['num_frames'])} | "
            f"Missing ratio: {float(row['missing_ratio']):.3f}"
        )
        print(
            f"    Stream steps : {int(row['num_stream_predictions'])} | "
            f"Emitted: {int(row['emitted_count'])}"
        )
        print(f"    GT           : {row['ground_truth'].split()}")
        print(f"    Prediction   : {row['prediction'].split()}")
        print(f"    WER          : {float(row['wer']):.4f}")
        if row.get("emit_regions"):
            print(f"    Emit regions : {row['emit_regions']}")
        if row.get("gt_segments"):
            print(f"    GT segments  : {row['gt_segments']}")


def evaluate_lstm_wer(
    samples: list[dict],
    split_name: str,
    model_obj: nn.Module,
    normalize_fn: Callable[[np.ndarray], np.ndarray],
    id_to_label: dict[int, str],
    normalization_name: str,
    print_examples: int         = 2,
    bag_size: int               = BAG_SIZE,
    aggregation: str            = BAG_AGGREGATION,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    sign_bg_margin: float       = SIGN_BG_MARGIN,
    min_sign_frames: int        = MIN_SIGN_FRAMES,
) -> pd.DataFrame:
    """Evaluate streaming WER for an entire LSTM sequence set.

    Returns
    -------
    pd.DataFrame — one row per sample with all WER fields.
    """
    rows = []

    for idx, sample in enumerate(samples, start=1):
        stream_steps, final_preds, emit_regions = stream_lstm_online(
            V=sample["V"],
            model_obj=model_obj,
            normalize_fn=normalize_fn,
            id_to_label=id_to_label,
            background_label=BACKGROUND_LABEL,
            bag_size=bag_size,
            aggregation=aggregation,
            confidence_threshold=confidence_threshold,
            sign_bg_margin=sign_bg_margin,
            min_sign_frames=min_sign_frames,
        )

        wer         = compute_wer(final_preds, sample["ground_truth"])
        gt_segments = sample.get("segmentation_regions", [])

        rows.append({
            "sample_idx":             idx,
            "split":                  split_name,
            "user":                   sample["user"],
            "recording_id":           sample["recording_id"],
            "num_frames":             sample["num_frames"],
            "missing_ratio":          sample["missing_ratio"],
            "gt_len":                 len(sample["ground_truth"]),
            "pred_len":               len(final_preds),
            "raw_len":                len(stream_steps),
            "wer":                    wer,
            "stream_mode":            STREAM_MODE,
            "stream_delay_frames":    0,
            "num_stream_predictions": len(stream_steps),
            "first_prediction_frame": (
                int(stream_steps[0]["frame_index"]) if stream_steps else None
            ),
            "emitted_count":          len(final_preds),
            "ground_truth":           " ".join(sample["ground_truth"]),
            "prediction":             " ".join(final_preds),
            "stream_steps":           stream_steps,
            "emit_regions":           emit_regions,
            "gt_segments":            gt_segments,
        })

    df = pd.DataFrame(rows)
    _print_wer_summary(
        df, split_name, normalization_name, STREAM_MODE, print_examples,
    )
    return df
