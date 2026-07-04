"""
Evaluation: window-level classification metrics and streaming WER evaluation harness.

Provides:
    - evaluate_model_full          : window-level accuracy + classification report
  - evaluate_model_wer           : streaming WER evaluation over a catalog of recordings
  - evaluate_streaming_metrics   : SHREC'21 streaming metrics (DR, FPR, Jaccard)
                                   using both original (baseline) and corrected modules
"""
from __future__ import annotations

import json
import os
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
    ONLINE_WINDOW_SIZE,
    ONLINE_STRIDE,
)
from decoder import stream_model_online
from utils import compute_wer

# Streaming metrics — Duo Streamers baseline reference implementation
from metrics_original import (
    initialize_globals as init_orig,
    evaluate_detection_rate as eval_dr_orig,
    evaluate_model_with_fpr as eval_fpr_orig,
    evaluate_jaccard_index as eval_jac_orig,
    print_global_results as print_orig,
)

# Streaming metrics — corrected SHREC'21 protocol
from metrics_corrected import (
    initialize_globals as init_corr,
    evaluate_all as eval_all_corr,
    print_global_results as print_corr,
)


# ──────────────────────────────────────────────────────────────────────
# Window-level evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_model_full(
    model_obj: nn.Module,
    loader: DataLoader,
    num_classes: int,
    split_name: str = "",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Window-level accuracy evaluation."""
    model_obj.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for sequences, labels, lengths in loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)

            logits = model_obj(sequences, lengths)  # (B, num_classes)
            preds = torch.argmax(logits, dim=1)     # (B,)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    if len(all_targets) == 0:
        return 0.0, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Filter out ignore_index (-1) if any
    all_preds_np = np.asarray(all_preds, dtype=np.int64)
    all_targets_np = np.asarray(all_targets, dtype=np.int64)
    mask = all_targets_np != -1
    if not mask.all():
        all_preds_np = all_preds_np[mask]
        all_targets_np = all_targets_np[mask]

    if len(all_targets_np) == 0:
        return 0.0, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    acc = accuracy_score(all_targets_np, all_preds_np)
    return (
        float(acc),
        all_preds_np,
        all_targets_np,
    )


def print_frame_level_report(
    model_obj: nn.Module,
    loaders: dict[str, DataLoader],
    id_to_label: dict[int, str],
    num_classes: int,
    save_confusion_matrix_dir: str | None = None,
    save_confusion_splits: set[str] | None = None,
) -> pd.DataFrame:
    """Run window-level evaluation on multiple splits and print reports."""
    summary_rows = []
    for split_name, loader in loaders.items():
        acc, preds, targets = evaluate_model_full(
            model_obj, loader, num_classes, split_name,
        )
        print(f"\n{'='*10} {split_name.upper()} WINDOW-LEVEL REPORT {'='*10}")
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

        if (
            save_confusion_matrix_dir is not None
            and (
                save_confusion_splits is None
                or split_name in save_confusion_splits
            )
        ):
            from utils import save_confusion_matrix_plots

            class_names = [id_to_label[i] for i in range(num_classes)]
            split_slug = split_name.lower().replace(" ", "_")
            save_confusion_matrix_plots(
                y_true=targets,
                y_pred=preds,
                class_names=class_names,
                save_dir=save_confusion_matrix_dir,
                split_name=split_slug,
            )

    return pd.DataFrame(summary_rows)


# ──────────────────────────────────────────────────────────────────────
# Streaming metrics helpers (bridge from decoder output → metrics input)
# ──────────────────────────────────────────────────────────────────────

def _build_metrics_inputs(
    sample: dict,
    stream_steps: list[dict],
    emit_regions: list[tuple],
    label_to_id: dict[str, int],
    background_label: str,
    num_classes: int,
) -> tuple[list[int], list[int], list[int], list[int], int]:
    """Convert decoder output into the four arrays expected by the metrics modules.

    Returns
    -------
    frame_sequence : flat [s0, e0, s1, e1, ...] ground-truth boundaries
    y_true         : list of GT class indices (one per gesture, 0-based, no background)
    gating_list    : flat [s0, e0, s1, e1, ...] predicted boundaries
    y_pred_list    : per-frame class index, -1 for background
    seq_len        : total number of frames
    """
    seq_len = int(sample["num_frames"])

    # ── Ground truth: frame_sequence + y_true ──
    # sample["segmentation_regions"] contains non-background intervals:
    #   [{"label": "book", "start_frame": 10, "end_frame": 45}, ...]
    frame_sequence: list[int] = []
    y_true: list[int] = []

    gt_regions = sample.get("segmentation_regions", [])
    for region in gt_regions:
        label_str = region["label"]
        if label_str == background_label:
            continue
        cls_id = label_to_id.get(label_str)
        if cls_id is None:
            continue  # label unseen in training
        frame_sequence.append(int(region["start_frame"]))
        frame_sequence.append(int(region["end_frame"]))
        y_true.append(cls_id)

    # ── Predicted boundaries: gating_list ──
    # emit_regions is [(start_frame, end_frame, label_str), ...]
    gating_list: list[int] = []
    for (start, end, _label) in emit_regions:
        gating_list.append(int(start))
        gating_list.append(int(end))

    # ── Per-frame predictions: y_pred_list ──
    # Default to -1 (background). Fill from stream_steps.
    y_pred_list: list[int] = [-1] * seq_len

    bg_id = label_to_id.get(background_label)

    for step in stream_steps:
        fi = step.get("frame_index")
        if fi is None or fi < 0 or fi >= seq_len:
            continue
        voted_label_str = step.get("voted_label", background_label)
        if voted_label_str == background_label:
            continue  # leave as -1
        cls_id = label_to_id.get(voted_label_str)
        if cls_id is not None and cls_id != bg_id:
            y_pred_list[fi] = cls_id

    return frame_sequence, y_true, gating_list, y_pred_list, seq_len


def _build_metrics_inputs_from_wer_row(
    row: pd.Series,
    label_to_id: dict[str, int],
    background_label: str,
    num_classes: int,
) -> tuple[list[int], list[int], list[int], list[int], int]:
    """Convert one WER result row into the metric inputs expected by SHREC evaluators."""
    seq_len = int(row["num_frames"])

    frame_sequence: list[int] = []
    y_true: list[int] = []

    gt_segments = row.get("gt_segments", []) or []
    for region in gt_segments:
        label_str = str(region.get("label", background_label))
        if label_str == background_label:
            continue
        cls_id = label_to_id.get(label_str)
        if cls_id is None:
            continue
        frame_sequence.append(int(region["start_frame"]))
        frame_sequence.append(int(region["end_frame"]))
        y_true.append(cls_id)

    gating_list: list[int] = []
    emit_regions = row.get("emit_regions", []) or []
    for (start, end, _label) in emit_regions:
        gating_list.append(int(start))
        gating_list.append(int(end))

    y_pred_list: list[int] = [-1] * seq_len
    stream_steps = row.get("stream_steps", []) or []
    bg_id = label_to_id.get(background_label)

    for step in stream_steps:
        fi = step.get("frame_index")
        if fi is None or fi < 0 or fi >= seq_len:
            continue
        voted_label_str = step.get("voted_label", background_label)
        if voted_label_str == background_label:
            continue
        cls_id = label_to_id.get(voted_label_str)
        if cls_id is not None and cls_id != bg_id:
            y_pred_list[fi] = cls_id

    return frame_sequence, y_true, gating_list, y_pred_list, seq_len


# ──────────────────────────────────────────────────────────────────────
# SHREC'21 streaming metrics evaluation
# ──────────────────────────────────────────────────────────────────────

def evaluate_streaming_metrics(
    samples: list[dict],
    split_name: str,
    model_obj: nn.Module,
    normalize_fn: Callable[[np.ndarray], np.ndarray],
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    num_classes: int,
    background_label: str          = BACKGROUND_LABEL,
    window_size: int               = ONLINE_WINDOW_SIZE,
    stride: int                    = ONLINE_STRIDE,
    bag_size: int                  = BAG_SIZE,
    aggregation: str               = BAG_AGGREGATION,
    confidence_threshold: float    = CONFIDENCE_THRESHOLD,
    sign_bg_margin: float          = SIGN_BG_MARGIN,
    min_sign_frames: int           = MIN_SIGN_FRAMES,
    run_original: bool             = True,
    run_corrected: bool            = True,
) -> None:
    """Run SHREC'21 streaming metrics over a catalog of recording samples.

    Calls the existing streaming decoder per sequence, converts its output
    into the format expected by ``metrics_original`` and ``metrics_corrected``,
    and accumulates results globally. Call ``print_streaming_metrics_results``
    afterwards to display.

    Parameters
    ----------
    samples          : WER-catalog-style dicts (each has V, segmentation_regions, …)
    split_name       : label for logging (e.g. "Test (user3)")
    model_obj        : trained THCT-Net model with forward(sequences, lengths) method
    normalize_fn     : per-frame normalization function
    label_to_id      : string → class-index mapping
    id_to_label      : class-index → string mapping
    num_classes      : total number of classes (including background)
    run_original     : whether to run the buggy Duo Streamers baseline metrics
    run_corrected    : whether to run the corrected SHREC'21 metrics
    """
    # Build class names in index order for final printout
    class_names = [id_to_label.get(i, f"class_{i}") for i in range(num_classes)]

    # Initialize accumulators
    if run_original:
        init_orig(n_classes=num_classes)
    if run_corrected:
        init_corr(n_classes=num_classes)

    print(f"\n{'='*60}")
    print(f"STREAMING METRICS EVALUATION - {split_name}")
    print(f"{'='*60}")
    print(f"Sequences: {len(samples)} | Classes: {num_classes}")
    print(f"Running: {'original+corrected' if run_original and run_corrected else 'corrected' if run_corrected else 'original'}")
    print()

    for idx, sample in enumerate(samples, start=1):
        # Run decoder
        stream_steps, emitted_preds, emit_regions = stream_model_online(
            V=sample["V"],
            model_obj=model_obj,
            normalize_fn=normalize_fn,
            id_to_label=id_to_label,
            background_label=background_label,
            window_size=window_size,
            stride=stride,
            bag_size=bag_size,
            aggregation=aggregation,
            confidence_threshold=confidence_threshold,
            sign_bg_margin=sign_bg_margin,
            min_sign_frames=min_sign_frames,
            user=sample.get("user"),
        )

        # Convert to metrics format
        frame_sequence, y_true, gating_list, y_pred_list, seq_len = (
            _build_metrics_inputs(
                sample, stream_steps, emit_regions,
                label_to_id, background_label, num_classes,
            )
        )

        # Skip sequences with no GT gestures or no predictions
        if len(y_true) == 0:
            continue
        if len(gating_list) == 0:
            # No detections — still feed to metrics so FN is counted
            pass

        # Feed to original (buggy) metrics
        if run_original:
            eval_dr_orig(
                frame_sequence, y_true, gating_list, y_pred_list,
                n_classes=num_classes, verbose=False,
            )
            # FPR and Jaccard were commented-out in the notebook;
            # run them here for completeness, but note they have bugs.
            if len(gating_list) > 0:
                eval_fpr_orig(
                    frame_sequence, y_true, gating_list, y_pred_list,
                    n_classes=num_classes, verbose=False,
                )
                eval_jac_orig(
                    frame_sequence, y_true, gating_list, y_pred_list,
                    n_classes=num_classes, verbose=False,
                )

        # Feed to corrected metrics
        if run_corrected:
            eval_all_corr(
                frame_sequence, y_true, gating_list, y_pred_list,
                seq_len=seq_len, n_classes=num_classes, verbose=False,
            )

    # Print results
    if run_original:
        print(f"\n{'─'*60}")
        print(f"ORIGINAL METRICS (Duo Streamers baseline) — {split_name}")
        print(f"{'─'*60}")
        print_orig(class_names=class_names)

    if run_corrected:
        print(f"\n{'─'*60}")
        print(f"CORRECTED METRICS (SHREC'21 protocol) — {split_name}")
        print(f"{'─'*60}")
        print_corr(class_names=class_names)


def evaluate_streaming_metrics_from_wer_df(
    wer_df: pd.DataFrame,
    split_name: str,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    num_classes: int,
    background_label: str          = BACKGROUND_LABEL,
    run_original: bool             = True,
    run_corrected: bool            = True,
) -> None:
    """Run SHREC'21 streaming metrics from WER outputs already computed elsewhere.

    This reuses the saved WER step outputs (`stream_steps`, `emit_regions`, `gt_segments`)
    and does not rerun the decoder.
    """
    class_names = [id_to_label.get(i, f"class_{i}") for i in range(num_classes)]

    if run_original:
        init_orig(n_classes=num_classes)
    if run_corrected:
        init_corr(n_classes=num_classes)

    print(f"\n{'='*60}")
    print(f"STREAMING METRICS EVALUATION (from WER outputs) — {split_name}")
    print(f"{'='*60}")
    print(f"Sequences: {len(wer_df)} | Classes: {num_classes}")
    print(f"Running: {'original+corrected' if run_original and run_corrected else 'corrected' if run_corrected else 'original'}")
    print()

    for _, row in wer_df.iterrows():
        frame_sequence, y_true, gating_list, y_pred_list, seq_len = (
            _build_metrics_inputs_from_wer_row(
                row=row,
                label_to_id=label_to_id,
                background_label=background_label,
                num_classes=num_classes,
            )
        )

        if len(y_true) == 0:
            continue

        if run_original:
            eval_dr_orig(
                frame_sequence, y_true, gating_list, y_pred_list,
                n_classes=num_classes, verbose=False,
            )
            if len(gating_list) > 0:
                eval_fpr_orig(
                    frame_sequence, y_true, gating_list, y_pred_list,
                    n_classes=num_classes, verbose=False,
                )
                eval_jac_orig(
                    frame_sequence, y_true, gating_list, y_pred_list,
                    n_classes=num_classes, verbose=False,
                )

        if run_corrected:
            eval_all_corr(
                frame_sequence, y_true, gating_list, y_pred_list,
                seq_len=seq_len, n_classes=num_classes, verbose=False,
            )

    if run_original:
        print(f"\n{'='*60}")
        print(f"ORIGINAL METRICS (Duo Streamers baseline) - {split_name}")
        print(f"{'='*60}")
        print_orig(class_names=class_names)

    if run_corrected:
        print(f"\n{'='*60}")
        print(f"CORRECTED METRICS (SHREC'21 protocol) - {split_name}")
        print(f"{'='*60}")
        print_corr(class_names=class_names)


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


def evaluate_model_wer(
    samples: list[dict],
    split_name: str,
    model_obj: nn.Module,
    normalize_fn: Callable[[np.ndarray], np.ndarray],
    id_to_label: dict[int, str],
    normalization_name: str,
    print_examples: int         = 2,
    window_size: int            = ONLINE_WINDOW_SIZE,
    stride: int                 = ONLINE_STRIDE,
    bag_size: int               = BAG_SIZE,
    aggregation: str            = BAG_AGGREGATION,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    sign_bg_margin: float       = SIGN_BG_MARGIN,
    min_sign_frames: int        = MIN_SIGN_FRAMES,
) -> pd.DataFrame:
    """Evaluate streaming WER for the THCT-Net model over a sequence set.

    Returns
    -------
    pd.DataFrame — one row per sample with all WER fields.
    """
    rows = []

    for idx, sample in enumerate(samples, start=1):
        stream_steps, final_preds, emit_regions = stream_model_online(
            V=sample["V"],
            model_obj=model_obj,
            normalize_fn=normalize_fn,
            id_to_label=id_to_label,
            background_label=BACKGROUND_LABEL,
            window_size=window_size,
            stride=stride,
            bag_size=bag_size,
            aggregation=aggregation,
            confidence_threshold=confidence_threshold,
            sign_bg_margin=sign_bg_margin,
            min_sign_frames=min_sign_frames,
            user=sample.get("user"),
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


# ──────────────────────────────────────────────────────────────────────
# Persist streaming results (metadata + per-frame arrays)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_RESULTS_DIR = "wer_results"


def save_split_results(
    df: pd.DataFrame,
    split_name: str,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> dict[str, str]:
    """Save WER DataFrame metadata and per-frame arrays for one split.

    Two files per split:
        {slug}_metadata.parquet — all scalar/list columns, one row per sequence
        {slug}_arrays.npz       — per-frame arrays (logits, probs, labels, …)
                                   indexed by recording_id

    Parameters
    ----------
    df          : DataFrame returned by evaluate_model_wer()
    split_name  : e.g. "Test (user3)" — used as filename slug
    results_dir : directory to write into (created if missing)

    Returns
    -------
    dict with paths {"parquet": ..., "npz": ...}
    """
    os.makedirs(results_dir, exist_ok=True)
    slug = (
        split_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    # ── 1. Metadata (drop heavy stream_steps column) ──
    df_meta = df.drop(columns=["stream_steps"], errors="ignore").copy()

    # Serialize list-of-tuple columns to JSON strings for parquet compat
    for col in ["emit_regions", "gt_segments"]:
        if col in df_meta.columns:
            df_meta[col] = df_meta[col].apply(
                lambda x: json.dumps(x) if x is not None else "[]"
            )

    parquet_path = os.path.join(results_dir, f"{slug}_metadata.parquet")
    df_meta.to_parquet(parquet_path, index=False)

    # ── 2. Per-frame arrays (logits, probs, labels, etc.) ──
    arrays_dict: dict[str, np.ndarray] = {}

    for row_idx, row in df.iterrows():
        steps = row.get("stream_steps")
        if not steps:
            continue

        rec_id = str(row["recording_id"]).replace("/", "_")
        prefix = f"{row_idx}__{rec_id}"

        # Infer num_classes from first non-None logit
        C = None
        for s in steps:
            pre = s.get("pre_bag_logits")
            if pre is not None:
                C = len(pre)
                break
        if C is None:
            for s in steps:
                post = s.get("post_bag_probs")
                if post is not None:
                    C = len(post)
                    break
        if C is None:
            continue  # no logit data at all

        nan_fill = np.full(C, np.nan, dtype=np.float32)

        pre_bag_list = [
            s.get("pre_bag_logits") if s.get("pre_bag_logits") is not None
            else nan_fill
            for s in steps
        ]
        post_bag_list = [
            s.get("post_bag_probs") if s.get("post_bag_probs") is not None
            else nan_fill
            for s in steps
        ]

        arrays_dict[f"{prefix}__pre_bag_logits"] = np.stack(
            pre_bag_list, axis=0
        ).astype(np.float32)
        arrays_dict[f"{prefix}__post_bag_probs"] = np.stack(
            post_bag_list, axis=0
        ).astype(np.float32)
        arrays_dict[f"{prefix}__frame_indices"] = np.array(
            [s["frame_index"] for s in steps], dtype=np.int32,
        )
        arrays_dict[f"{prefix}__raw_labels"] = np.array(
            [s["raw_label"] for s in steps], dtype=object,
        )
        arrays_dict[f"{prefix}__voted_labels"] = np.array(
            [s["voted_label"] for s in steps], dtype=object,
        )
        arrays_dict[f"{prefix}__raw_conf"] = np.array(
            [float(s["raw_conf"]) for s in steps], dtype=np.float32,
        )
        arrays_dict[f"{prefix}__bg_conf"] = np.array(
            [float(s["bg_conf"]) for s in steps], dtype=np.float32,
        )
        arrays_dict[f"{prefix}__states"] = np.array(
            [s["state"] for s in steps], dtype=object,
        )

    npz_path = os.path.join(results_dir, f"{slug}_arrays.npz")
    np.savez_compressed(npz_path, **arrays_dict)

    print(f"[{split_name}] metadata -> {parquet_path}")
    print(f"[{split_name}] arrays   -> {npz_path}")

    print(f"  sequences : {len(df)}")
    print(f"  npz keys  : {len(arrays_dict)}")

    return {"parquet": parquet_path, "npz": npz_path}


def load_split_results(
    split_name: str,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> tuple[pd.DataFrame, dict]:
    """Load metadata DataFrame and per-frame arrays for one split.

    Returns
    -------
    df     : DataFrame with scalar/list columns restored
    arrays : dict — keys are "{row_idx}__{rec_id}__{field}", values np.ndarray
    """
    slug = (
        split_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )
    parquet_path = os.path.join(results_dir, f"{slug}_metadata.parquet")
    npz_path     = os.path.join(results_dir, f"{slug}_arrays.npz")

    df = pd.read_parquet(parquet_path)

    # Deserialize JSON strings back to Python lists
    for col in ["emit_regions", "gt_segments"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: (
                    [tuple(r) for r in json.loads(x)]
                    if isinstance(x, str) else []
                )
            )

    npz    = np.load(npz_path, allow_pickle=True)
    arrays = {k: npz[k] for k in npz.files}

    return df, arrays


def get_sequence_arrays(
    row: pd.Series,
    arrays: dict,
) -> dict:
    """Extract all per-frame arrays for one DataFrame row.

    Usage
    -----
    df, arrays = load_split_results("test (user3)")
    seq = get_sequence_arrays(df.iloc[0], arrays)
    seq["pre_bag_logits"]   # (T, C)
    seq["post_bag_probs"]   # (T, C)  — first (bag_size-1) rows are NaN
    seq["frame_indices"]    # (T,)
    """
    row_idx = row.name
    rec_id  = str(row["recording_id"]).replace("/", "_")
    prefix  = f"{row_idx}__{rec_id}"

    fields = [
        "pre_bag_logits", "post_bag_probs",
        "frame_indices",
        "raw_labels", "voted_labels",
        "raw_conf", "bg_conf",
        "states",
    ]
    return {
        field: arrays.get(f"{prefix}__{field}")
        for field in fields
    }
