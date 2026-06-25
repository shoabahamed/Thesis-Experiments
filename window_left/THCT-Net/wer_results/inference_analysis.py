"""
visualize_decoder.py
--------------------
Visualization functions for streaming decoder results saved by
save_split_results() / load_split_results().

All functions work from the two files per split:
    {split}_metadata.parquet   — scalar + list columns
    {split}_arrays.npz         — per-frame logit arrays

Public API
----------
load_split_results(split_name)              → df, arrays
get_sequence_arrays(row, arrays)            → dict of (T, *) arrays

plot_sequence_timeline(row, arrays, ...)    → Figure
    Three-row plot:
        Row 0 — GT segmentation regions
        Row 1 — per-frame decoder vote (pre-bag top-1, colored by label)
        Row 2 — confidence curves (pre-bag vs post-bag top-1, bg conf,
                 threshold lines, emission markers)

plot_confidence_heatmap(row, arrays, ...)   → Figure
    Class × Frame heatmap of post-bag probabilities with GT and
    emit-region overlays.

plot_wer_distribution(df, ...)              → Figure
    WER histogram + per-user box plot across a split.

plot_split_overview(df, arrays, ...)        → Figure
    Grid of mini-timelines for all sequences in a split — quick
    overview of where errors occur.

plot_confusion_from_emissions(df, ...)      → Figure
    Sign-level confusion matrix built from emit_regions vs gt_segments.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Constants — must match decoder constants
# ---------------------------------------------------------------------------

RESULTS_DIR          = "./"
LEAP_FPS             = 30
BAG_SIZE             = 5
CONFIDENCE_THRESHOLD = 0.35
SIGN_BG_MARGIN       = 0.10
BACKGROUND_LABEL     = "background"    # adjust if your project uses a different name


# ---------------------------------------------------------------------------
# I/O helpers (mirrors the save/load functions in the decoder file)
# ---------------------------------------------------------------------------

def load_split_results(
    split_name: str,
    results_dir: str = RESULTS_DIR,
) -> tuple[pd.DataFrame, dict]:
    """
    Load metadata DataFrame and per-frame arrays for one split.

    Parameters
    ----------
    split_name  : e.g. "test", "dev_val", "test_(user1)" — must match
                  the slug used when saving
    results_dir : directory that contains the parquet / npz files

    Returns
    -------
    df     : DataFrame with all WER scalars, GT glosses, predictions,
             emit_regions (list of tuples), gt_segments (list of tuples)
    arrays : dict keyed by "{row_idx}__{rec_id}__{field}"
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

    for col in ["emit_regions", "gt_segments"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: [tuple(r) for r in json.loads(x)]
                if isinstance(x, str) else []
            )

    npz    = np.load(npz_path, allow_pickle=True)
    arrays = {k: npz[k] for k in npz.files}

    return df, arrays


def get_sequence_arrays(row: pd.Series, arrays: dict) -> dict:
    """
    Extract all per-frame arrays for one DataFrame row.

    Returns
    -------
    dict with keys:
        pre_bag_logits  (T, C) float32  — raw model logits before bag
        post_bag_probs  (T, C) float32  — first (BAG_SIZE-1) rows are NaN
        frame_indices   (T,)   int32
        raw_labels      (T,)   object
        voted_labels    (T,)   object
        raw_conf        (T,)   float32
        bg_conf         (T,)   float32
        states          (T,)   object
    """
    row_idx = row.name
    rec_id  = str(row["recording_id"]).replace("/", "_")
    prefix  = f"{row_idx}__{rec_id}"

    fields = [
        "pre_bag_logits", "post_bag_probs",
        "frame_indices",
        "raw_labels",     "voted_labels",
        "raw_conf",       "bg_conf",
        "states",
    ]
    return {f: arrays.get(f"{prefix}__{f}") for f in fields}


# ---------------------------------------------------------------------------
# Color palette helpers
# ---------------------------------------------------------------------------

def _build_label_colors(
    all_labels: Sequence[str],
    background_label: str = BACKGROUND_LABEL,
    cmap_name: str = "tab20",
) -> dict[str, tuple]:
    """
    Assign a stable color to every label.
    Background is always light grey.
    """
    sign_labels = sorted(set(all_labels) - {background_label})
    # use newer matplotlib API
    try:
        cmap = plt.colormaps[cmap_name]
    except AttributeError:
        cmap = plt.cm.get_cmap(cmap_name)
    # create list of colors of required length
    if len(sign_labels) > 0:
        colors = [cmap(i) for i in range(len(sign_labels))]
    else:
        colors = []
    color_dict = {lbl: colors[i] for i, lbl in enumerate(sign_labels)}
    color_dict[background_label] = (0.88, 0.88, 0.88, 1.0)
    return color_dict


# ---------------------------------------------------------------------------
# Helper to safely unpack segment tuples (ensure ints)
# ---------------------------------------------------------------------------

def _safe_segment_tuple(seg):
    """Convert a segment tuple/list to (int, int, str)."""
    if isinstance(seg, (tuple, list)) and len(seg) >= 3:
        try:
            return (int(seg[0]), int(seg[1]), str(seg[2]))
        except (ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Plot 1 — Sequence timeline
# ---------------------------------------------------------------------------

def plot_sequence_timeline(
    row: pd.Series,
    arrays: dict,
    label_colors: dict | None = None,
    background_label: str     = BACKGROUND_LABEL,
    figsize: tuple            = (20, 7),
    title: str | None         = None,
    save_path: str | None     = None,
) -> plt.Figure:
    """
    Three-row timeline for one sequence.

    Row 0 — Ground-truth segmentation regions (colored bars)
    Row 1 — Per-frame decoder voted label (pre-bag top-1, colored by label)
             Red dashed vertical lines mark emission events.
    Row 2 — Confidence curves
             pre-bag top-1 conf  (blue)
             post-bag top-1 conf (green)
             background conf     (grey dashed)
             CONFIDENCE_THRESHOLD (red dotted)
             SIGN_BG_MARGIN       (orange dotted)

    Parameters
    ----------
    row          : one row from a split DataFrame
    arrays       : arrays dict from load_split_results()
    label_colors : optional pre-built color dict (shared across calls)
    save_path    : if given, saves figure to this path instead of showing
    """
    seq        = get_sequence_arrays(row, arrays)
    frames     = seq["frame_indices"]           # (T,)
    raw_labels = seq["raw_labels"]              # (T,) pre-bag top-1
    raw_conf   = seq["raw_conf"]                # (T,)
    bg_conf    = seq["bg_conf"]                 # (T,) post-bag bg confidence

    # Post-bag top-1 confidence — NaN where bag not yet full
    post_bag   = seq["post_bag_probs"]          # (T, C) or None
    if post_bag is not None:
        # suppress warnings for all-NaN slices
        with np.errstate(invalid='ignore'):
            post_conf = np.nanmax(post_bag, axis=1)  # (T,)
    else:
        post_conf = np.full_like(raw_conf, np.nan)

    T           = frames[-1] + 1 if frames is not None and len(frames) > 0 else 1
    gt_segments = row.get("gt_segments", [])    # [(start, end, label), ...]
    emit_regions= row.get("emit_regions", [])   # [(start, end, label), ...]

    # Convert to safe tuples
    gt_segments = [_safe_segment_tuple(s) for s in gt_segments if _safe_segment_tuple(s) is not None]
    emit_regions = [_safe_segment_tuple(r) for r in emit_regions if _safe_segment_tuple(r) is not None]

    # Collect all labels for color palette
    all_labels  = set(raw_labels.tolist()) if raw_labels is not None else set()
    all_labels |= {seg[2] for seg in gt_segments}
    all_labels |= {r[2]   for r in emit_regions}
    all_labels.add(background_label)

    if label_colors is None:
        label_colors = _build_label_colors(all_labels, background_label)

    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs  = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1, 1, 2.5],
        hspace=0.08,
    )
    ax0 = fig.add_subplot(gs[0])   # GT regions
    ax1 = fig.add_subplot(gs[1])   # per-frame voted label
    ax2 = fig.add_subplot(gs[2])   # confidence curves

    # ---- Row 0 : GT segmentation regions ----
    if gt_segments:
        for (start, end, label) in gt_segments:
            color = label_colors.get(label, "steelblue")
            ax0.barh(
                0, end - start, left=start, height=0.6,
                color=color, edgecolor="black", linewidth=0.5,
            )
            cx = (start + end) / 2
            ax0.text(
                cx, 0, label,
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="white" if sum(color[:3]) < 1.5 else "black",
                clip_on=True,
            )
    else:
        ax0.text(
            0.5, 0.5, "No GT segments available",
            ha="center", va="center", transform=ax0.transAxes, fontsize=9,
        )
    ax0.set_yticks([0])
    ax0.set_yticklabels(["GT"], fontsize=9)
    ax0.set_xlim(0, T)
    ax0.set_ylim(-0.5, 0.5)
    ax0.tick_params(bottom=False, labelbottom=False)
    ax0.set_title(
        title or (
            f"{row.get('user', '?')} | {row.get('recording_id', '?')} | "
            f"WER={float(row['wer']):.3f}  "
            f"GT: {row.get('ground_truth', '')}  "
            f"Pred: {row.get('prediction', '')}"
        ),
        fontsize=10, loc="left",
    )

    # ---- Row 1 : per-frame raw (pre-bag) voted label ----
    if raw_labels is not None and len(raw_labels) == len(frames):
        for i, fi in enumerate(frames):
            label = str(raw_labels[i])
            color = label_colors.get(label, "lightgrey")
            ax1.barh(0, 1, left=fi, height=0.6, color=color, edgecolor="none")

    # Emission markers — vertical red dashed lines + label at top
    for (start, end, label) in emit_regions:
        mid = (start + end) / 2
        ax1.axvline(mid, color="red", linewidth=1.5, linestyle="--", alpha=0.85)
        ax1.text(
            mid, 0.45, label,
            ha="center", va="top", fontsize=6.5, color="red",
            rotation=90, clip_on=True,
        )

    ax1.set_yticks([0])
    ax1.set_yticklabels(["Pre-bag\nvote"], fontsize=9)
    ax1.set_xlim(0, T)
    ax1.set_ylim(-0.5, 0.5)
    ax1.tick_params(bottom=False, labelbottom=False)

    # ---- Row 2 : confidence curves ----
    ax2.plot(frames, raw_conf,  color="steelblue",   linewidth=1.2,
             label="Pre-bag top-1 conf", zorder=3)
    ax2.plot(frames, post_conf, color="seagreen",    linewidth=1.2,
             label="Post-bag top-1 conf", zorder=3)
    ax2.plot(frames, bg_conf,   color="grey",        linewidth=1.0,
             linestyle="--", alpha=0.7, label="Post-bag BG conf", zorder=2)

    ax2.axhline(
        CONFIDENCE_THRESHOLD, color="red",    linewidth=1.0,
        linestyle=":", label=f"Conf threshold ({CONFIDENCE_THRESHOLD})", zorder=1,
    )
    ax2.axhline(
        SIGN_BG_MARGIN, color="darkorange", linewidth=1.0,
        linestyle=":", label=f"BG margin ({SIGN_BG_MARGIN})", zorder=1,
    )

    # Shade IN_SIGN regions from decoder state
    states    = seq["states"]
    if states is not None and len(states) == len(frames):
        in_sign   = False
        sign_start= None
        for i, fi in enumerate(frames):
            st = str(states[i])
            if st == "IN_SIGN" and not in_sign:
                in_sign    = True
                sign_start = fi
            elif st != "IN_SIGN" and in_sign:
                ax2.axvspan(sign_start, fi, alpha=0.08, color="steelblue", zorder=0)
                in_sign = False
        if in_sign:
            ax2.axvspan(sign_start, frames[-1], alpha=0.08, color="steelblue", zorder=0)

    # Emission markers on confidence plot too
    for (start, end, label) in emit_regions:
        mid = (start + end) / 2
        ax2.axvline(mid, color="red", linewidth=1.2, linestyle="--", alpha=0.7)

    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(0, T)
    ax2.set_xlabel("Frame index", fontsize=9)
    ax2.set_ylabel("Confidence", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right", ncol=3)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.2))

    # Shared legend for label colors
    handles = [
        mpatches.Patch(color=c, label=l)
        for l, c in label_colors.items()
        if l != background_label
    ]
    handles.append(mpatches.Patch(color=label_colors[background_label], label=background_label))
    handles.append(Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Emission"))
    fig.legend(
        handles=handles, loc="lower center", fontsize=7,
        ncol=min(len(handles), 8), bbox_to_anchor=(0.5, -0.02),
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Plot 2 — Post-bag probability heatmap
# ---------------------------------------------------------------------------

def plot_confidence_heatmap(
    row: pd.Series,
    arrays: dict,
    id_to_label: dict[int, str],
    background_label: str = BACKGROUND_LABEL,
    figsize: tuple        = (20, 6),
    title: str | None     = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Class × Frame heatmap of post-bag probabilities.

    Shows which classes the model was uncertain between over time.
    GT segmentation boundaries and emit region markers overlaid on top.

    Parameters
    ----------
    id_to_label : {class_index: label_string} — same dict used during inference
    """
    seq      = get_sequence_arrays(row, arrays)
    frames   = seq["frame_indices"]
    post_bag = seq["post_bag_probs"]    # (T, C)

    if post_bag is None or np.all(np.isnan(post_bag)):
        print(f"No post-bag probs available for {row.get('recording_id', '?')}")
        return None

    T, C       = post_bag.shape
    label_list = [id_to_label.get(i, f"cls_{i}") for i in range(C)]

    # Sort rows: background last, rest alphabetically
    sign_idxs = sorted(
        [i for i, l in enumerate(label_list) if l != background_label],
        key=lambda i: label_list[i],
    )
    bg_idxs = [i for i, l in enumerate(label_list) if l == background_label]
    row_order = sign_idxs + bg_idxs
    ordered_labels = [label_list[i] for i in row_order]
    ordered_probs  = post_bag[:, row_order].T      # (C, T)

    gt_segments  = [_safe_segment_tuple(s) for s in row.get("gt_segments",  []) if _safe_segment_tuple(s) is not None]
    emit_regions = [_safe_segment_tuple(r) for r in row.get("emit_regions", []) if _safe_segment_tuple(r) is not None]

    fig, axes = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [1, 6]},
        constrained_layout=True,
    )
    ax_gt  = axes[0]
    ax_hm  = axes[1]

    # ---- GT bar ----
    all_labels   = {seg[2] for seg in gt_segments} | {background_label}
    label_colors = _build_label_colors(all_labels, background_label)

    if gt_segments:
        for (start, end, label) in gt_segments:
            ax_gt.barh(
                0, end - start, left=start, height=0.6,
                color=label_colors.get(label, "steelblue"),
                edgecolor="black", linewidth=0.4,
            )
            ax_gt.text(
                (start + end) / 2, 0, label,
                ha="center", va="center", fontsize=7, fontweight="bold",
            )
    ax_gt.set_xlim(frames[0], frames[-1] + 1)
    ax_gt.set_ylim(-0.5, 0.5)
    ax_gt.set_yticks([0])
    ax_gt.set_yticklabels(["GT"], fontsize=8)
    ax_gt.tick_params(bottom=False, labelbottom=False)
    ax_gt.set_title(
        title or (
            f"{row.get('user', '?')} | {row.get('recording_id', '?')} | "
            f"WER={float(row['wer']):.3f}"
        ),
        fontsize=10, loc="left",
    )

    # ---- Heatmap ----
    im = ax_hm.imshow(
        ordered_probs,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        vmin=0, vmax=1,
        extent=[frames[0], frames[-1] + 1, -0.5, C - 0.5],
        interpolation="nearest",
    )

    # GT boundaries as white vertical lines
    for (start, end, _) in gt_segments:
        ax_hm.axvline(start, color="white", linewidth=1.2, alpha=0.9)
        ax_hm.axvline(end,   color="white", linewidth=1.2, alpha=0.9)

    # Emission markers
    for (start, end, label) in emit_regions:
        mid = (start + end) / 2
        ax_hm.axvline(mid, color="cyan", linewidth=1.5, linestyle="--", alpha=0.9)

    ax_hm.set_yticks(range(C))
    ax_hm.set_yticklabels(ordered_labels, fontsize=7)
    ax_hm.set_xlabel("Frame index", fontsize=9)

    # Background class separator line
    n_sign = len(sign_idxs)
    ax_hm.axhline(n_sign - 0.5, color="white", linewidth=1.5, linestyle="--")

    plt.colorbar(im, ax=ax_hm, label="Post-bag probability", fraction=0.015, pad=0.01)

    # Legend for overlay markers
    legend_handles = [
        Line2D([0], [0], color="white",  linewidth=1.2, label="GT boundary"),
        Line2D([0], [0], color="cyan",   linewidth=1.5, linestyle="--", label="Emission"),
    ]
    ax_hm.legend(handles=legend_handles, fontsize=8, loc="upper right")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Plot 3 — WER distribution across a split
# ---------------------------------------------------------------------------

def plot_wer_distribution(
    df: pd.DataFrame,
    split_name: str       = "",
    figsize: tuple        = (14, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Two-panel WER distribution plot for an entire split.

    Left  — histogram of WER values with mean/median lines
    Right — per-user box plot (one box per unique user)

    Parameters
    ----------
    df         : DataFrame from load_split_results()
    split_name : used in the figure title
    """
    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    fig.suptitle(f"WER Distribution — {split_name}", fontsize=12)

    wer = df["wer"].dropna().values

    # ---- Left: histogram ----
    ax_hist.hist(wer, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
    ax_hist.axvline(wer.mean(),   color="red",    linewidth=1.5, linestyle="--",
                    label=f"Mean   {wer.mean():.3f}")
    ax_hist.axvline(np.median(wer), color="orange", linewidth=1.5, linestyle="--",
                    label=f"Median {np.median(wer):.3f}")
    ax_hist.set_xlabel("WER", fontsize=9)
    ax_hist.set_ylabel("Count", fontsize=9)
    ax_hist.set_title("WER histogram", fontsize=10)
    ax_hist.legend(fontsize=8)
    ax_hist.set_xlim(0, max(1.0, wer.max() + 0.05))

    # Annotate std
    ax_hist.text(
        0.98, 0.97,
        f"n={len(wer)}\nstd={wer.std():.3f}",
        ha="right", va="top", transform=ax_hist.transAxes,
        fontsize=8, color="grey",
    )

    # ---- Right: per-user box plot ----
    if "user" in df.columns and df["user"].nunique() > 1:
        users      = sorted(df["user"].unique())
        user_wers  = [df.loc[df["user"] == u, "wer"].values for u in users]
        bp = ax_box.boxplot(
            user_wers, labels=users, patch_artist=True,
            medianprops=dict(color="red", linewidth=1.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.5)
        ax_box.set_xlabel("User", fontsize=9)
        ax_box.set_ylabel("WER", fontsize=9)
        ax_box.set_title("WER by user", fontsize=10)
        ax_box.tick_params(axis="x", rotation=30)
        ax_box.set_ylim(0, max(1.0, wer.max() + 0.05))
    else:
        # Only one user — show per-sequence scatter instead
        ax_box.scatter(
            df["sample_idx"], df["wer"],
            color="steelblue", s=25, alpha=0.7,
        )
        ax_box.set_xlabel("Sequence index", fontsize=9)
        ax_box.set_ylabel("WER", fontsize=9)
        ax_box.set_title("WER per sequence", fontsize=10)
        ax_box.set_ylim(0, max(1.0, wer.max() + 0.05))

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Plot 4 — Split overview grid (mini-timelines)
# ---------------------------------------------------------------------------

def plot_split_overview(
    df: pd.DataFrame,
    arrays: dict,
    background_label: str = BACKGROUND_LABEL,
    max_sequences: int    = 30,
    figsize: tuple        = (22, None),  # height auto-computed
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grid of mini-timelines for all (up to max_sequences) sequences in a split.

    Each row shows:
        Top    — GT segmentation
        Bottom — per-frame pre-bag voted label

    Red = WER > 0, green = WER == 0.
    Useful for a quick overview of where the decoder fails.

    Parameters
    ----------
    max_sequences : cap to avoid unreadable figures
    """
    rows_to_show = df.head(max_sequences)
    n            = len(rows_to_show)

    if n == 0:
        print("DataFrame is empty — nothing to plot.")
        return None

    # Collect all labels for a shared color palette
    all_labels = set()
    for _, row in rows_to_show.iterrows():
        seq = get_sequence_arrays(row, arrays)
        if seq["raw_labels"] is not None:
            all_labels.update(seq["raw_labels"].tolist())
        # add GT labels
        for seg in row.get("gt_segments", []):
            t = _safe_segment_tuple(seg)
            if t:
                all_labels.add(t[2])
    all_labels.add(background_label)
    label_colors = _build_label_colors(all_labels, background_label)

    row_height   = 0.55    # inches per mini-row
    auto_height  = max(6, n * row_height * 2 + 2)
    fig_height   = figsize[1] if figsize[1] else auto_height
    fig, axes    = plt.subplots(
        n, 2, figsize=(figsize[0], fig_height),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.02, "hspace": 0.05},
        squeeze=False,
    )
    fig.suptitle(
        f"Split overview — {len(rows_to_show)} sequences  "
        f"(mean WER={df['wer'].mean():.3f})",
        fontsize=11,
    )

    for plot_idx, (_, row) in enumerate(rows_to_show.iterrows()):
        ax_gt   = axes[plot_idx][0]
        ax_pred = axes[plot_idx][1]

        seq    = get_sequence_arrays(row, arrays)
        frames = seq["frame_indices"]
        T      = (frames[-1] + 1) if frames is not None and len(frames) > 0 else 1

        wer_color = "green" if float(row["wer"]) == 0.0 else "red"

        # GT segments (safe conversion)
        gt_segs = [_safe_segment_tuple(s) for s in row.get("gt_segments", []) if _safe_segment_tuple(s) is not None]
        for (start, end, label) in gt_segs:
            c = label_colors.get(label, "steelblue")
            ax_gt.barh(0, end - start, left=start, height=0.6,
                       color=c, edgecolor="none")
        ax_gt.set_xlim(0, T)
        ax_gt.set_ylim(-0.5, 0.5)
        ax_gt.axis("off")
        ax_gt.set_ylabel(
            f"{row.get('recording_id', '')[:18]}",
            fontsize=6, rotation=0, ha="right", va="center",
        )
        # WER label on the left
        ax_gt.text(
            -0.01, 0.5,
            f"WER={float(row['wer']):.2f}",
            transform=ax_gt.transAxes, fontsize=6,
            ha="right", va="center", color=wer_color,
        )

        # Per-frame pred
        if seq["raw_labels"] is not None and len(seq["raw_labels"]) == len(frames):
            for i, fi in enumerate(frames):
                label = str(seq["raw_labels"][i])
                c     = label_colors.get(label, "lightgrey")
                ax_pred.barh(0, 1, left=fi, height=0.6, color=c, edgecolor="none")

        # Emission markers
        emit_regs = [_safe_segment_tuple(r) for r in row.get("emit_regions", []) if _safe_segment_tuple(r) is not None]
        for (start, end, _) in emit_regs:
            ax_pred.axvline((start + end) / 2, color="red",
                            linewidth=0.8, linestyle="--", alpha=0.8)

        ax_pred.set_xlim(0, T)
        ax_pred.set_ylim(-0.5, 0.5)
        ax_pred.axis("off")

    # Column headers on first row only
    axes[0][0].set_title("Ground truth", fontsize=9)
    axes[0][1].set_title("Pre-bag decoder vote", fontsize=9)

    # Shared legend
    handles = [
        mpatches.Patch(color=c, label=l)
        for l, c in label_colors.items()
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=min(len(handles), 10),
        fontsize=6, bbox_to_anchor=(0.5, -0.01),
    )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=130)
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Plot 5 — Sign-level confusion matrix from emissions vs GT
# ---------------------------------------------------------------------------

def plot_confusion_from_emissions(
    df: pd.DataFrame,
    background_label: str = BACKGROUND_LABEL,
    figsize: tuple        = (12, 10),
    normalize: bool       = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Sign-level confusion matrix built by aligning emit_regions with gt_segments.

    Alignment strategy: for each GT segment, the emitted sign whose region
    midpoint falls inside the GT segment is its prediction.
    Unmatched GT segments → "deletion" (missing prediction).
    Unmatched emissions   → "insertion" (spurious prediction).

    Parameters
    ----------
    normalize : if True, normalize rows to sum to 1 (shows recall per class)
    """
    from collections import defaultdict

    # Collect all sign labels
    all_signs = set()
    for _, row in df.iterrows():
        for seg in row.get("gt_segments", []):
            t = _safe_segment_tuple(seg)
            if t and t[2] != background_label:
                all_signs.add(t[2])
        for reg in row.get("emit_regions", []):
            t = _safe_segment_tuple(reg)
            if t and t[2] != background_label:
                all_signs.add(t[2])

    labels      = sorted(all_signs)
    label_to_i  = {l: i for i, l in enumerate(labels)}
    n           = len(labels)

    if n == 0:
        print("No sign labels found in emit_regions / gt_segments.")
        return None

    # Count: rows = GT, cols = Pred; extra col for "deletion" (no match)
    conf_matrix  = np.zeros((n, n), dtype=np.float32)
    deletions    = np.zeros(n,      dtype=np.float32)
    insertions   = 0

    for _, row in df.iterrows():
        gt_segs    = [_safe_segment_tuple(s) for s in row.get("gt_segments", []) if _safe_segment_tuple(s) is not None]
        gt_segs    = [s for s in gt_segs if s[2] != background_label]
        emit_regs  = [_safe_segment_tuple(r) for r in row.get("emit_regions", []) if _safe_segment_tuple(r) is not None]
        emit_regs  = [r for r in emit_regs if r[2] != background_label]

        matched_emit = set()

        for (gs, ge, gl) in gt_segs:
            gt_i   = label_to_i.get(gl)
            if gt_i is None:
                continue

            # Find emission whose midpoint falls inside this GT region
            match  = None
            for eidx, (es, ee, el) in enumerate(emit_regs):
                mid = (es + ee) / 2
                if gs <= mid <= ge:
                    match = (eidx, el)
                    break

            if match is not None:
                eidx, el = match
                pred_i   = label_to_i.get(el)
                matched_emit.add(eidx)
                if pred_i is not None:
                    conf_matrix[gt_i, pred_i] += 1
            else:
                deletions[gt_i] += 1

        # Count unmatched emissions as insertions
        for eidx in range(len(emit_regs)):
            if eidx not in matched_emit:
                insertions += 1

    # Normalize rows (recall perspective)
    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True) + deletions[:, np.newaxis]
        row_sums = np.where(row_sums == 0, 1, row_sums)
        plot_matrix = conf_matrix / row_sums
    else:
        plot_matrix = conf_matrix

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    im = ax.imshow(plot_matrix, cmap="Blues", vmin=0, vmax=1 if normalize else None)

    # Annotate cells
    thresh = plot_matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = plot_matrix[i, j]
            if val > 0:
                ax.text(
                    j, i, f"{val:.2f}" if normalize else f"{int(val)}",
                    ha="center", va="center", fontsize=7,
                    color="white" if val > thresh else "black",
                )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted label", fontsize=9)
    ax.set_ylabel("Ground truth label", fontsize=9)
    ax.set_title(
        f"Sign confusion matrix ({'normalized recall' if normalize else 'counts'})\n"
        f"Total deletions: {int(deletions.sum())}  |  Total insertions: {insertions}",
        fontsize=10,
    )

    plt.colorbar(im, ax=ax, label="Recall" if normalize else "Count",
                 fraction=0.03, pad=0.02)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Convenience — plot all sequences in a split one by one
# ---------------------------------------------------------------------------

def plot_all_sequences(
    df: pd.DataFrame,
    arrays: dict,
    id_to_label: dict[int, str],
    output_dir: str           = "./plots",
    background_label: str     = BACKGROUND_LABEL,
    include_heatmap: bool     = True,
) -> None:
    """
    Save a timeline + (optionally) heatmap for every sequence in df.

    Files are named {recording_id}_timeline.png and {recording_id}_heatmap.png.

    Parameters
    ----------
    id_to_label    : {class_index: label_string} — needed for heatmap y-axis
    output_dir     : directory to save plots into
    include_heatmap: set False to skip heatmap (faster, less disk space)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build shared color palette once across all sequences
    all_labels = set()
    for _, row in df.iterrows():
        seq = get_sequence_arrays(row, arrays)
        if seq["raw_labels"] is not None:
            all_labels.update(seq["raw_labels"].tolist())
        for seg in row.get("gt_segments", []):
            t = _safe_segment_tuple(seg)
            if t:
                all_labels.add(t[2])
    all_labels.add(background_label)
    label_colors = _build_label_colors(all_labels, background_label)

    n = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        rec_id = str(row.get("recording_id", f"seq_{i}")).replace("/", "_")
        print(f"[{i}/{n}] {rec_id}  WER={float(row['wer']):.3f}")

        plot_sequence_timeline(
            row, arrays,
            label_colors=label_colors,
            background_label=background_label,
            save_path=os.path.join(output_dir, f"{rec_id}_timeline.png"),
        )
        plt.close("all")

        if include_heatmap:
            plot_confidence_heatmap(
                row, arrays,
                id_to_label=id_to_label,
                background_label=background_label,
                save_path=os.path.join(output_dir, f"{rec_id}_heatmap.png"),
            )
            plt.close("all")

    print(f"\nDone. {n} sequences saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Example usage (run this cell after saving results with save_split_results)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Load ----
    test_df, test_arrays = load_split_results("test")
    val_df,  val_arrays  = load_split_results("val")

    # ---- Single sequence — timeline ----
    row = test_df.iloc[0]
    plot_sequence_timeline(row, test_arrays)

    # ---- Single sequence — heatmap ----
    # id_to_label must be the same dict used during training/inference
    # (you need to provide it from your training notebook)
    # plot_confidence_heatmap(row, test_arrays, id_to_label=id_to_label)

    # ---- WER distribution for test split ----
    plot_wer_distribution(test_df, split_name="Test")

    # ---- Mini-grid overview ----
    plot_split_overview(test_df, test_arrays, max_sequences=20)

    # ---- Confusion matrix ----
    plot_confusion_from_emissions(test_df)

    # ---- Save all sequences to disk ----
    # plot_all_sequences(
    #     test_df, test_arrays,
    #     id_to_label=id_to_label,
    #     output_dir="./plots/test",
    #     include_heatmap=True,
    # )