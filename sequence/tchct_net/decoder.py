"""
Online streaming decoder for causal sign-language recognition.

This module implements the bag-aggregated hysteresis decoder used for
frame-by-frame inference with the THCT-Net model.

**CAUTION**: The logic in _BagAggregator, SimplifiedBagDecoder, and
stream_model_online is the core inference engine. Any changes here will
directly affect WER results. Modify with extreme care.

The THCT-Net model is strictly causal: output at frame t depends only on
frames 0…t.  Therefore we run a single batch forward pass over the full
sequence and then iterate over the resulting per-frame logits, feeding
them one at a time to the bag-aggregated decoder — the same decoder state
machine used by the LSTM variant.

Provides:
  - _BagAggregator         : causal sliding bag over raw logits
  - SimplifiedBagDecoder   : hysteresis state machine (SEEKING ↔ IN_SIGN)
  - stream_model_online    : batch-mode THCT-Net streaming inference
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np
import torch

from config import (
    BAG_AGGREGATION,
    BAG_SIZE,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    MIN_SIGN_FRAMES,
    SIGN_BG_MARGIN,
)


# ---------------------------------------------------------------------------
# Bag aggregator — unchanged from notebook
# ---------------------------------------------------------------------------

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

    def update(self, logits: np.ndarray) -> np.ndarray | None:
        """
        Push one logit vector and return aggregated probs.

        Returns None until bag is full (first bag_size frames are skipped).
        """
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


# ---------------------------------------------------------------------------
# Simplified streaming decoder — unchanged from notebook
# ---------------------------------------------------------------------------

class SimplifiedBagDecoder:
    """
    Causal streaming decoder using bag-aggregated logits.

    States
    ------
    SEEKING : waiting for a sign to begin
    IN_SIGN : inside an active sign region, accumulating votes

    Emission
    --------
    Fires at the TRAILING edge when the bag transitions to background.
    Emits the majority label observed across the entire region.
    Discards regions shorter than min_sign_frames (noise / glitches).

    Storage additions vs original
    -----------------------------
    pre_bag_logits     : raw logits from model before bag, stored per frame
    post_bag_probs     : aggregated probability vector after bag, stored per frame
                         None for first (bag_size - 1) frames until bag is full
    emit_region        : (start_frame, end_frame, label) on emission, else None
    region_start_frame : frame index where current IN_SIGN region began
    """

    def __init__(
        self,
        id_to_label: dict[int, str],
        background_label: str,
        bag_size: int               = BAG_SIZE,
        aggregation: str            = BAG_AGGREGATION,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        sign_bg_margin: float       = SIGN_BG_MARGIN,
        min_sign_frames: int        = MIN_SIGN_FRAMES,
    ):
        self.id_to_label          = id_to_label
        self.background_label     = background_label
        self.confidence_threshold = float(confidence_threshold)
        self.sign_bg_margin       = float(sign_bg_margin)
        self.min_sign_frames      = max(1, int(min_sign_frames))

        self.background_id = next(
            (k for k, v in id_to_label.items() if v == background_label), None
        )

        self._bag = _BagAggregator(bag_size, aggregation, len(id_to_label))

        # Hysteresis state
        self.state              = "SEEKING"
        self.region_votes       = Counter()
        self.sign_frames        = 0
        self.region_start_frame = None      # frame where current IN_SIGN region began

    # ------------------------------------------------------------------

    def _gate(self, agg_probs: np.ndarray):
        """
        Apply confidence gate to aggregated probabilities.

        Returns
        -------
        voted_label, is_background, pred_label, pred_conf, bg_conf, agg_probs
        agg_probs passed through so caller can store it as post_bag_probs
        without recomputing.
        """
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
        logits      : (C,) raw logits from model — stored as pre_bag_logits
        frame_index : int current frame index, needed for emit_region tracking

        Returns
        -------
        dict containing:
            raw_label      : top-1 label from pre-bag logits
            raw_conf       : top-1 confidence from pre-bag logits
            bg_conf        : background confidence from post-bag probs (0 until bag full)
            gated_label    : label after confidence gate (post-bag)
            voted_label    : same as gated_label
            state          : decoder state after this step (SEEKING / IN_SIGN)
            emitted_label  : emitted sign label if trailing edge fired, else None
            emit_region    : (start_frame, end_frame, label) on emission, else None
            pre_bag_logits : (C,) raw logits before bag — for visualization
            post_bag_probs : (C,) aggregated probs after bag — for visualization
                             None for first (bag_size - 1) frames
        """
        pre_bag_logits = logits.copy()              # capture before bag sees it
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
                "pre_bag_logits": pre_bag_logits,   # (C,) always stored
                "post_bag_probs": None,             # bag not full yet
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
                self.region_start_frame = frame_index   # record region start

        elif self.state == "IN_SIGN":
            if not is_background:
                self.region_votes[voted_label] += 1
                self.sign_frames += 1
            else:
                # Bag confirmed background — trailing edge reached
                if self.sign_frames >= self.min_sign_frames:
                    emitted_label = self.region_votes.most_common(1)[0][0]
                    emit_region   = (
                        self.region_start_frame,    # start of region
                        frame_index,                # end of region (trailing edge)
                        emitted_label,
                    )
                # else: region too short → discard silently

                self.state              = "SEEKING"
                self.region_votes       = Counter()
                self.sign_frames        = 0
                self.region_start_frame = None

        return {
            "raw_label":      pred_label,
            "raw_conf":       pred_conf,
            "bg_conf":        bg_conf,
            "gated_label":    voted_label,
            "voted_label":    voted_label,
            "state":          self.state,
            "emitted_label":  emitted_label,
            "emit_region":    emit_region,          # (start, end, label) or None
            "pre_bag_logits": pre_bag_logits,       # (C,) raw pre-bag logits
            "post_bag_probs": agg_probs,            # (C,) post-bag aggregated probs
        }

    # ------------------------------------------------------------------

    def flush(self) -> tuple[str | None, tuple | None]:
        """
        Call once after all frames are processed.

        Emits any sign region still open at sequence end.
        Necessary when a sequence ends without returning to background.

        Returns
        -------
        (emitted_label, emit_region)
        emit_region end frame is None — caller fills with t_len - 1.
        """
        emitted     = None
        emit_region = None

        if self.state == "IN_SIGN" and self.sign_frames >= self.min_sign_frames:
            emitted     = self.region_votes.most_common(1)[0][0]
            emit_region = (
                self.region_start_frame,
                None,       # end unknown — caller fills with t_len - 1
                emitted,
            )

        # Always reset — decoder is invalid after flush
        self.state              = "SEEKING"
        self.region_votes       = Counter()
        self.sign_frames        = 0
        self.region_start_frame = None
        self._bag.reset()

        return emitted, emit_region


# ---------------------------------------------------------------------------
# THCT-Net streaming inference (batch forward + per-frame decode)
# ---------------------------------------------------------------------------

def stream_model_online(
    V: np.ndarray,
    model_obj,
    normalize_fn,
    id_to_label: dict[int, str],
    background_label: str,
    bag_size: int               = BAG_SIZE,
    aggregation: str            = BAG_AGGREGATION,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    sign_bg_margin: float       = SIGN_BG_MARGIN,
    min_sign_frames: int        = MIN_SIGN_FRAMES,
) -> tuple[list[dict], list[str], list[tuple]]:
    """
    Batch-mode THCT-Net streaming inference with bag-aggregated decoder.

    Because THCT-Net is strictly causal (output at frame t depends only on
    frames 0…t), we can obtain identical per-frame logits by running a
    single forward pass over the entire sequence and then iterating over
    the per-frame logits, feeding them to the bag decoder one at a time.

    This is mathematically equivalent to running the model frame-by-frame
    (as the LSTM .step() variant does) but far more efficient.

    Parameters
    ----------
    V            : (T, D) input feature array
    model_obj    : THCT-Net model with forward(sequences, lengths) method
    normalize_fn : per-frame normalization callable

    Returns
    -------
    stream_steps  : list of per-frame dicts
    emitted_preds : ordered list of emitted sign labels
    emit_regions  : list of (start_frame, end_frame, label) — one per emission
    """
    if V.ndim != 2:
        raise ValueError(f"Expected (T, D), got {V.shape}")

    t_len = V.shape[0]
    if t_len == 0:
        return [], [], []

    decoder = SimplifiedBagDecoder(
        id_to_label=id_to_label,
        background_label=background_label,
        bag_size=bag_size,
        aggregation=aggregation,
        confidence_threshold=confidence_threshold,
        sign_bg_margin=sign_bg_margin,
        min_sign_frames=min_sign_frames,
    )

    model_obj.eval()
    stream_steps   : list[dict]  = []
    emitted_preds  : list[str]   = []
    emit_regions   : list[tuple] = []

    # ── Single batch forward pass ──
    # Normalize the full sequence, wrap in batch dimension, run model
    V_norm = normalize_fn(V.astype(np.float32, copy=False))    # (T, D)
    seq_tensor = torch.tensor(
        V_norm, dtype=torch.float32, device=DEVICE,
    ).unsqueeze(0)                                              # (1, T, D)
    lengths = torch.tensor([t_len], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits_batch = model_obj(seq_tensor, lengths)           # (1, T, C)

    all_logits = logits_batch[0].cpu().numpy().astype(np.float32)  # (T, C)

    # ── Per-frame decode using pre-computed logits ──
    for frame_idx in range(t_len):
        logits_np = all_logits[frame_idx]                       # (C,)

        decoded                = decoder.update(logits_np, frame_index=frame_idx)
        decoded["frame_index"] = int(frame_idx)
        stream_steps.append(decoded)

        if decoded["emitted_label"] is not None:
            emitted_preds.append(decoded["emitted_label"])
        if decoded["emit_region"] is not None:
            emit_regions.append(decoded["emit_region"])

    # Flush — emit any sign still open at sequence end
    final_emission, final_emit_region = decoder.flush()

    if final_emission is not None:
        # Fill the None end frame from flush with the last frame index
        if final_emit_region is not None:
            final_emit_region = (
                final_emit_region[0],
                t_len - 1,
                final_emit_region[2],
            )
        emitted_preds.append(final_emission)
        emit_regions.append(final_emit_region)
        stream_steps.append({
            "raw_label":      final_emission,
            "raw_conf":       1.0,
            "bg_conf":        0.0,
            "gated_label":    final_emission,
            "voted_label":    final_emission,
            "state":          "FLUSH",
            "emitted_label":  final_emission,
            "emit_region":    final_emit_region,    # (start, t_len-1, label)
            "pre_bag_logits": None,                 # no new frame at flush
            "post_bag_probs": None,
            "frame_index":    t_len - 1,
        })

    return stream_steps, emitted_preds, emit_regions
