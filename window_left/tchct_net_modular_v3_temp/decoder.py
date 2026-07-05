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

import config
from config import (
    BAG_AGGREGATION,
    BAG_SIZE,
    CONFIDENCE_THRESHOLD,
    DEVICE,
    MIN_SIGN_FRAMES,
    SIGN_BG_MARGIN,
    ONLINE_WINDOW_SIZE,
    ONLINE_STRIDE,
    ENERGY_CONF_THRESH,
    ENERGY_GRAB_WEIGHT,
    ENERGY_PINCH_WEIGHT,
    BG_MARGIN_RESCUE_EPS,
    HIST_N_BINS,
    HIST_RESAMPLE_STEPS,
    HIST_MOTION_EPS,
    DISAMBIG_TAU_MARGIN,
    DISAMBIG_LAMBDA,
    DISAMBIG_TOP_K,
)
import disambiguation

# NOTE: USE_DISAMBIGUATION is intentionally read as `config.USE_DISAMBIGUATION`
# everywhere below (never imported by name) so that callers can flip it at
# runtime (e.g. `import config; config.USE_DISAMBIGUATION = True`) and have
# it take effect immediately — a plain `from config import USE_DISAMBIGUATION`
# would bind a stale copy at import time that later mutation can't reach.


# ---------------------------------------------------------------------------
# Bag aggregator — unchanged from notebook
# ---------------------------------------------------------------------------
# ... [Keeping class _BagAggregator and SimplifiedBagDecoder unchanged] ...
# [Just replace the stream_model_online function at the bottom]



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
        theta_high: float | None    = None,
        templates: np.ndarray | None = None,
        class_order: list[str] | None = None,
        bg_margin_rescue_eps: float = BG_MARGIN_RESCUE_EPS,
        tau_margin: float           = DISAMBIG_TAU_MARGIN,
        lam: float                  = DISAMBIG_LAMBDA,
        top_k: int                  = DISAMBIG_TOP_K,
        n_bins: int                 = HIST_N_BINS,
        resample_steps: int         = HIST_RESAMPLE_STEPS,
        motion_eps: float           = HIST_MOTION_EPS,
    ):
        self.id_to_label          = id_to_label
        self.background_label     = background_label
        self.confidence_threshold = float(confidence_threshold)
        self.sign_bg_margin       = float(sign_bg_margin)
        self.min_sign_frames      = max(1, int(min_sign_frames))

        # Disambiguation (Hook A / Hook B) — all optional, no-op if None.
        # Passed explicitly (not read from config at call time) so per-fold
        # calibration can override them without mutating global config state.
        self.theta_high  = theta_high
        self.templates   = templates
        self.class_order = class_order
        self.bg_margin_rescue_eps = float(bg_margin_rescue_eps)
        self.tau_margin           = float(tau_margin)
        self.lam                  = float(lam)
        self.top_k                = int(top_k)
        self.n_bins               = int(n_bins)
        self.resample_steps       = int(resample_steps)
        self.motion_eps           = float(motion_eps)

        self.background_id = next(
            (k for k, v in id_to_label.items() if v == background_label), None
        )

        self._bag = _BagAggregator(bag_size, aggregation, len(id_to_label))

        # Hysteresis state
        self.state              = "SEEKING"
        self.region_votes       = Counter()
        self.region_probs       : list[np.ndarray] = []
        self.region_raw_aux     : list[np.ndarray] = []
        self.sign_frames        = 0
        self.region_start_frame = None      # frame where current IN_SIGN region began

        # Diagnostics (only meaningful when USE_DISAMBIGUATION is True)
        self.hook_a_rescue_count      = 0
        self.hook_b_triggered_count   = 0
        self.hook_b_total_emissions   = 0
        self.hook_b_label_changed_count = 0

    # ------------------------------------------------------------------

    def _gate(self, agg_probs: np.ndarray, energy_now: float | None = None):
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

        # Hook A — only ever rescues a narrow near-miss; never overrides a
        # clear rejection or a clear acceptance.
        if (
            config.USE_DISAMBIGUATION
            and not is_sign
            and pred_label != self.background_label
            and energy_now is not None
        ):
            margin = pred_conf - bg_conf
            is_near_miss = (
                pred_conf >= self.confidence_threshold
                and (self.sign_bg_margin - margin) <= self.bg_margin_rescue_eps
            )
            rescued = disambiguation.disambiguate_background(
                is_near_miss=is_near_miss,
                energy_now=energy_now,
                theta_high=self.theta_high,
            )
            if rescued:
                is_sign = True
                self.hook_a_rescue_count += 1

        voted_label   = pred_label if is_sign else self.background_label
        is_background = not is_sign

        return voted_label, is_background, pred_label, pred_conf, bg_conf, agg_probs

    # ------------------------------------------------------------------

    def _finalize_region(self) -> str:
        """
        Compute the emitted label for the region that's about to close,
        combining the plain majority vote with Hook B (sign-vs-sign
        refinement) when enabled and applicable. Does not reset state —
        caller resets region_votes/region_probs/region_raw_aux/sign_frames.
        """
        majority_label = self.region_votes.most_common(1)[0][0]

        if not config.USE_DISAMBIGUATION or self.templates is None or self.class_order is None:
            return majority_label

        self.hook_b_total_emissions += 1

        region_probs   = np.stack(self.region_probs, axis=0)
        region_raw_aux = np.stack(self.region_raw_aux, axis=0) if self.region_raw_aux else None
        if region_raw_aux is None:
            return majority_label

        final_label, triggered = disambiguation.disambiguate_region_label(
            region_probs=region_probs,
            region_raw_aux=region_raw_aux,
            templates=self.templates,
            class_order=self.class_order,
            background_idx=self.background_id,
            fallback_label=majority_label,
            tau_margin=self.tau_margin,
            lam=self.lam,
            top_k=self.top_k,
            n_bins=self.n_bins,
            resample_steps=self.resample_steps,
            motion_eps=self.motion_eps,
        )

        if triggered:
            self.hook_b_triggered_count += 1
            if final_label != majority_label:
                self.hook_b_label_changed_count += 1

        return final_label

    def get_diagnostics(self) -> dict:
        return {
            "hook_a_rescue_count":        self.hook_a_rescue_count,
            "hook_b_triggered_count":     self.hook_b_triggered_count,
            "hook_b_total_emissions":     self.hook_b_total_emissions,
            "hook_b_label_changed_count": self.hook_b_label_changed_count,
        }

    # ------------------------------------------------------------------

    def update(self, logits: np.ndarray, frame_index: int, energy_now: float | None = None,
               aux_row: np.ndarray | None = None) -> dict:
        """
        Process one frame.

        Parameters
        ----------
        logits      : (C,) raw logits from model — stored as pre_bag_logits
        frame_index : int current frame index, needed for emit_region tracking
        energy_now  : optional Hook A motion-energy scalar for this frame
                      (None disables Hook A regardless of USE_DISAMBIGUATION)
        aux_row     : optional (12,) raw aux sensor row for this frame,
                      buffered across an IN_SIGN region for Hook B

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
            self._gate(agg_probs, energy_now=energy_now)

        emitted_label = None
        emit_region   = None

        if self.state == "SEEKING":
            if not is_background:
                self.state              = "IN_SIGN"
                self.region_votes[voted_label] += 1
                self.region_probs       = [agg_probs]
                self.region_raw_aux     = [aux_row] if aux_row is not None else []
                self.sign_frames        = 1
                self.region_start_frame = frame_index   # record region start

        elif self.state == "IN_SIGN":
            if not is_background:
                self.region_votes[voted_label] += 1
                self.region_probs.append(agg_probs)
                if aux_row is not None:
                    self.region_raw_aux.append(aux_row)
                self.sign_frames += 1
            else:
                # Bag confirmed background — trailing edge reached
                if self.sign_frames >= self.min_sign_frames:
                    emitted_label = self._finalize_region()
                    emit_region   = (
                        self.region_start_frame,    # start of region
                        frame_index,                # end of region (trailing edge)
                        emitted_label,
                    )
                # else: region too short → discard silently

                self.state              = "SEEKING"
                self.region_votes       = Counter()
                self.region_probs       = []
                self.region_raw_aux     = []
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
            emitted     = self._finalize_region()
            emit_region = (
                self.region_start_frame,
                None,       # end unknown — caller fills with t_len - 1
                emitted,
            )

        # Always reset — decoder is invalid after flush
        self.state              = "SEEKING"
        self.region_votes       = Counter()
        self.region_probs       = []
        self.region_raw_aux     = []
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
    window_size: int            = ONLINE_WINDOW_SIZE,
    stride: int                 = ONLINE_STRIDE,
    bag_size: int               = BAG_SIZE,
    aggregation: str            = BAG_AGGREGATION,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    sign_bg_margin: float       = SIGN_BG_MARGIN,
    min_sign_frames: int        = MIN_SIGN_FRAMES,
    V_aux: np.ndarray | None    = None,
    theta_high: float | None    = None,
    templates: np.ndarray | None = None,
    class_order: list[str] | None = None,
    energy_conf_thresh: float   = ENERGY_CONF_THRESH,
    energy_grab_weight: float   = ENERGY_GRAB_WEIGHT,
    energy_pinch_weight: float  = ENERGY_PINCH_WEIGHT,
    bg_margin_rescue_eps: float = BG_MARGIN_RESCUE_EPS,
    tau_margin: float           = DISAMBIG_TAU_MARGIN,
    lam: float                  = DISAMBIG_LAMBDA,
    top_k: int                  = DISAMBIG_TOP_K,
    n_bins: int                 = HIST_N_BINS,
    resample_steps: int         = HIST_RESAMPLE_STEPS,
    motion_eps: float           = HIST_MOTION_EPS,
) -> tuple[list[dict], list[str], list[tuple], dict]:
    """
    Sliding-window based streaming inference with bag-aggregated decoder.

    Parameters
    ----------
    V            : (T, D) input feature array
    model_obj    : THCT-Net model
    normalize_fn : per-window normalization callable
    window_size  : sliding window size
    stride       : sliding window stride
    V_aux        : optional (T, 12) raw aux sensor array (disambiguation.RAW_AUX_KEYS
                   order) — enables Hook A/B when provided alongside theta_high
                   / templates. None disables disambiguation regardless of the
                   USE_DISAMBIGUATION flag.
    theta_high   : per-fold calibrated Hook A rescue threshold
    templates    : (num_sign_classes, 2*HIST_N_BINS) Hook B template bank
    class_order  : label order matching `templates` rows
    energy_conf_thresh, energy_grab_weight, energy_pinch_weight : Hook A energy knobs
    bg_margin_rescue_eps : Hook A near-miss window
    tau_margin, lam, top_k, n_bins, resample_steps, motion_eps  : Hook B knobs
                   (all default to config.py, but can be overridden per-call so
                   calibration sweeps don't need to mutate global config state)

    Returns
    -------
    stream_steps  : list of per-frame dicts
    emitted_preds : ordered list of emitted sign labels
    emit_regions  : list of (start_frame, end_frame, label) — one per emission
    diagnostics   : dict of Hook A/B counters (see SimplifiedBagDecoder.get_diagnostics)
    """
    if V.ndim != 2:
        raise ValueError(f"Expected (T, D), got {V.shape}")

    t_len = V.shape[0]
    if t_len == 0:
        return [], [], [], {
            "hook_a_rescue_count": 0, "hook_b_triggered_count": 0,
            "hook_b_total_emissions": 0, "hook_b_label_changed_count": 0,
        }

    decoder = SimplifiedBagDecoder(
        id_to_label=id_to_label,
        background_label=background_label,
        bag_size=bag_size,
        aggregation=aggregation,
        confidence_threshold=confidence_threshold,
        sign_bg_margin=sign_bg_margin,
        min_sign_frames=min_sign_frames,
        theta_high=theta_high,
        templates=templates,
        class_order=class_order,
        bg_margin_rescue_eps=bg_margin_rescue_eps,
        tau_margin=tau_margin,
        lam=lam,
        top_k=top_k,
        n_bins=n_bins,
        resample_steps=resample_steps,
        motion_eps=motion_eps,
    )

    model_obj.eval()
    stream_steps   : list[dict]  = []
    emitted_preds  : list[str]   = []
    emit_regions   : list[tuple] = []

    frame_buffer = deque(maxlen=window_size)

    for frame_idx in range(t_len):
        frame_buffer.append(V[frame_idx].astype(np.float32, copy=False))

        # Wait until window is full
        if len(frame_buffer) < window_size:
            continue

        # Respect stride
        if ((frame_idx - (window_size - 1)) % stride) != 0:
            continue

        window   = np.stack(frame_buffer, axis=0).astype(np.float32, copy=False)
        norm_win = normalize_fn(window)  # (W, D)

        # Forward pass:
        # sequences: (1, W, D)
        # lengths: (1,)
        x = torch.tensor(norm_win, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        lengths = torch.tensor([window_size], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            logits = model_obj(x, lengths)
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            logits_np = logits[0].cpu().numpy().astype(np.float32)  # (C,)

        energy_now = None
        aux_row    = None
        if V_aux is not None:
            aux_row = V_aux[frame_idx]
            prev_aux_row = V_aux[frame_idx - 1] if frame_idx > 0 else None
            energy_now = disambiguation.compute_motion_energy(
                aux_row, prev_aux_row,
                conf_thresh=energy_conf_thresh,
                grab_w=energy_grab_weight,
                pinch_w=energy_pinch_weight,
            )

        decoded                = decoder.update(
            logits_np, frame_index=frame_idx, energy_now=energy_now, aux_row=aux_row,
        )
        decoded["frame_index"] = int(frame_idx)
        decoded["window_start"] = int(frame_idx - window_size + 1)
        decoded["window_end"]   = int(frame_idx)
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
        
        last = stream_steps[-1] if stream_steps else {}
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
            "window_start":   last.get("window_start", t_len - window_size),
            "window_end":     t_len - 1,
        })

    return stream_steps, emitted_preds, emit_regions, decoder.get_diagnostics()
