"""
metrics_corrected.py
=====================
Corrected implementation of all three SHREC'21 online gesture recognition
metrics, faithful to the official MATLAB evaluation script (evaluateShrec21.m).

  Caputo et al., "SHREC 2021: Skeleton-based Hand Gesture Recognition
  in the Wild", Computers & Graphics, 2021.
  https://arxiv.org/abs/2106.10980

Fixes applied vs. original Duo Streamers notebook
---------------------------------------------------
FIX 1 – Detection threshold: replaced IoU with overlap/GT_length > 0.5,
         matching MATLAB lines 73-76 exactly (no +1 in overlap).

FIX 2 – Matching loop structure: outer loop is over PREDICTIONS, inner loop
         is over GT gestures. A `found` flag lives on the GT side (not the
         prediction side), matching MATLAB exactly. A single prediction can
         overlap and be evaluated against every GT gesture. A prediction that
         overlaps nothing is counted as a False Positive.

FIX 3 – Misclassification tracked separately: a prediction that overlaps a
         GT window (overlap/GT_length > 0.5) but has the wrong label is a
         Misclassification, not a False Positive. Matches MATLAB line 103.

FIX 4 – Jaccard is interval-level (not frame-level): for every pred-GT pair
         of the same class with overlap > 0, compute overlap/union and
         accumulate the sum. A separate jaccardCounts counter tracks how many
         pairs contributed. Matches MATLAB lines 79-83.

FIX 5 – Jaccard denominator: jaccardCounts + Missed + Misclassified + FP,
         matching MATLAB line 141 exactly.

FIX 6 – FPR denominator: total GT gestures per class (TP+FN), matching the
         SHREC'21 paper definition.

FIX 7 – FP class attribution: attributed to the predicted class (correct
         behaviour). Note: the official MATLAB script has a bug here (it uses
         the last GT class instead of the predicted class); we intentionally
         do NOT replicate that bug.

SHREC'21 metric definitions (from paper)
-----------------------------------------
  Detection Rate      = TP / (TP + FN)
  False Positive Rate = FP / (TP + FN)   [denominator = total GT gestures]
  Misclassification   = Misclassified / (TP + FN)
  Jaccard Index       = mean over classes of:
                          sum(overlap/union for matching pairs)
                          / (jaccardCounts + Missed + Misclassified + FP)

Usage
-----
    from metrics_corrected import initialize_globals, evaluate_all, print_global_results

    initialize_globals(n_classes=20)   # gesture classes only, exclude background

    for sequence in test_set:
        frame_sequence, y_true, y_pred_list, gating_list = run_model(sequence)
        evaluate_all(frame_sequence, y_true, gating_list, y_pred_list)

    print_global_results(class_names=YOUR_CLASS_LIST)

Inputs
------
  frame_sequence : array-like, shape (2 * num_gestures,)
                   Alternating [start_0, end_0, start_1, end_1, ...] frame
                   indices (0-based) for each ground-truth gesture instance.

  y_true         : list of int, length num_gestures
                   Ground-truth class index for each gesture instance.

  gating_list    : list of int, length 2 * num_predicted_gestures
                   Alternating [start_0, end_0, start_1, end_1, ...] frame
                   indices for each predicted gesture instance.

  y_pred_list    : list/array of int, length = total frames in sequence
                   Per-frame class predictions. Use -1 or num_classes for
                   background frames.

  seq_len        : int (optional) — total frames, defaults to len(y_pred_list)
"""

# ---------------------------------------------------------------------------
# Global accumulators
# ---------------------------------------------------------------------------
num_classes = 17

_global_total_gestures      = None   # GT count per class
_global_correct_predictions = None   # TP per class
_global_missed              = None   # GT gestures never overlapped by any pred
_global_misclassified       = None   # overlapped but wrong label
_global_false_positives     = None   # predictions that overlapped no GT
_global_latencies           = None   # latency samples per class
_global_jaccard_sum         = None   # running sum of overlap/union scores
_global_jaccard_counts      = None   # how many pairs contributed to jaccard_sum


def initialize_globals(n_classes=17):
    """Reset all global accumulators. Call once before the evaluation loop."""
    global num_classes
    global _global_total_gestures, _global_correct_predictions
    global _global_missed, _global_misclassified, _global_false_positives
    global _global_latencies, _global_jaccard_sum, _global_jaccard_counts

    num_classes                 = n_classes
    _global_total_gestures      = [0]   * num_classes
    _global_correct_predictions = [0]   * num_classes
    _global_missed              = [0]   * num_classes
    _global_misclassified       = [0]   * num_classes
    _global_false_positives     = [0]   * num_classes
    _global_latencies           = [[]   for _ in range(num_classes)]
    _global_jaccard_sum         = [0.0] * num_classes
    _global_jaccard_counts      = [0]   * num_classes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_intervals(flat_list):
    """Convert [s0, e0, s1, e1, ...] into [(s0,e0), (s1,e1), ...]."""
    n = len(flat_list) // 2
    return [(int(flat_list[2*i]), int(flat_list[2*i+1])) for i in range(n)]


def _overlap_ratio(s_true, e_true, s_pred, e_pred):
    """
    FIX 1 — overlap / GT_length, no +1, matching MATLAB lines 73-76.
    A value > 0.5 means the prediction sufficiently covers the GT gesture.
    """
    overlap   = min(e_true, e_pred) - max(s_true, s_pred)   # no +1, matches MATLAB
    gt_length = e_true - s_true
    if gt_length <= 0:
        gt_length = 1
    return overlap / gt_length


def _majority_class(y_pred_list, s, e):
    """
    Most frequent non-background class in y_pred_list[s:e+1].
    Both -1 and num_classes are treated as background and excluded.
    Returns None if every frame is background.
    """
    segment = [
        y_pred_list[i]
        for i in range(s, min(e + 1, len(y_pred_list)))
        if y_pred_list[i] not in (-1, num_classes)
    ]
    if not segment:
        return None
    return max(set(segment), key=segment.count)


def _build_pred_intervals(gating_list, y_pred_list):
    """
    Parse gating_list into a list of dicts: {start, end, class}.
    Class is determined by majority vote of per-frame predictions.
    """
    intervals = []
    for (s, e) in _parse_intervals(gating_list):
        cls = _majority_class(y_pred_list, s, e)
        intervals.append({'start': s, 'end': e, 'class': cls})
    return intervals


# ---------------------------------------------------------------------------
# Core metric computation — matches MATLAB loop structure exactly
# ---------------------------------------------------------------------------

def _compute_metrics(gt_intervals, gt_classes, pred_intervals, y_pred_list, n_classes):
    """
    Compute detection rate, misclassification, FPR, latency, and Jaccard
    in one pass, mirroring the MATLAB script structure:

      outer loop  → predictions  (MATLAB: for r = 1:3:size(R,2))
      inner loop  → GT gestures  (MATLAB: for a = 1:3:size(A,2))
      found[]     → flag on GT side, prevents double-counting a GT as TP

    FIX 2 — loop structure matches MATLAB.
    FIX 3 — misclassification is separate from FP.
    FIX 4 — interval-level Jaccard accumulated here.

    Returns
    -------
    total_gestures      : list[int]
    correct_predictions : list[int]
    missed              : list[int]
    misclassified       : list[int]
    false_positives     : list[int]
    latencies           : list[list[int]]
    jaccard_sum         : list[float]
    jaccard_counts      : list[int]
    """
    total_gestures      = [0]   * n_classes
    correct_predictions = [0]   * n_classes
    missed              = [0]   * n_classes
    misclassified       = [0]   * n_classes
    false_positives     = [0]   * n_classes
    latencies           = [[]   for _ in range(n_classes)]
    jaccard_sum         = [0.0] * n_classes
    jaccard_counts      = [0]   * n_classes

    num_gt = len(gt_intervals)

    # Count total GT gestures per class
    for i in range(num_gt):
        total_gestures[int(gt_classes[i])] += 1

    # found[i] = 1 if GT gesture i has already been matched to a TP prediction
    # This is on the GT side, matching MATLAB's `found` array (line 60, 94)
    found = [0] * num_gt

    # --- FIX 2: outer loop over predictions ---
    for pred in pred_intervals:
        s_pred    = pred['start']
        e_pred    = pred['end']
        pred_cls  = pred['class']
        detected  = False   # did this prediction overlap any GT? (MATLAB: detected=false)

        # --- inner loop over GT gestures ---
        for i, (s_true, e_true) in enumerate(gt_intervals):
            true_class = int(gt_classes[i])

            # FIX 4 — Jaccard: accumulate for every overlapping same-class pair
            overlap = min(e_true, e_pred) - max(s_true, s_pred)   # no +1
            if overlap > 0 and pred_cls == true_class:
                u = max(e_true, e_pred) - min(s_true, s_pred)     # no +1
                if u > 0:
                    jaccard_sum[true_class]    += overlap / u
                    jaccard_counts[true_class] += 1

            # FIX 1 — detection threshold: overlap/GT_length > 0.5
            ratio = _overlap_ratio(s_true, e_true, s_pred, e_pred)
            if ratio > 0.5:
                detected = True
                if pred_cls == true_class:
                    # FIX 3 — correct detection, but only count GT once (MATLAB line 90-95)
                    if found[i] == 0:
                        found[i] = 1
                        correct_predictions[true_class] += 1

                        # Latency: first frame >= s_pred where model predicted true_class
                        first_correct = s_pred
                        for f in range(s_pred, e_pred + 1):
                            if f < len(y_pred_list) and y_pred_list[f] == true_class:
                                first_correct = f
                                break
                        latencies[true_class].append(max(0, first_correct - s_true))
                else:
                    # FIX 3 — wrong label, overlaps GT → misclassification on GT class
                    # (MATLAB line 103: classResults(GT_class, 4) += 1)
                    misclassified[true_class] += 1

        # FIX 7 — FP attributed to predicted class (intentionally better than MATLAB bug)
        if not detected:
            if pred_cls is not None and pred_cls < n_classes:
                false_positives[pred_cls] += 1

    # Missed: GT gestures never matched by any prediction (MATLAB lines 130-139)
    for i in range(num_gt):
        if found[i] == 0:
            missed[int(gt_classes[i])] += 1

    return (total_gestures, correct_predictions, missed,
            misclassified, false_positives, latencies,
            jaccard_sum, jaccard_counts)


# ---------------------------------------------------------------------------
# Main evaluation function — call once per sequence
# ---------------------------------------------------------------------------

def evaluate_all(frame_sequence, y_true, gating_list, y_pred_list,
                 seq_len=None, n_classes=None, verbose=True):
    """
    Run all metrics for one sequence and update global accumulators.

    Parameters
    ----------
    frame_sequence : array-like [s0,e0,s1,e1,...] — GT boundaries
    y_true         : list[int] — GT class per gesture
    gating_list    : list[int] [s0,e0,s1,e1,...] — predicted boundaries
    y_pred_list    : list[int] — per-frame predictions (-1 = background)
    seq_len        : int — total frames (defaults to len(y_pred_list))
    n_classes      : int — override global num_classes
    verbose        : bool — print per-sequence results
    """
    global _global_total_gestures, _global_correct_predictions
    global _global_missed, _global_misclassified, _global_false_positives
    global _global_latencies, _global_jaccard_sum, _global_jaccard_counts

    nc = n_classes or num_classes

    if seq_len is None:
        seq_len = len(y_pred_list)

    gt_intervals   = _parse_intervals(frame_sequence)
    pred_intervals = _build_pred_intervals(gating_list, y_pred_list)

    (total_gest, correct_pred, missed, misclassified,
     false_positives, latencies,
     jaccard_sum, jaccard_counts) = _compute_metrics(
        gt_intervals, y_true, pred_intervals, y_pred_list, nc)

    # Accumulate into globals
    for i in range(nc):
        _global_total_gestures[i]      += total_gest[i]
        _global_correct_predictions[i] += correct_pred[i]
        _global_missed[i]              += missed[i]
        _global_misclassified[i]       += misclassified[i]
        _global_false_positives[i]     += false_positives[i]
        _global_latencies[i].extend(latencies[i])
        _global_jaccard_sum[i]         += jaccard_sum[i]
        _global_jaccard_counts[i]      += jaccard_counts[i]

    if verbose:
        print("Results for the current sequence:")
        for i in range(nc):
            if total_gest[i] > 0 or false_positives[i] > 0:
                dr  = correct_pred[i] / total_gest[i] if total_gest[i] > 0 else 0.0
                mr  = misclassified[i] / total_gest[i] if total_gest[i] > 0 else 0.0
                fpr = false_positives[i] / total_gest[i] if total_gest[i] > 0 else 0.0
                avg_lat = (sum(latencies[i]) / len(latencies[i])
                           if latencies[i] else float('nan'))
                # FIX 5 — Jaccard denominator: jaccardCounts + Missed + Misclassified + FP
                jac_denom = jaccard_counts[i] + missed[i] + misclassified[i] + false_positives[i]
                avg_jac   = jaccard_sum[i] / jac_denom if jac_denom > 0 else 0.0
                print(f"  Class {i}: DR={dr:.2f}  MR={mr:.2f}  FPR={fpr:.2f}  "
                      f"Latency={avg_lat:.1f}fr  Jaccard={avg_jac:.3f}")
        print()


# ---------------------------------------------------------------------------
# Print global results — call after all sequences are processed
# ---------------------------------------------------------------------------

def print_global_results(class_names=None):
    """
    Print per-class and macro-averaged results matching the SHREC'21 protocol.

    Reported metrics (matching MATLAB output):
      Detection Rate      = TP / total_GT                (MATLAB: correctScore)
      Misclassification   = Misclassified / total_GT     (MATLAB: misclassifiedRate)
      False Positive Rate = FP / total_GT                (MATLAB: falsePositiveRate)
      Jaccard Index       = jaccard_sum /
                            (jaccardCounts + Missed + Misclassified + FP)
                                                         (MATLAB: line 141)
    """
    nc = num_classes
    print("=" * 60)
    print("GLOBAL RESULTS  (Corrected — SHREC'21 protocol)")
    print("=" * 60)

    total_tp   = 0
    total_gt   = 0
    total_fp   = 0
    total_misc = 0
    all_jac    = []

    for i in range(nc):
        gt   = _global_total_gestures[i]
        tp   = _global_correct_predictions[i]
        miss = _global_missed[i]
        misc = _global_misclassified[i]
        fp   = _global_false_positives[i]
        j_sum  = _global_jaccard_sum[i]
        j_cnt  = _global_jaccard_counts[i]

        if gt == 0 and fp == 0:
            continue

        name    = class_names[i] if class_names else str(i)
        dr      = tp   / gt if gt > 0 else 0.0
        mr      = misc / gt if gt > 0 else 0.0   # FIX 6
        fpr     = fp   / gt if gt > 0 else 0.0   # FIX 6
        avg_lat = (sum(_global_latencies[i]) / len(_global_latencies[i])
                   if _global_latencies[i] else float('nan'))

        # FIX 5 — Jaccard denominator matches MATLAB line 141
        jac_denom = j_cnt + miss + misc + fp
        avg_jac   = j_sum / jac_denom if jac_denom > 0 else 0.0

        print(f"Class {name}:")
        print(f"  Detection Rate : {dr:.4f}  ({tp}/{gt})")
        print(f"  Misclass Rate  : {mr:.4f}  ({misc}/{gt})")
        print(f"  False Pos Rate : {fpr:.4f}  ({fp} FP / {gt} GT)")
        print(f"  Avg Latency    : {avg_lat:.2f} frames")
        print(f"  Avg Jaccard    : {avg_jac:.4f}")

        total_tp   += tp
        total_gt   += gt
        total_fp   += fp
        total_misc += misc
        all_jac.append(avg_jac)

    print()
    print("--- Macro averages (across all classes) ---")
    macro_dr   = total_tp   / total_gt if total_gt > 0 else 0.0
    macro_mr   = total_misc / total_gt if total_gt > 0 else 0.0
    macro_fpr  = total_fp   / total_gt if total_gt > 0 else 0.0
    macro_jac  = sum(all_jac) / len(all_jac) if all_jac else float('nan')
    print(f"  Detection Rate : {macro_dr:.4f}")
    print(f"  Misclass Rate  : {macro_mr:.4f}")
    print(f"  False Pos Rate : {macro_fpr:.4f}")
    print(f"  Jaccard Index  : {macro_jac:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Sanity check — run directly: python metrics_corrected.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(42)

    N_CLASSES = 5
    SEQ_LEN   = 200

    # GT: 3 gestures
    frame_seq  = [10, 40, 70, 100, 130, 160]
    y_true_seq = [0, 1, 2]

    # Per-frame predictions
    y_pred = [-1] * SEQ_LEN
    for f in range(12, 38):   y_pred[f] = 0   # class 0, slightly offset  → TP
    for f in range(75, 98):   y_pred[f] = 1   # class 1                   → TP
    for f in range(140, 165): y_pred[f] = 3   # wrong class for gesture 2 → misclassification

    # Gating (detector output)
    gating = [11, 38, 74, 99, 139, 166]

    initialize_globals(n_classes=N_CLASSES)
    evaluate_all(frame_seq, y_true_seq, gating, y_pred, seq_len=SEQ_LEN, verbose=True)
    print_global_results(class_names=["A", "B", "C", "D", "E"])