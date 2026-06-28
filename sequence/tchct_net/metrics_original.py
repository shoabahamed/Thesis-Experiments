"""
metrics_original.py
====================
Faithful re-implementation of the metric functions from the Duo Streamers
notebook (DuoStreamers_Code_and_Experiments.ipynb).

BUGS ARE INTENTIONALLY PRESERVED so you can reproduce their numbers exactly
and use them as your baseline comparison.

Known issues (documented but NOT fixed here):
  1. IoU formula wraps abs() around the intersection — non-overlapping
     intervals are never zero, inflating detection rate.
  2. FPR denominator is (FP + total_gestures) instead of total_gestures.
  3. Jaccard global accumulation is per-class per-sequence, causing all
     entries to appear as "No data" when classes are missing in a sequence.
  4. evaluate_jaccard_index and evaluate_model_with_fpr are defined but
     commented-out in the notebook's main loop — only detection rate ran.

Usage
-----
Call initialize_globals() before your evaluation loop.
Then call one or more of the three evaluate_* functions per sequence.
At the end call print_global_results() to see the aggregated numbers.

Inputs expected (same format as Duo Streamers notebook):
  frame_sequence : 1-D array/tensor of alternating [start, end, start, end ...]
                   frame indices for each ground-truth gesture instance.
  y_true         : list of integer class indices, one per gesture instance.
  gating_list    : 1-D list of frame indices marking predicted gesture
                   boundaries [start, end, start, end ...].
  y_pred_list    : frame-level list of predicted class labels (-1 = background).
  num_classes    : number of gesture classes (default 17 for SHREC'21,
                   change to your class count).
"""

# ---------------------------------------------------------------------------
# Global accumulators  (call initialize_globals() before your loop)
# ---------------------------------------------------------------------------
num_classes = 17  # change to your number of sign classes

global_total_gestures = None
global_correct_predictions = None
global_latencies = None
global_jaccard_indices = None
global_false_positives = None


def initialize_globals(n_classes=17):
    """Reset all global accumulators. Call once before the evaluation loop."""
    global num_classes, global_total_gestures, global_correct_predictions
    global global_latencies, global_jaccard_indices, global_false_positives

    num_classes = n_classes
    global_total_gestures = [0] * num_classes
    global_correct_predictions = [0] * num_classes
    global_latencies = [[] for _ in range(num_classes)]
    global_jaccard_indices = [[] for _ in range(num_classes)]
    global_false_positives = [0] * num_classes


# ---------------------------------------------------------------------------
# Metric 1 – Detection Rate + Latency  (the only one actually called in
#             the notebook's main loop)
# ---------------------------------------------------------------------------

def evaluate_detection_rate(frame_sequence, y_true, gating_list, y_pred_list,
                             n_classes=None, verbose=True):
    """
    Original detection-rate function from the Duo Streamers notebook.

    BUG: IoU formula uses abs(), so non-overlapping intervals can still
    produce IoU > 0, inflating the detection rate.

    BUG: Boundary matching picks the single closest boundary point from the
    entire gating_list independently for start and end — may mix boundaries
    from two different predicted gestures.
    """
    global global_total_gestures, global_correct_predictions, global_latencies

    nc = n_classes or num_classes
    total_gestures = [0] * nc
    correct_predictions = [0] * nc
    latencies = [[] for _ in range(nc)]

    num_gestures = len(frame_sequence) // 2

    if not gating_list:
        print("[Warning] metrics_original/DR: 'gating_list' is empty. Model made no predictions for this sequence.")

    for i in range(num_gestures):
        s_true = int(frame_sequence[2 * i])
        e_true = int(frame_sequence[2 * i + 1])
        true_class = int(y_true[i])

        total_gestures[true_class] += 1

        if not gating_list:
            continue

        # ---- BUG: picks closest boundary point independently for s and e ----
        s_pred = min(gating_list, key=lambda x: abs(x - s_true))
        e_pred = min(gating_list, key=lambda x: abs(x - e_true))

        if s_pred > e_pred:
            s_pred, e_pred = e_pred, s_pred

        y_pred_interval = list(y_pred_list[s_pred:e_pred + 1])
        y_pred_interval = [label for label in y_pred_interval if label != -1]

        if not y_pred_interval:
            continue

        predicted_class = max(set(y_pred_interval), key=y_pred_interval.count)

        # ---- BUG: abs() means intersection is never truly 0 ----
        intersection = max(0, abs(min(e_true, e_pred) - max(s_true, s_pred)) + 1)
        union = max(e_true, e_pred) - min(s_true, s_pred) + 1
        iou = intersection / union

        if predicted_class == true_class and iou > 0.5:
            correct_predictions[true_class] += 1

            # Latency: first frame inside predicted interval that matches class
            first_correct_frame = s_pred
            for idx in range(s_pred, e_pred + 1):
                if idx < len(y_pred_list) and y_pred_list[idx] == predicted_class:
                    first_correct_frame = idx
                    break
            latency = first_correct_frame - s_true
            latencies[true_class].append(latency)

    # Update globals
    for i in range(nc):
        global_total_gestures[i] += total_gestures[i]
        global_correct_predictions[i] += correct_predictions[i]
        global_latencies[i].extend(latencies[i])

    if verbose:
        recalls = []
        avg_lats = []
        for i in range(nc):
            r = correct_predictions[i] / total_gestures[i] if total_gestures[i] > 0 else None
            l = sum(latencies[i]) / len(latencies[i]) if latencies[i] else None
            recalls.append(r)
            avg_lats.append(l)

        print("Results for the current sequence:")
        for i in range(nc):
            if total_gestures[i] > 0:
                print(f"  Class {i}: Recall={recalls[i]:.2f}",
                      f"  Avg Latency={'N/A' if avg_lats[i] is None else f'{avg_lats[i]:.2f} frames'}")
        print()


# ---------------------------------------------------------------------------
# Metric 2 – False Positive Rate  (defined but commented-out in notebook)
# ---------------------------------------------------------------------------

def evaluate_model_with_fpr(frame_sequence, y_true, gating_list, y_pred_list,
                             n_classes=None, verbose=True):
    """
    Original FPR function from the Duo Streamers notebook.

    BUG: FPR denominator is (FP + total_gestures) instead of just
    total_gestures, dampening the FPR when FP is large.

    BUG: Same independent boundary matching bug as detect-rate.
    """
    global global_total_gestures, global_false_positives

    nc = n_classes or num_classes
    total_gestures = [0] * nc
    false_positives = [0] * nc
    num_gestures = len(frame_sequence) // 2

    predicted_intervals = []
    num_preds = len(gating_list) // 2
    for i in range(num_preds):
        s_pred = gating_list[2 * i]
        e_pred = gating_list[2 * i + 1]
        predicted_intervals.append({'start': s_pred, 'end': e_pred, 'used': False})

    for i in range(num_gestures):
        s_true = int(frame_sequence[2 * i])
        e_true = int(frame_sequence[2 * i + 1])
        true_class = int(y_true[i])
        total_gestures[true_class] += 1

        all_boundaries = (
            [interval['start'] for interval in predicted_intervals if not interval['used']] +
            [interval['end']   for interval in predicted_intervals if not interval['used']]
        )
        if not all_boundaries:
            if i == 0:  # Only print once per sequence
                print("[Warning] metrics_original/FPR: No predicted boundaries available for FPR calculation.")
            continue

        s_pred = min(all_boundaries, key=lambda x: abs(x - s_true))
        e_pred = min(all_boundaries, key=lambda x: abs(x - e_true))
        if s_pred > e_pred:
            s_pred, e_pred = e_pred, s_pred

        used_intervals = []
        for idx, interval in enumerate(predicted_intervals):
            if interval['used']:
                continue
            if interval['end'] >= s_pred and interval['start'] <= e_pred:
                predicted_intervals[idx]['used'] = True
                used_intervals.append(interval)

        if not used_intervals:
            continue

        y_pred_interval = list(y_pred_list[s_pred:e_pred + 1])
        y_pred_interval = [label for label in y_pred_interval if label != -1]
        if not y_pred_interval:
            continue

        predicted_class = max(set(y_pred_interval), key=y_pred_interval.count)

        intersection = max(0, min(e_true, e_pred) - max(s_true, s_pred) + 1)
        union = max(e_true, e_pred) - min(s_true, s_pred) + 1
        iou = intersection / union

        if not (predicted_class == true_class and iou > 0.5):
            false_positives[predicted_class] += 1

    # Remaining unused intervals → false positives
    for interval in predicted_intervals:
        if interval['used']:
            continue
        y_pred_interval = list(y_pred_list[interval['start']:interval['end'] + 1])
        y_pred_interval = [label for label in y_pred_interval if label != -1]
        if not y_pred_interval:
            continue
        predicted_class = max(set(y_pred_interval), key=y_pred_interval.count)
        false_positives[predicted_class] += 1

    for i in range(nc):
        global_total_gestures[i] += total_gestures[i]
        global_false_positives[i] += false_positives[i]

    if verbose:
        print("Results for the current sequence (FPR):")
        for i in range(nc):
            if total_gestures[i] > 0 or false_positives[i] > 0:
                # ---- BUG: wrong denominator ----
                denom = false_positives[i] + total_gestures[i]
                fpr = false_positives[i] / denom if denom > 0 else None
                print(f"  Class {i}: FPR={'N/A' if fpr is None else f'{fpr:.2f}'}")
        print()


# ---------------------------------------------------------------------------
# Metric 3 – Jaccard Index  (defined but commented-out in notebook)
# ---------------------------------------------------------------------------

def evaluate_jaccard_index(frame_sequence, y_true, gating_list, y_pred_list,
                            n_classes=None, verbose=True):
    """
    Original Jaccard function from the Duo Streamers notebook.

    BUG: global accumulation skips classes not present in a sequence,
    so most entries remain empty and print as "No data".

    BUG: Same independent boundary matching.
    """
    global global_total_gestures, global_jaccard_indices

    nc = n_classes or num_classes
    total_gestures = [0] * nc
    jaccard_indices = [[] for _ in range(nc)]
    num_gestures = len(frame_sequence) // 2

    if not gating_list:
        print("[Warning] metrics_original/Jaccard: 'gating_list' is empty. Model made no predictions for this sequence.")

    for i in range(num_gestures):
        s_true = int(frame_sequence[2 * i])
        e_true = int(frame_sequence[2 * i + 1])
        true_class = int(y_true[i])
        total_gestures[true_class] += 1

        if not gating_list:
            jaccard_indices[true_class].append(0.0)
            continue

        s_pred = min(gating_list, key=lambda x: abs(x - s_true))
        e_pred = min(gating_list, key=lambda x: abs(x - e_true))
        if s_pred > e_pred:
            s_pred, e_pred = e_pred, s_pred

        start = min(s_true, s_pred)
        end   = max(e_true, e_pred)

        gt_sequence   = [0] * (end - start + 1)
        pred_sequence = [0] * (end - start + 1)

        for idx in range(s_true - start, e_true - start + 1):
            gt_sequence[idx] = 1

        for idx in range(s_pred - start, e_pred - start + 1):
            frame_idx = start + idx
            if frame_idx < len(y_pred_list) and y_pred_list[frame_idx] == true_class:
                pred_sequence[idx] = 1

        intersection = sum(1 for g, p in zip(gt_sequence, pred_sequence) if g == 1 and p == 1)
        union        = sum(1 for g, p in zip(gt_sequence, pred_sequence) if g == 1 or p == 1)
        jaccard = intersection / union if union != 0 else 0
        jaccard_indices[true_class].append(jaccard)

    average_jaccards = []
    for indices in jaccard_indices:
        avg_jaccard = sum(indices) / len(indices) if indices else None
        average_jaccards.append(avg_jaccard)

    for i in range(nc):
        global_total_gestures[i] += total_gestures[i]
        # ---- BUG: None entries never accumulate globally ----
        if average_jaccards[i] is not None:
            global_jaccard_indices[i].append(average_jaccards[i])

    if verbose:
        print("Results for the current sequence (Jaccard):")
        for i in range(nc):
            if average_jaccards[i] is not None:
                print(f"  Class {i}: Avg Jaccard={average_jaccards[i]:.2f}")
            else:
                print(f"  Class {i}: No data")
        print()


# ---------------------------------------------------------------------------
# Print global results (call after all sequences processed)
# ---------------------------------------------------------------------------

def print_global_results(class_names=None):
    """Print the aggregated detection rate, latency, Jaccard, and FPR."""
    nc = num_classes

    print("=" * 50)
    print("GLOBAL RESULTS (Original / Duo Streamers version)")
    print("=" * 50)

    total_dr_num, total_dr_den = 0, 0
    total_fp, total_gt = 0, 0
    all_jaccards = []

    for i in range(nc):
        name = class_names[i] if class_names else str(i)
        gt   = global_total_gestures[i]
        if gt == 0 and global_false_positives[i] == 0:
            continue

        dr = global_correct_predictions[i] / gt if gt > 0 else None
        avg_lat = (sum(global_latencies[i]) / len(global_latencies[i])
                   if global_latencies[i] else None)
        # BUG denominator preserved
        denom_fpr = global_false_positives[i] + gt
        fpr = global_false_positives[i] / denom_fpr if denom_fpr > 0 else None
        avg_jac = (sum(global_jaccard_indices[i]) / len(global_jaccard_indices[i])
                   if global_jaccard_indices[i] else None)

        print(f"Class {name}:")
        if dr  is not None: print(f"  Detection Rate : {dr:.4f}")
        if fpr is not None: print(f"  FP Rate (buggy): {fpr:.4f}")
        if avg_lat is not None: print(f"  Avg Latency    : {avg_lat:.2f} frames")
        if avg_jac is not None: print(f"  Avg Jaccard    : {avg_jac:.4f}")
        else:                   print(f"  Avg Jaccard    : No data (known bug)")

        if dr is not None:
            total_dr_num += global_correct_predictions[i]
            total_dr_den += gt
        total_fp += global_false_positives[i]
        total_gt += gt
        if avg_jac is not None:
            all_jaccards.append(avg_jac)

    print()
    print("--- Macro averages ---")
    overall_dr  = total_dr_num / total_dr_den if total_dr_den > 0 else 0
    overall_fpr = total_fp / (total_fp + total_gt) if (total_fp + total_gt) > 0 else 0
    overall_jac = sum(all_jaccards) / len(all_jaccards) if all_jaccards else float('nan')
    print(f"  Overall Detection Rate : {overall_dr:.4f}")
    print(f"  Overall FPR (buggy)    : {overall_fpr:.4f}")
    print(f"  Overall Jaccard Index  : {overall_jac:.4f}  {'(likely nan — known bug)' if overall_jac != overall_jac else ''}")
    print("=" * 50)
