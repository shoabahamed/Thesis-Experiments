"""
Offline, per-LOUO-fold builder for the disambiguation template bank.

Builds Hook B's per-class turning-angle-histogram templates from dev_users'
training segments, and (optionally, with --calibrate) calibrates Hook A's
theta_high plus Hook B's tau_margin/lambda on dev_users' validation split.
Never touches the held-out test user.

Usage
-----
    python build_templates.py --dataset-root <path> --dev-users user1 user2 user3 \\
        --test-user user5 --out trained_models/disambiguation/user5/templates.npz

    # Also calibrate theta_high / tau_margin / lambda (requires a trained
    # checkpoint for this fold, used only for streaming dev-val recordings):
    python build_templates.py --dataset-root <path> --dev-users user1 user2 user3 \\
        --test-user user5 --checkpoint <fold.pt> --calibrate \\
        --out trained_models/disambiguation/user5/templates.npz

Run once per fold — never reuse one fold's templates.npz on another fold.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import (
    BASE_CH,
    D_MODEL,
    DATASET_ROOT,
    DEFAULT_DEV_USERS,
    DEFAULT_TEST_USER,
    DISAMBIG_LAMBDA,
    DISAMBIG_TAU_MARGIN,
    HIST_MOTION_EPS,
    HIST_N_BINS,
    HIST_RESAMPLE_STEPS,
    ENERGY_CONF_THRESH,
    ENERGY_GRAB_WEIGHT,
    ENERGY_PINCH_WEIGHT,
    NUM_HEADS,
    NUM_TRANSFORMER_LAYERS,
    TEMPLATE_DIR,
    WINDOW_SIZE,
)
from data_loading import load_all_segments
from data_splitting import prepare_split
from disambiguation import build_class_templates, compute_motion_energy
from evaluation import evaluate_model_wer
from features import palm_reference_normalize_sequence
from model import THCTNet
from utils import load_model_checkpoint


def build_templates(split_data: dict, n_bins: int, resample_steps: int, motion_eps: float):
    """Build the (num_classes, 2*n_bins) template bank from dev_users' train split."""
    id_to_label = split_data["id_to_label"]
    num_classes = split_data["num_classes"]
    class_order = [id_to_label[i] for i in range(num_classes)]

    labeled_segments = [
        {"label": item["label"], "raw_aux": item["segment_aux"]}
        for item in split_data["train_segments"]
        if not item.get("is_background", False)
    ]

    templates = build_class_templates(
        labeled_segments, class_order,
        n_bins=n_bins, resample_steps=resample_steps, motion_eps=motion_eps,
    )
    return templates, class_order


def calibrate_theta_high(dev_val_wer_catalog: list[dict], percentile: float) -> float | None:
    """
    Percentile of per-frame motion energy E(t), restricted to frames inside
    ground-truth sign regions, over dev_users' validation recordings.
    """
    energies = []
    for sample in dev_val_wer_catalog:
        v_aux = sample.get("V_aux")
        if v_aux is None:
            continue
        for region in sample.get("segmentation_regions", []):
            start, end = int(region["start_frame"]), int(region["end_frame"])
            prev_row = None
            for t in range(start, end + 1):
                if t < 0 or t >= v_aux.shape[0]:
                    continue
                row = v_aux[t]
                e = compute_motion_energy(
                    row, prev_row,
                    conf_thresh=ENERGY_CONF_THRESH,
                    grab_w=ENERGY_GRAB_WEIGHT,
                    pinch_w=ENERGY_PINCH_WEIGHT,
                )
                energies.append(e)
                prev_row = row

    if not energies:
        return None
    return float(np.percentile(energies, percentile))


def grid_search_tau_lambda(
    dev_val_wer_catalog: list[dict],
    model_obj,
    id_to_label: dict[int, str],
    templates: np.ndarray,
    class_order: list[str],
    theta_high: float | None,
    tau_grid: list[float],
    lambda_grid: list[float],
) -> tuple[float, float]:
    """Sweep (tau_margin, lambda) on dev_val, pick the pair with lowest mean WER."""
    import config as _cfg

    # USE_DISAMBIGUATION is read dynamically by decoder.py (config.USE_DISAMBIGUATION),
    # so this toggle takes effect immediately without reloading the module.
    # tau_margin/lambda are passed explicitly per call instead of mutated on
    # config, since stream_model_online's defaults are bound at import time.
    prev_flag = _cfg.USE_DISAMBIGUATION
    _cfg.USE_DISAMBIGUATION = True

    best_tau, best_lam, best_wer = tau_grid[0], lambda_grid[0], float("inf")
    try:
        for tau, lam in itertools.product(tau_grid, lambda_grid):
            df = evaluate_model_wer(
                samples=dev_val_wer_catalog,
                split_name="dev_val_calibration",
                model_obj=model_obj,
                normalize_fn=palm_reference_normalize_sequence,
                id_to_label=id_to_label,
                normalization_name="palm_ref",
                theta_high=theta_high,
                templates=templates,
                class_order=class_order,
                tau_margin=tau,
                lam=lam,
            )
            mean_wer = float(df["wer"].mean()) if len(df) else float("inf")
            print(f"  tau={tau:.3f} lambda={lam:.3f} -> dev-val mean WER={mean_wer:.4f}")

            if mean_wer < best_wer:
                best_wer, best_tau, best_lam = mean_wer, tau, lam
    finally:
        _cfg.USE_DISAMBIGUATION = prev_flag

    return best_tau, best_lam


def main():
    parser = argparse.ArgumentParser(
        description="Build per-fold Hook B templates (and optionally calibrate Hook A/B constants).",
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT,
                         help=f"Dataset root directory (default: {DATASET_ROOT}).")
    parser.add_argument("--dev-users", type=str, nargs="+", default=DEFAULT_DEV_USERS,
                         help=f"Dev users to build templates/calibrate from (default: {DEFAULT_DEV_USERS}).")
    parser.add_argument("--test-user", type=str, default=DEFAULT_TEST_USER,
                         help=f"Held-out test user for this fold — never touched (default: {DEFAULT_TEST_USER}).")
    parser.add_argument("--n-bins", type=int, default=HIST_N_BINS,
                         help=f"Turning-angle histogram bins (default: {HIST_N_BINS}).")
    parser.add_argument("--resample-steps", type=int, default=HIST_RESAMPLE_STEPS,
                         help=f"Resample length before angle computation (default: {HIST_RESAMPLE_STEPS}).")
    parser.add_argument("--motion-eps", type=float, default=HIST_MOTION_EPS,
                         help=f"Minimum speed to define a direction (default: {HIST_MOTION_EPS}).")
    parser.add_argument("--out", type=str, required=True,
                         help="Output path for templates.npz.")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Trained checkpoint for this fold (required with --calibrate).")
    parser.add_argument("--calibrate", action="store_true",
                         help="Also calibrate theta_high / tau_margin / lambda on dev_users' val split.")
    parser.add_argument("--theta-percentile", type=float, default=15.0,
                         help="Percentile (10-25 recommended) of sign-frame motion energy for theta_high (default: 15.0).")
    parser.add_argument("--tau-grid", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30],
                         help="Grid of tau_margin values to sweep during calibration.")
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8],
                         help="Grid of lambda values to sweep during calibration.")
    args = parser.parse_args()

    if args.calibrate and not args.checkpoint:
        parser.error("--calibrate requires --checkpoint")

    print(f"Loading segments from {args.dataset_root} ...")
    _, segments_by_user = load_all_segments(args.dataset_root)

    split_data = prepare_split(
        segments_by_user=segments_by_user,
        dev_users=args.dev_users,
        test_user=args.test_user,
    )

    print("Building Hook B templates from dev_users' training segments ...")
    templates, class_order = build_templates(
        split_data, n_bins=args.n_bins, resample_steps=args.resample_steps, motion_eps=args.motion_eps,
    )

    theta_high = None
    tau_margin = DISAMBIG_TAU_MARGIN
    lam = DISAMBIG_LAMBDA

    if args.calibrate:
        print(f"Calibrating theta_high (percentile={args.theta_percentile}) on dev_users' val split ...")
        theta_high = calibrate_theta_high(
            split_data["dev_val_wer_catalog"], percentile=args.theta_percentile,
        )
        print(f"  theta_high = {theta_high}")

        print(f"Loading checkpoint {args.checkpoint} ...")
        model = THCTNet(
            num_classes=split_data["num_classes"],
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_TRANSFORMER_LAYERS,
            base_ch=BASE_CH,
            window_size=WINDOW_SIZE,
        )
        model_state_dict, _ = load_model_checkpoint(args.checkpoint)
        model.load_state_dict(model_state_dict)
        model.eval()

        print("Grid-searching tau_margin / lambda on dev_users' val split ...")
        tau_margin, lam = grid_search_tau_lambda(
            dev_val_wer_catalog=split_data["dev_val_wer_catalog"],
            model_obj=model,
            id_to_label=split_data["id_to_label"],
            templates=templates,
            class_order=class_order,
            theta_high=theta_high,
            tau_grid=args.tau_grid,
            lambda_grid=args.lambda_grid,
        )
        print(f"  Chosen tau_margin={tau_margin}, lambda={lam}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        templates=templates,
        class_order=np.array(class_order, dtype=object),
    )

    metadata = {
        "test_user": args.test_user,
        "dev_users": args.dev_users,
        "n_bins": args.n_bins,
        "resample_steps": args.resample_steps,
        "motion_eps": args.motion_eps,
        "theta_high": theta_high,
        "tau_margin": tau_margin,
        "lambda": lam,
        "background_id": split_data["background_id"],
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    with open(str(out_path) + ".json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved templates to {out_path} (+ .json metadata)")


if __name__ == "__main__":
    main()
