"""
Main orchestrator for THCT-Net sign-language recognition.

Supports two modes:
    1. Single-split   : train on dev_users, test on one test_user
  2. Leave-One-Out  : iterate over ALL_USERS, each becomes the test user once

Usage examples:
  # Single split (default: test on user3)
  python main.py

    # Override dataset root so the script can be run from anywhere
    python main.py --dataset-root C:/Shoab/Thesis/Experiments/window_left/dataset

    # Override gloss-balanced batch sampler settings
    python main.py --glosses-per-batch 8 --samples-per-gloss 4

  # Single split with specific test user
  python main.py --test-user user1

  # Leave-one-out cross-validation over all users
  python main.py --louo

  # Override epochs / learning rate
  python main.py --louo --epochs 50 --lr 1e-3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from config import (
    ALL_USERS,
    BACKGROUND_LABEL,
    BAG_SIZE,
    BASE_CH,
    BATCH_SIZE,
    CONFIDENCE_THRESHOLD,
    D_MODEL,
    DATASET_ROOT,
    DEFAULT_DEV_USERS,
    DEFAULT_TEST_USER,
    DEV_VAL_RATIO,
    DEVICE,
    DROPOUT,
    EPOCHS,
    CHECKPOINT_PATHS_BY_USER,
    HIST_MOTION_EPS,
    HIST_N_BINS,
    HIST_RESAMPLE_STEPS,
    INPUT_DIM,
    LEARNING_RATE,
    MODEL_NAME,
    NORMALIZATION_NAME,
    NUM_HEADS,
    NUM_TRANSFORMER_LAYERS,
    SEED,
    STREAM_MODE,
    TEMPLATE_DIR,
    WER_EXAMPLE_PRINT_COUNT,
    USE_AUGMENTATION,
    AUGMENT_ROTATION_PROB,
    AUGMENT_ROTATION_RANGE,
    AUGMENT_SCALING_PROB,
    AUGMENT_SCALING_RANGE,
    AUGMENT_NOISE_PROB,
    AUGMENT_NOISE_STD,
    AUGMENT_DROPOUT_PROB,
    AUGMENT_DROPOUT_RATE,
    GLOSS_BALANCED_GLOSSES_PER_BATCH,
    GLOSS_BALANCED_SAMPLES_PER_GLOSS,
    WINDOW_SIZE,
    STRIDE,
)
from build_templates import (
    build_templates as build_disambig_templates,
    calibrate_theta_high,
    grid_search_tau_lambda,
)
from data_loading import load_all_segments
from data_splitting import prepare_split
from dataset import (
    FullSequenceDataset,
    collate_full_sequences,
    LeapSignDataset,
    collate_batch,
    GlossBalancedBatchSampler,
    build_windows_from_segments,
)
from evaluation import (
    evaluate_model_full,
    evaluate_model_wer,
    evaluate_streaming_metrics_from_wer_df,
    print_frame_level_report,
    save_split_results,
)
from features import palm_reference_normalize_sequence
from model import THCTNet
from trainer import train_model
from utils import (
    FocalLoss,
    TeeLogger,
    compute_class_weights,
    load_model_checkpoint,
    plot_training_curves,
    save_confusion_matrix_plots,
    save_unique_model,
)


NORMALIZE_FN = palm_reference_normalize_sequence


def build_dataloaders(
    split_data: dict,
    batch_size: int = BATCH_SIZE,
    augment_pipeline = None,
    glosses_per_batch: int = GLOSS_BALANCED_GLOSSES_PER_BATCH,
    samples_per_gloss: int = GLOSS_BALANCED_SAMPLES_PER_GLOSS,
) -> tuple[DataLoader, DataLoader, DataLoader, LeapSignDataset]:
    """Build train/val/test DataLoaders from split data."""
    import numpy as np
    
    # Generate windows from segments
    X_train, y_train = build_windows_from_segments(
        split_data["train_segments"],
        split_data["label_to_id"],
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
    X_val, y_val = build_windows_from_segments(
        split_data["val_segments"],
        split_data["label_to_id"],
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )
    X_test, y_test = build_windows_from_segments(
        split_data["test_segments"],
        split_data["label_to_id"],
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    # Filter val/test to only include labels seen in training
    known_labels = set(y_train.tolist())
    val_mask = np.array([lbl in known_labels for lbl in y_val], dtype=bool) if len(y_val) > 0 else np.array([], dtype=bool)
    if len(val_mask) > 0:
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]

    test_mask = np.array([lbl in known_labels for lbl in y_test], dtype=bool) if len(y_test) > 0 else np.array([], dtype=bool)
    if len(test_mask) > 0:
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

    train_ds = LeapSignDataset(
        X_train, y_train, normalize_fn=NORMALIZE_FN, augment_pipeline=augment_pipeline
    )
    val_ds = LeapSignDataset(
        X_val, y_val, normalize_fn=NORMALIZE_FN
    )
    test_ds = LeapSignDataset(
        X_test, y_test, normalize_fn=NORMALIZE_FN
    )

    # Instantiate GlossBalancedBatchSampler
    train_sampler = GlossBalancedBatchSampler(
        labels=y_train,
        glosses_per_batch=glosses_per_batch,
        samples_per_gloss=samples_per_gloss,
        seed=SEED,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    print(f"Train windows: {len(X_train)} | Val windows: {len(X_val)} | Test windows: {len(X_test)}")
    print(f"Train batches: {len(train_loader)} | "
          f"Val batches: {len(val_loader)} | "
          f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, train_ds


def run_single_fold(
    segments_by_user: dict[str, list[dict]],
    dev_users: list[str],
    test_user: str,
    epochs: int,
    lr: float,
    save_dir: str = "trained_models",
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    exclude_train_seq: str = "",
    run_timestamp: str = "",
    augment_pipeline = None,
    checkpoint_path: str | None = None,
    glosses_per_batch: int = GLOSS_BALANCED_GLOSSES_PER_BATCH,
    samples_per_gloss: int = GLOSS_BALANCED_SAMPLES_PER_GLOSS,
    use_disambiguation: bool = False,
    disambig_calibrate: bool = False,
    disambig_dir: str = TEMPLATE_DIR,
    disambig_theta_percentile: float = 15.0,
) -> dict:
    """Run a complete train→evaluate cycle for one fold.

    Returns a dict with all results for this fold.
    """
    print(f"\n{'='*70}")
    print(f"FOLD: test_user={test_user}, dev_users={dev_users}")
    print(f"{'='*70}")

    # ── 1. Split data ──
    split_data = prepare_split(
        segments_by_user=segments_by_user,
        dev_users=dev_users,
        test_user=test_user,
        exclude_train_seq=exclude_train_seq,
    )

    # ── 2. Build dataloaders ──
    train_loader, val_loader, test_loader, train_ds = build_dataloaders(
        split_data,
        augment_pipeline=augment_pipeline,
        glosses_per_batch=glosses_per_batch,
        samples_per_gloss=samples_per_gloss,
    )

    # ── 3. Class weights + loss ──
    num_classes = split_data["num_classes"]
    if use_focal_loss:
        class_weight_tensor = compute_class_weights(train_ds, num_classes)
        criterion = FocalLoss(
            weight=class_weight_tensor, gamma=focal_gamma, ignore_index=-1,
        )
        print(f"Loss function: FocalLoss (gamma={focal_gamma}, weighted)")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        print(f"Loss function: CrossEntropyLoss")

    # ── 4. Model ──
    model = THCTNet(
        num_classes=num_classes,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_TRANSFORMER_LAYERS,
        base_ch=BASE_CH,
        window_size=WINDOW_SIZE,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_NAME} | Trainable params: {trainable:,}")

    # ── 5. Train or load checkpoint ──
    train_result: dict
    if checkpoint_path:
        print(f"Loading checkpoint instead of training: {checkpoint_path}")
        model_state_dict, metadata = load_model_checkpoint(checkpoint_path)
        model.load_state_dict(model_state_dict)
        train_result = {
            "best_val_acc": float(metadata.get("best_val_acc", float("nan"))),
            "best_val_f1": float(metadata.get("best_val_f1", float("nan"))),
            "best_model_state": None,
            "history_train_loss": [],
            "history_train_batch_acc": [],
            "history_val_acc": [],
            "history_train_f1": [],
            "history_val_f1": [],
            "epoch_history": [],
        }
    else:
        train_result = train_model(
            model_obj=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            num_classes=num_classes,
            epochs=epochs,
            lr=lr,
            use_grad_clip=True,
            grad_clip_norm=1.0,
            early_stopping_patience=15,
        )

        # Restore best model
        if train_result["best_model_state"] is not None:
            model.load_state_dict(train_result["best_model_state"])

    # ── 6. Window-level evaluation ──
    confusion_dir = os.path.join(save_dir, "plots", "confusion_matrices", test_user)
    frame_summary = print_frame_level_report(
        model_obj=model,
        loaders={"Train": train_loader, "Val": val_loader, "Test": test_loader},
        id_to_label=split_data["id_to_label"],
        num_classes=num_classes,
        save_confusion_matrix_dir=confusion_dir,
        save_confusion_splits={"Val", "Test"},
    )
    print("\n========== WINDOW ACCURACY SUMMARY ==========")
    print(frame_summary.to_string(index=False))

    # ── 6b. Disambiguation templates (Hook A/B) — build once per fold, reuse after ──
    theta_high      = None
    disambig_templates = None
    disambig_class_order = None
    disambig_tau    = None
    disambig_lambda = None

    if use_disambiguation:
        templates_path = Path(disambig_dir) / test_user / "templates.npz"
        meta_path = Path(str(templates_path) + ".json")

        if templates_path.exists() and meta_path.exists():
            print(f"\nLoading existing disambiguation templates: {templates_path}")
            npz = np.load(templates_path, allow_pickle=True)
            disambig_templates = npz["templates"]
            disambig_class_order = list(npz["class_order"])
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            theta_high      = meta.get("theta_high")
            disambig_tau    = meta.get("tau_margin")
            disambig_lambda = meta.get("lambda")
        else:
            print(f"\nBuilding disambiguation templates for fold test_user={test_user} ...")
            disambig_templates, disambig_class_order = build_disambig_templates(
                split_data,
                n_bins=HIST_N_BINS,
                resample_steps=HIST_RESAMPLE_STEPS,
                motion_eps=HIST_MOTION_EPS,
            )

            if disambig_calibrate:
                print(f"Calibrating theta_high (percentile={disambig_theta_percentile}) on dev_users' val split ...")
                theta_high = calibrate_theta_high(
                    split_data["dev_val_wer_catalog"], percentile=disambig_theta_percentile,
                )
                print(f"  theta_high = {theta_high}")

                print("Grid-searching tau_margin / lambda on dev_users' val split ...")
                disambig_tau, disambig_lambda = grid_search_tau_lambda(
                    dev_val_wer_catalog=split_data["dev_val_wer_catalog"],
                    model_obj=model,
                    id_to_label=split_data["id_to_label"],
                    templates=disambig_templates,
                    class_order=disambig_class_order,
                    theta_high=theta_high,
                    tau_grid=[0.05, 0.10, 0.15, 0.20, 0.30],
                    lambda_grid=[0.2, 0.4, 0.6, 0.8],
                )
                print(f"  Chosen tau_margin={disambig_tau}, lambda={disambig_lambda}")

            templates_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                templates_path,
                templates=disambig_templates,
                class_order=np.array(disambig_class_order, dtype=object),
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "test_user": test_user,
                    "dev_users": dev_users,
                    "n_bins": HIST_N_BINS,
                    "resample_steps": HIST_RESAMPLE_STEPS,
                    "motion_eps": HIST_MOTION_EPS,
                    "theta_high": theta_high,
                    "tau_margin": disambig_tau,
                    "lambda": disambig_lambda,
                    "background_id": split_data["background_id"],
                }, f, indent=2)
            print(f"Saved disambiguation templates to {templates_path}")

    # ── 7. WER evaluation ──
    id_to_label = split_data["id_to_label"]
    results_dir = os.path.join(save_dir, "results")

    test_wer_df = pd.DataFrame()
    if split_data["test_wer_catalog"]:
        test_wer_df = evaluate_model_wer(
            samples=split_data["test_wer_catalog"],
            split_name=f"Test ({test_user})",
            model_obj=model,
            normalize_fn=NORMALIZE_FN,
            id_to_label=id_to_label,
            normalization_name=NORMALIZATION_NAME,
            print_examples=WER_EXAMPLE_PRINT_COUNT,
            theta_high=theta_high,
            templates=disambig_templates,
            class_order=disambig_class_order,
            tau_margin=disambig_tau,
            lam=disambig_lambda,
        )
        save_split_results(test_wer_df, split_name=f"test_{test_user}_{run_timestamp}", results_dir=results_dir)

    val_wer_df = pd.DataFrame()
    if split_data["dev_val_wer_catalog"]:
        val_wer_df = evaluate_model_wer(
            samples=split_data["dev_val_wer_catalog"],
            split_name="Dev val",
            model_obj=model,
            normalize_fn=NORMALIZE_FN,
            id_to_label=id_to_label,
            normalization_name=NORMALIZATION_NAME,
            print_examples=WER_EXAMPLE_PRINT_COUNT,
            theta_high=theta_high,
            templates=disambig_templates,
            class_order=disambig_class_order,
            tau_margin=disambig_tau,
            lam=disambig_lambda,
        )
        save_split_results(val_wer_df, split_name=f"val_{test_user}_{run_timestamp}", results_dir=results_dir)

    # ── 7b. Streaming metrics (SHREC'21: DR, FPR, Jaccard) ──
    if len(test_wer_df) > 0:
        evaluate_streaming_metrics_from_wer_df(
            wer_df=test_wer_df,
            split_name=f"Test ({test_user})",
            label_to_id=split_data["label_to_id"],
            id_to_label=id_to_label,
            num_classes=num_classes,
        )

    # ── 8. Save model ──
    test_mean_wer = (
        float(test_wer_df["wer"].mean()) if len(test_wer_df) else None
    )
    if checkpoint_path:
        saved_path = checkpoint_path
    else:
        saved_path = save_unique_model(
            model_obj=model,
            best_val_acc=train_result["best_val_acc"],
            save_dir=save_dir,
            model_name=MODEL_NAME,
            info={
                "test_user": test_user,
                "dev_users": dev_users,
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "num_transformer_layers": NUM_TRANSFORMER_LAYERS,
                "base_ch": BASE_CH,
                "test_mean_wer": test_mean_wer,
                "best_val_f1": train_result["best_val_f1"],
            },
        )

    # ── 9. Training curves ──
    curves_dir = os.path.join(save_dir, "plots")
    os.makedirs(curves_dir, exist_ok=True)
    plot_training_curves(
        train_result,
        save_path=os.path.join(curves_dir, f"curves_test_{test_user}_{run_timestamp}.png"),
    )

    return {
        "test_user": test_user,
        "dev_users": dev_users,
        "best_val_acc": train_result["best_val_acc"],
        "best_val_f1": train_result["best_val_f1"],
        "test_mean_wer": test_mean_wer,
        "test_wer_df": test_wer_df,
        "val_wer_df": val_wer_df,
        "frame_summary": frame_summary,
        "saved_path": saved_path,
        "train_result": train_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="THCT-Net Sign Language Recognition — Modular Pipeline",
    )
    parser.add_argument(
        "--louo", action="store_true",
        help="Run leave-one-out user cross-validation over all users.",
    )
    parser.add_argument(
        "--test-mode", action="store_true",
        help="Run in test mode with a tiny subset of data (2 recordings per user) to quickly verify execution.",
    )
    parser.add_argument(
        "--test-user", type=str, default=DEFAULT_TEST_USER,
        help=f"Test user for single-split mode (default: {DEFAULT_TEST_USER}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--save-dir", type=str, default="trained_models",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=DATASET_ROOT,
        help=f"Dataset root directory (default: {DATASET_ROOT}).",
    )
    parser.add_argument(
        "--glosses-per-batch", type=int, default=GLOSS_BALANCED_GLOSSES_PER_BATCH,
        help=(
            "Number of unique glosses per batch for the gloss-balanced sampler "
            f"(default: {GLOSS_BALANCED_GLOSSES_PER_BATCH})."
        ),
    )
    parser.add_argument(
        "--samples-per-gloss", type=int, default=GLOSS_BALANCED_SAMPLES_PER_GLOSS,
        help=(
            "Number of samples per gloss in the gloss-balanced sampler "
            f"(default: {GLOSS_BALANCED_SAMPLES_PER_GLOSS})."
        ),
    )
    parser.add_argument(
        "--comment", type=str, default="",
        help="Custom comment to print at the top of the log/run history.",
    )
    parser.add_argument(
        "--from-checkpoint", action="store_true",
        help="Skip training and load pre-trained checkpoints defined in config for each fold.",
    )
    parser.add_argument(
        "--focal-loss", action="store_true",
        help="Use Focal Loss instead of CrossEntropyLoss.",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Gamma parameter for Focal Loss (default: 2.0).",
    )
    parser.add_argument(
        "--exclude-train-seq", type=str, default="",
        help="Sequence identifier to exclude from the training set (e.g. S11).",
    )
    # Disambiguation arguments
    parser.add_argument(
        "--use-disambiguation", action="store_true",
        help=(
            "Enable post-logit disambiguation (Hook A background rescue + Hook B "
            "sign-vs-sign re-ranking) during WER evaluation. Per-fold templates are "
            "built automatically the first time a fold runs and reused afterwards."
        ),
    )
    parser.add_argument(
        "--disambig-calibrate", action="store_true",
        help=(
            "When building a fold's templates, also calibrate theta_high (Hook A) "
            "and tau_margin/lambda (Hook B) on dev_users' val split. Requires extra "
            "WER evaluation sweeps — slower than building templates alone. Ignored "
            "if templates already exist for this fold."
        ),
    )
    parser.add_argument(
        "--disambig-dir", type=str, default=TEMPLATE_DIR,
        help=f"Directory to store/load per-fold disambiguation templates (default: {TEMPLATE_DIR}).",
    )
    parser.add_argument(
        "--disambig-theta-percentile", type=float, default=15.0,
        help="Percentile (10-25 recommended) of sign-frame motion energy used for theta_high during calibration (default: 15.0).",
    )
    # Augmentation arguments
    parser.add_argument(
        "--augment", action="store_true", default=USE_AUGMENTATION,
        help="Enable training data augmentation.",
    )
    parser.add_argument(
        "--rotation-prob", type=float, default=AUGMENT_ROTATION_PROB,
        help="Probability of applying small 3D rotation.",
    )
    parser.add_argument(
        "--rotation-range", type=float, default=AUGMENT_ROTATION_RANGE,
        help="Maximum degrees of X/Y/Z rotation (max safe limit: 10, absolute max: 15).",
    )
    parser.add_argument(
        "--scaling-prob", type=float, default=AUGMENT_SCALING_PROB,
        help="Probability of applying small uniform scaling.",
    )
    parser.add_argument(
        "--scaling-min", type=float, default=AUGMENT_SCALING_RANGE[0],
        help="Minimum scaling factor (max range: 0.90).",
    )
    parser.add_argument(
        "--scaling-max", type=float, default=AUGMENT_SCALING_RANGE[1],
        help="Maximum scaling factor (max range: 1.10).",
    )
    parser.add_argument(
        "--noise-prob", type=float, default=AUGMENT_NOISE_PROB,
        help="Probability of adding Gaussian coordinate noise.",
    )
    parser.add_argument(
        "--noise-std", type=float, default=AUGMENT_NOISE_STD,
        help="Standard deviation of Gaussian coordinate noise in mm (recommended: 1.0-3.0 mm).",
    )
    parser.add_argument(
        "--dropout-prob", type=float, default=AUGMENT_DROPOUT_PROB,
        help="Probability of applying random frame dropout.",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=AUGMENT_DROPOUT_RATE,
        help="Percentage of frames to drop (max safe limit: 15%%).",
    )

    args = parser.parse_args()

    # config.USE_DISAMBIGUATION is read dynamically by decoder.py at call
    # time, so setting it here takes effect for the whole run.
    config.USE_DISAMBIGUATION = args.use_disambiguation

    # ── Initialize Logging (Tee) ──
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode_str = "LOUO" if args.louo else f"single_{args.test_user}"
    log_filename = f"run_{mode_str}_{ts}.log"
    log_filepath = os.path.join(args.save_dir, "logs", log_filename)
    
    logger = TeeLogger(log_filepath)
    
    try:
        print(f"[ignoring loop detection]")
        print(f"==================================================")
        if args.comment:
            print(f"COMMENT: {args.comment}")
            print(f"==================================================")
        print(f"RUN START TIME (UTC): {ts}")
        print(f"Log filepath        : {log_filepath}")
        print(f"==================================================")
        print(f"COMMAND LINE ARGUMENTS:")
        for arg_name, arg_val in vars(args).items():
            print(f"  --{arg_name:<10}: {arg_val}")
        print(f"==================================================")
        print(f"CENTRAL HYPERPARAMETERS:")
        print(f"  Device            : {DEVICE}")
        print(f"  Dataset root      : {args.dataset_root}")
        print(f"  Glosses/batch     : {args.glosses_per_batch}")
        print(f"  Samples/gloss     : {args.samples_per_gloss}")
        print(f"  Seed              : {SEED}")
        print(f"  Dev-Val Ratio     : {DEV_VAL_RATIO}")
        print(f"  Batch size        : {BATCH_SIZE}")
        print(f"  Test mode         : {args.test_mode}")
        print(f"  Model name        : {MODEL_NAME}")
        print(f"  Normalization     : {NORMALIZATION_NAME}")
        print(f"  Learning rate     : {args.lr}")
        print(f"  Epochs            : {args.epochs}")
        print(f"  From checkpoint   : {args.from_checkpoint}")
        print(f"  D_MODEL           : {D_MODEL}")
        print(f"  NUM_HEADS         : {NUM_HEADS}")
        print(f"  NUM_TRANS_LAYERS  : {NUM_TRANSFORMER_LAYERS}")
        print(f"  BASE_CH           : {BASE_CH}")
        print(f"  Dropout           : {DROPOUT}")
        print(f"  Decoder bag size  : {BAG_SIZE}")
        print(f"  Decoder threshold : {CONFIDENCE_THRESHOLD}")
        print(f"  Loss function     : {'FocalLoss (gamma=' + str(args.focal_gamma) + ')' if args.focal_loss else 'CrossEntropyLoss'}")
        print(f"  Disambiguation    : {args.use_disambiguation}")
        if args.use_disambiguation:
            print(f"    Calibrate       : {args.disambig_calibrate}")
            print(f"    Templates dir   : {args.disambig_dir}")
        if args.exclude_train_seq:
            print(f"  Excluded train seq: {args.exclude_train_seq}")
        print(f"  Augmentation      : {args.augment}")
        if args.augment:
            print(f"    Rotation prob   : {args.rotation_prob} (range: ±{args.rotation_range} deg)")
            print(f"    Scaling prob    : {args.scaling_prob} (range: [{args.scaling_min}, {args.scaling_max}])")
            print(f"    Noise prob      : {args.noise_prob} (std: {args.noise_std} mm)")
            print(f"    Dropout prob    : {args.dropout_prob} (rate: {args.dropout_rate})")
        print(f"==================================================\n")

        print(f"Using device: {DEVICE}")
        print(f"Dataset root: {args.dataset_root}")

        # ── Load all data once ──
        _, segments_by_user = load_all_segments(args.dataset_root)

        # ── Initialize Augmentation Pipeline ──
        if args.augment:
            from augmentations import SignLanguageAugmentationPipeline
            augment_pipeline = SignLanguageAugmentationPipeline(
                rotation_prob=args.rotation_prob,
                rotation_range=args.rotation_range,
                scaling_prob=args.scaling_prob,
                scaling_range=(args.scaling_min, args.scaling_max),
                noise_prob=args.noise_prob,
                noise_std=args.noise_std,
                dropout_prob=args.dropout_prob,
                dropout_rate=args.dropout_rate,
                seed=SEED,
            )
        else:
            augment_pipeline = None
        
        if args.test_mode:
            print(f"\n[!] TEST MODE ENABLED: Limiting to 2 recordings per user.")
            for user in segments_by_user:
                seen_recs = []
                filtered_segments = []
                for s in segments_by_user[user]:
                    rid = s["recording_id"]
                    if rid not in seen_recs:
                        if len(seen_recs) >= 2:
                            continue
                        seen_recs.append(rid)
                    filtered_segments.append(s)
                segments_by_user[user] = filtered_segments
                
            args.epochs = min(args.epochs, 2)
            print(f"[!] TEST MODE: Limiting training to {args.epochs} epoch(s).")

        available_users = sorted(segments_by_user.keys())
        print(f"Available users: {available_users}")

        # ── Determine folds ──
        if args.louo:
            # Leave-one-out: each user becomes the test user
            users_to_test = [u for u in ALL_USERS if u in available_users]
            print(f"\n{'#'*70}")
            print(f"LEAVE-ONE-OUT USER CROSS-VALIDATION")
            print(f"Users to evaluate: {users_to_test}")
            print(f"{'#'*70}")
        else:
            users_to_test = [args.test_user]

        checkpoint_map: dict[str, str] = {}
        if args.from_checkpoint:
            missing_users = [user for user in users_to_test if user not in CHECKPOINT_PATHS_BY_USER]
            if missing_users:
                raise RuntimeError(
                    "--from-checkpoint was enabled but no checkpoint path was configured for: "
                    + ", ".join(missing_users)
                )

            checkpoint_map = {
                user: str(Path(CHECKPOINT_PATHS_BY_USER[user]))
                for user in users_to_test
            }

            missing_files = [path for path in checkpoint_map.values() if not Path(path).exists()]
            if missing_files:
                raise FileNotFoundError(
                    "Configured checkpoint file(s) do not exist: " + ", ".join(missing_files)
                )

        # ── Run folds ──
        all_fold_results = []

        for test_user in users_to_test:
            dev_users = [u for u in available_users if u != test_user]

            fold_result = run_single_fold(
                segments_by_user=segments_by_user,
                dev_users=dev_users,
                test_user=test_user,
                epochs=args.epochs,
                lr=args.lr,
                save_dir=args.save_dir,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                exclude_train_seq=args.exclude_train_seq,
                run_timestamp=ts,
                augment_pipeline=augment_pipeline,
                checkpoint_path=checkpoint_map.get(test_user),
                glosses_per_batch=args.glosses_per_batch,
                samples_per_gloss=args.samples_per_gloss,
                use_disambiguation=args.use_disambiguation,
                disambig_calibrate=args.disambig_calibrate,
                disambig_dir=args.disambig_dir,
                disambig_theta_percentile=args.disambig_theta_percentile,
            )
            all_fold_results.append(fold_result)

        # ── Final summary ──
        print(f"\n{'='*70}")
        print("FINAL SUMMARY ACROSS ALL FOLDS")
        print(f"{'='*70}")

        summary_rows = []
        for r in all_fold_results:
            summary_rows.append({
                "test_user": r["test_user"],
                "dev_users": ", ".join(r["dev_users"]),
                "best_val_acc": r["best_val_acc"],
                "best_val_f1": r["best_val_f1"],
                "test_mean_wer": r["test_mean_wer"],
                "saved_path": r["saved_path"],
            })

        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))

        if args.louo and len(all_fold_results) > 1:
            wers = [
                r["test_mean_wer"] for r in all_fold_results
                if r["test_mean_wer"] is not None
            ]
            accs = [r["best_val_acc"] for r in all_fold_results]
            f1s = [r["best_val_f1"] for r in all_fold_results]
            if wers:
                import numpy as np
                print(f"\nLOUO Mean Test WER: {np.mean(wers):.4f} ± {np.std(wers):.4f}")
                print(f"LOUO Best Val Acc:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
                print(f"LOUO Best Val F1:   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
