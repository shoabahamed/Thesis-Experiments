"""
Main orchestrator for LSTM sign-language recognition.

Supports two modes:
  1. Single-split   : train on dev_users, test on one test_user (legacy behavior)
  2. Leave-One-Out  : iterate over ALL_USERS, each becomes the test user once

Usage examples:
  # Single split (default: test on user3)
  python main.py

  # Single split with specific test user
  python main.py --test-user user1

  # Leave-one-out cross-validation over all users
  python main.py --louo

  # Override epochs / learning rate
  python main.py --louo --epochs 50 --lr 1e-3
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    ALL_USERS,
    BACKGROUND_LABEL,
    BAG_SIZE,
    BATCH_SIZE,
    CONFIDENCE_THRESHOLD,
    DATASET_ROOT,
    DEFAULT_DEV_USERS,
    DEFAULT_TEST_USER,
    DEV_VAL_RATIO,
    DEVICE,
    DROPOUT,
    EPOCHS,
    FEAT_DIM,
    HIDDEN_SIZE,
    INPUT_DIM,
    LEARNING_RATE,
    MODEL_NAME,
    NORMALIZATION_NAME,
    NUM_LSTM_LAYERS,
    SEED,
    STREAM_MODE,
    WER_EXAMPLE_PRINT_COUNT,
)
from data_loading import load_all_segments
from data_splitting import prepare_split
from dataset import FullSequenceDataset, collate_full_sequences
from evaluation import (
    evaluate_lstm_wer,
    evaluate_streaming_metrics,
    print_frame_level_report,
    save_split_results,
)
from features import palm_reference_normalize_sequence
from model import FullSequenceLSTM
from trainer import train_lstm_model
from utils import (
    FocalLoss,
    TeeLogger,
    compute_class_weights,
    plot_training_curves,
    save_unique_model,
)


NORMALIZE_FN = palm_reference_normalize_sequence


def build_dataloaders(
    split_data: dict,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from split data."""
    train_ds = FullSequenceDataset(
        split_data["train_segments"],
        split_data["label_to_id"],
        normalize_fn=NORMALIZE_FN,
        background_id=split_data["background_id"],
    )
    val_ds = FullSequenceDataset(
        split_data["val_segments"],
        split_data["label_to_id"],
        normalize_fn=NORMALIZE_FN,
        background_id=split_data["background_id"],
    )
    test_ds = FullSequenceDataset(
        split_data["test_segments"],
        split_data["label_to_id"],
        normalize_fn=NORMALIZE_FN,
        background_id=split_data["background_id"],
    )

    loader_kwargs = dict(num_workers=0, collate_fn=collate_full_sequences)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )

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
    )

    # ── 2. Build dataloaders ──
    train_loader, val_loader, test_loader, train_ds = build_dataloaders(split_data)

    # ── 3. Class weights + loss ──
    num_classes = split_data["num_classes"]
    class_weight_tensor = compute_class_weights(train_ds, num_classes)
    if use_focal_loss:
        criterion = FocalLoss(
            weight=class_weight_tensor, gamma=focal_gamma, ignore_index=-1,
        )
        print(f"Loss function: FocalLoss (gamma={focal_gamma}, weighted)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, ignore_index=-1)
        print(f"Loss function: CrossEntropyLoss (weighted)")

    # ── 4. Model ──
    model = FullSequenceLSTM(
        input_dim=INPUT_DIM,
        feat_dim=FEAT_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LSTM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {MODEL_NAME} | Trainable params: {trainable:,}")

    # ── 5. Train ──
    train_result = train_lstm_model(
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

    # ── 6. Frame-level evaluation ──
    frame_summary = print_frame_level_report(
        model_obj=model,
        loaders={"Train": train_loader, "Val": val_loader, "Test": test_loader},
        id_to_label=split_data["id_to_label"],
        num_classes=num_classes,
    )
    print("\n========== FRAME ACCURACY SUMMARY ==========")
    print(frame_summary.to_string(index=False))

    # ── 7. WER evaluation ──
    id_to_label = split_data["id_to_label"]
    results_dir = os.path.join(save_dir, "results")

    test_wer_df = pd.DataFrame()
    if split_data["test_wer_catalog"]:
        test_wer_df = evaluate_lstm_wer(
            samples=split_data["test_wer_catalog"],
            split_name=f"Test ({test_user})",
            model_obj=model,
            normalize_fn=NORMALIZE_FN,
            id_to_label=id_to_label,
            normalization_name=NORMALIZATION_NAME,
            print_examples=WER_EXAMPLE_PRINT_COUNT,
        )
        save_split_results(test_wer_df, split_name=f"test_{test_user}", results_dir=results_dir)

    val_wer_df = pd.DataFrame()
    if split_data["dev_val_wer_catalog"]:
        val_wer_df = evaluate_lstm_wer(
            samples=split_data["dev_val_wer_catalog"],
            split_name="Dev val",
            model_obj=model,
            normalize_fn=NORMALIZE_FN,
            id_to_label=id_to_label,
            normalization_name=NORMALIZATION_NAME,
            print_examples=WER_EXAMPLE_PRINT_COUNT,
        )
        save_split_results(val_wer_df, split_name=f"val_{test_user}", results_dir=results_dir)

    # ── 7b. Streaming metrics (SHREC'21: DR, FPR, Jaccard) ──
    if split_data["test_wer_catalog"]:
        evaluate_streaming_metrics(
            samples=split_data["test_wer_catalog"],
            split_name=f"Test ({test_user})",
            model_obj=model,
            normalize_fn=NORMALIZE_FN,
            label_to_id=split_data["label_to_id"],
            id_to_label=id_to_label,
            num_classes=num_classes,
        )

    # ── 8. Save model ──
    test_mean_wer = (
        float(test_wer_df["wer"].mean()) if len(test_wer_df) else None
    )
    saved_path = save_unique_model(
        model_obj=model,
        best_val_acc=train_result["best_val_acc"],
        save_dir=save_dir,
        model_name=MODEL_NAME,
        info={
            "test_user": test_user,
            "dev_users": dev_users,
            "feat_dim": FEAT_DIM,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LSTM_LAYERS,
            "test_mean_wer": test_mean_wer,
            "best_val_f1": train_result["best_val_f1"],
        },
    )

    # ── 9. Training curves ──
    curves_dir = os.path.join(save_dir, "plots")
    os.makedirs(curves_dir, exist_ok=True)
    plot_training_curves(
        train_result,
        save_path=os.path.join(curves_dir, f"curves_test_{test_user}.png"),
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
        description="LSTM Sign Language Recognition — Modular Pipeline",
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
        "--comment", type=str, default="",
        help="Custom comment to print at the top of the log/run history.",
    )
    parser.add_argument(
        "--focal-loss", action="store_true",
        help="Use Focal Loss instead of CrossEntropyLoss.",
    )
    parser.add_argument(
        "--focal-gamma", type=float, default=2.0,
        help="Gamma parameter for Focal Loss (default: 2.0).",
    )

    args = parser.parse_args()

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
        print(f"  Dataset root      : {DATASET_ROOT}")
        print(f"  Seed              : {SEED}")
        print(f"  Dev-Val Ratio     : {DEV_VAL_RATIO}")
        print(f"  Batch size        : {BATCH_SIZE}")
        print(f"  Test mode         : {args.test_mode}")
        print(f"  Model name        : {MODEL_NAME}")
        print(f"  Normalization     : {NORMALIZATION_NAME}")
        print(f"  Learning rate     : {args.lr}")
        print(f"  Epochs            : {args.epochs}")
        print(f"  Feature dim       : {FEAT_DIM}")
        print(f"  Hidden size       : {HIDDEN_SIZE}")
        print(f"  Num layers        : {NUM_LSTM_LAYERS}")
        print(f"  Dropout           : {DROPOUT}")
        print(f"  Decoder bag size  : {BAG_SIZE}")
        print(f"  Decoder threshold : {CONFIDENCE_THRESHOLD}")
        print(f"  Loss function     : {'FocalLoss (gamma=' + str(args.focal_gamma) + ')' if args.focal_loss else 'CrossEntropyLoss'}")
        print(f"==================================================\n")

        print(f"Using device: {DEVICE}")
        print(f"Dataset root: {DATASET_ROOT}")

        # ── Load all data once ──
        _, segments_by_user = load_all_segments(DATASET_ROOT)
        
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
                print(f"\nLOUO Mean WER:     {np.mean(wers):.4f} ± {np.std(wers):.4f}")
                print(f"LOUO Mean Val Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
                print(f"LOUO Mean Val F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
