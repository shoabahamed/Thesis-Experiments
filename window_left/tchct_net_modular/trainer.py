"""
Training loop and per-epoch evaluation for the THCT-Net model.

Provides:
  - evaluate_model    : compute loss + accuracy + macro-F1 on a DataLoader
  - train_model       : full training loop with early stopping, scheduling
"""
from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_NAME,
    NORMALIZATION_NAME,
)


def evaluate_model(
    model_obj: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
) -> tuple[float, float, float]:
    """Evaluate model on a DataLoader. Returns (accuracy, avg_loss, macro_f1)."""
    model_obj.eval()
    total_loss = 0.0
    total_correct = 0
    total_frames = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for sequences, labels, lengths in loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)

            logits = model_obj(sequences, lengths)
            flat_logits = logits.reshape(-1, num_classes)
            flat_labels = labels.reshape(-1)
            loss = criterion(flat_logits, flat_labels)
            total_loss += loss.item()

            mask = flat_labels != -1
            if mask.any():
                preds = torch.argmax(flat_logits, dim=1)
                total_correct += (preds[mask] == flat_labels[mask]).sum().item()
                total_frames += mask.sum().item()
                all_preds.extend(preds[mask].cpu().tolist())
                all_targets.extend(flat_labels[mask].cpu().tolist())

    if total_frames == 0:
        return 0.0, 0.0, 0.0

    acc = total_correct / total_frames
    avg_loss = total_loss / max(1, len(loader))
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return float(acc), float(avg_loss), float(macro_f1)


def train_model(
    model_obj: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    use_grad_clip: bool = True,
    grad_clip_norm: float = 1.0,
    use_scheduler: bool = True,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 15,
    model_name: str = MODEL_NAME,
    normalization_name: str = NORMALIZATION_NAME,
) -> dict:
    """Full training loop with early stopping and LR scheduling.

    Best model is selected by **validation macro-F1** (more robust than
    accuracy when background frames dominate the dataset).

    Returns
    -------
    dict with training results including best_model_state, history, etc.
    """
    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty.")

    optimizer = torch.optim.Adam(model_obj.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=scheduler_factor,
            patience=scheduler_patience,
        )

    best_val_f1 = -1.0
    best_val_acc_at_best_f1 = -1.0
    best_model_state = None
    epochs_without_improvement = 0

    history_train_loss = []
    history_train_acc = []
    history_train_batch_acc = []
    history_val_acc = []
    history_train_f1 = []
    history_val_f1 = []
    epoch_history = []

    print(f"[{model_name}] using normalization: {normalization_name}")
    print(
        f"Stabilization: grad_clip={use_grad_clip} (norm={grad_clip_norm}), "
        f"scheduler={use_scheduler}, early_stopping_patience={early_stopping_patience}"
    )
    print(f"Model selection criterion: validation macro-F1")

    for epoch in range(1, epochs + 1):
        model_obj.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_preds: list[int] = []
        epoch_targets: list[int] = []

        for sequences, labels, lengths in tqdm(
            train_loader,
            desc=f"{model_name} | Epoch {epoch}/{epochs}",
            leave=False,
        ):
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)
            lengths = lengths.to(DEVICE)

            optimizer.zero_grad()
            logits = model_obj(sequences, lengths)
            flat_logits = logits.reshape(-1, num_classes)
            flat_labels = labels.reshape(-1)
            loss = criterion(flat_logits, flat_labels)
            loss.backward()

            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    model_obj.parameters(), max_norm=grad_clip_norm,
                )

            optimizer.step()

            running_loss += loss.item()
            mask = flat_labels != -1
            if mask.any():
                batch_preds = torch.argmax(flat_logits, dim=1)
                running_correct += (
                    (batch_preds[mask] == flat_labels[mask]).sum().item()
                )
                running_total += mask.sum().item()
                epoch_preds.extend(batch_preds[mask].detach().cpu().tolist())
                epoch_targets.extend(flat_labels[mask].detach().cpu().tolist())

        epoch_loss = running_loss / max(1, len(train_loader))
        history_train_loss.append(epoch_loss)

        epoch_train_batch_acc = running_correct / max(1, running_total)
        history_train_batch_acc.append(epoch_train_batch_acc)
        history_train_acc.append(float("nan"))

        epoch_train_f1 = f1_score(
            epoch_targets, epoch_preds, average="macro", zero_division=0,
        ) if epoch_targets else 0.0
        history_train_f1.append(float(epoch_train_f1))

        val_acc, _, val_f1 = evaluate_model(
            model_obj, val_loader, criterion, num_classes,
        )
        history_val_acc.append(val_acc)
        history_val_f1.append(val_f1)

        # LR scheduler tracks val macro-F1
        if scheduler is not None:
            scheduler.step(val_f1)

        # Best model selection by val macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc_at_best_f1 = val_acc
            best_model_state = {
                k: v.detach().clone() for k, v in model_obj.state_dict().items()
            }
            epochs_without_improvement = 0
            print(
                f"  -> [{model_name}] new best in memory "
                f"(val_f1={val_f1:.4f}, val_acc={val_acc:.4f})"
            )
        else:
            epochs_without_improvement += 1

        epoch_history.append({
            "model": model_name,
            "normalization": normalization_name,
            "epoch": epoch,
            "train_loss": float(epoch_loss),
            "train_batch_acc": float(epoch_train_batch_acc),
            "train_f1": float(epoch_train_f1),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
        })

        print(
            f"{model_name} | Epoch {epoch:02d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {epoch_train_batch_acc:.4f} | "
            f"Train F1: {epoch_train_f1:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement for {early_stopping_patience} epochs)."
            )
            break

    history_df_local = pd.DataFrame(epoch_history)
    print(f"\n{model_name} per-epoch summary:")
    print(history_df_local.tail(min(10, len(history_df_local))).to_string())

    if best_model_state is not None:
        model_obj.load_state_dict(best_model_state)
        print(f"[{model_name}] best model restored from memory.")

    return {
        "model_name": model_name,
        "normalization": normalization_name,
        "best_val_acc": float(best_val_acc_at_best_f1),
        "best_val_f1": float(best_val_f1),
        "best_model_state": best_model_state,
        "history_train_loss": history_train_loss,
        "history_train_acc": history_train_acc,
        "history_train_batch_acc": history_train_batch_acc,
        "history_val_acc": history_val_acc,
        "history_train_f1": history_train_f1,
        "history_val_f1": history_val_f1,
        "epoch_history": epoch_history,
    }
