"""
Prints THCT-Net's architecture and parameter count, instantiated with the
exact same defaults used during training (see main.py's THCTNet(...) call),
then runs one dummy forward pass to confirm the output shape.
"""
import torch

from config import (
    BASE_CH,
    BATCH_SIZE,
    D_MODEL,
    DEVICE,
    INPUT_DIM,
    NUM_HEADS,
    NUM_TRANSFORMER_LAYERS,
    WINDOW_SIZE,
)
from model import THCTNet

NUM_CLASSES = 21  # 20 signs + background, matches id_to_label used elsewhere

model = THCTNet(
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_TRANSFORMER_LAYERS,
    base_ch=BASE_CH,
    window_size=WINDOW_SIZE,
).to(DEVICE)

print("=" * 70)
print("THCT-Net architecture (training defaults)")
print("=" * 70)
print(model)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print()
print(f"Device            : {DEVICE}")
print(f"num_classes       : {NUM_CLASSES}")
print(f"d_model           : {D_MODEL}")
print(f"num_heads         : {NUM_HEADS}")
print(f"num_layers        : {NUM_TRANSFORMER_LAYERS}")
print(f"base_ch           : {BASE_CH}")
print(f"window_size       : {WINDOW_SIZE}")
print(f"input_dim         : {INPUT_DIM}")
print(f"Trainable params  : {trainable:,}")
print(f"Total params      : {total:,}")

model.eval()
sequences = torch.randn(BATCH_SIZE, WINDOW_SIZE, INPUT_DIM, device=DEVICE)
lengths = torch.full((BATCH_SIZE,), WINDOW_SIZE, dtype=torch.long, device=DEVICE)

with torch.no_grad():
    logits = model(sequences, lengths)

print()
print(f"Dummy input shape : {tuple(sequences.shape)}  (B, T, D)")
print(f"Output shape      : {tuple(logits.shape)}  (B, num_classes)")
assert logits.shape == (BATCH_SIZE, NUM_CLASSES), f"Unexpected output shape: {logits.shape}"
print("OK: forward pass shape matches expectations.")
