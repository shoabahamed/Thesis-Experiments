"""
Measure parameter count and GPU forward latency of the THCT-Net model
exactly as it is constructed for training/inference in main.py.

Usage:
    python measure_latency.py
    python measure_latency.py --num-classes 21

Latency is measured at batch size 1 — one forward pass over a single sliding
window (WINDOW_SIZE frames, INPUT_DIM features) in eval mode under
torch.no_grad(), with GPU synchronization around each timed iteration —
i.e. the per-window cost paid by the streaming decoder at every frame
(stride 1).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from config import (
    BASE_CH,
    D_MODEL,
    DEVICE,
    INPUT_DIM,
    NUM_HEADS,
    NUM_TRANSFORMER_LAYERS,
    WINDOW_SIZE,
)
from model import THCTNet


def build_training_model(num_classes: int) -> THCTNet:
    """Construct THCTNet with the identical arguments used in main.py."""
    return THCTNet(
        num_classes=num_classes,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_TRANSFORMER_LAYERS,
        base_ch=BASE_CH,
        window_size=WINDOW_SIZE,
    ).to(DEVICE)


def main() -> None:
    parser = argparse.ArgumentParser(description="THCT-Net parameter/latency benchmark")
    parser.add_argument(
        "--num-classes", type=int, default=21,
        help="Number of classes incl. background (default: 21, as in the LOUO runs).",
    )
    parser.add_argument(
        "--warmup", type=int, default=50,
        help="Warmup iterations before timing (default: 50).",
    )
    parser.add_argument(
        "--iters", type=int, default=300,
        help="Timed iterations (default: 300).",
    )
    args = parser.parse_args()

    print(f"Device            : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU               : {torch.cuda.get_device_name(DEVICE)}")

    model = build_training_model(args.num_classes)
    model.eval()

    # ── Parameters ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    print(f"\n--- Model (as built in main.py) ---")
    print(f"num_classes       : {args.num_classes}")
    print(f"window_size       : {WINDOW_SIZE} frames | input dim: {INPUT_DIM}")
    print(f"d_model/heads/lyrs: {D_MODEL}/{NUM_HEADS}/{NUM_TRANSFORMER_LAYERS} | base_ch: {BASE_CH}")
    print(f"Total parameters  : {total_params:,}")
    print(f"Trainable params  : {trainable_params:,}")
    print(f"Parameter size    : {size_mb:.2f} MB (fp32)")

    # ── Latency (batch size 1 — the streaming case) ──
    B = 1
    x = torch.randn(B, WINDOW_SIZE, INPUT_DIM, device=DEVICE)
    lengths = torch.full((B,), WINDOW_SIZE, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x, lengths)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)

    times_ms = np.empty(args.iters, dtype=np.float64)
    with torch.no_grad():
        for i in range(args.iters):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            t0 = time.perf_counter()
            model(x, lengths)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize(DEVICE)
            times_ms[i] = (time.perf_counter() - t0) * 1000.0

    fps = 1000.0 / times_ms.mean()
    print(f"\n--- Forward latency (batch=1, {args.iters} iters after {args.warmup} warmup) ---")
    print(f"Mean              : {times_ms.mean():.3f} ms")
    print(f"Std               : {times_ms.std(ddof=0):.3f} ms")
    print(f"Median (p50)      : {np.percentile(times_ms, 50):.3f} ms")
    print(f"p95               : {np.percentile(times_ms, 95):.3f} ms")
    print(f"Min / Max         : {times_ms.min():.3f} / {times_ms.max():.3f} ms")
    print(f"Throughput        : {fps:,.1f} windows/s")
    print(
        f"Real-time check   : one window per frame at 30 fps requires < 33.3 ms/window "
        f"-> {'PASS' if times_ms.mean() < 1000.0 / 30.0 else 'FAIL'}"
    )

    if DEVICE.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
        print(f"Peak GPU memory   : {mem_mb:.1f} MB")


if __name__ == "__main__":
    main()
