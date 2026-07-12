import time
import torch
from config import (
    D_MODEL, NUM_HEADS, NUM_TRANSFORMER_LAYERS, BASE_CH, DEVICE, INPUT_DIM
)
from model import THCTNet

def main():
    print(f"Benchmarking THCT-Net on {DEVICE}...")
    
    # SHREC'21 typically has 14 gestures + 1 background class
    NUM_CLASSES = 15 
    
    # 1. Initialize model
    model = THCTNet(
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_TRANSFORMER_LAYERS,
        base_ch=BASE_CH,
    ).to(DEVICE)
    model.eval()

    # 2. Model Size & Parameters
    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"\n--- Model Size ---")
    print(f"Total Parameters : {param_count:,}")
    print(f"Model Size       : {param_size_mb:.2f} MB")

    # 3. Latency & Memory setup
    B, T, D = 1, 750, INPUT_DIM
    dummy_input = torch.randn(B, T, D, device=DEVICE)
    lengths = torch.tensor([T], dtype=torch.long, device=DEVICE)

    print(f"\n--- Performance (Batch=1, Frames={T}) ---")
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input, lengths)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure Latency
    runs = 100
    start_time = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(dummy_input, lengths)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_latency = (end_time - start_time) / runs
    fps = T / avg_latency
    
    print(f"Full Sequence Latency    : {avg_latency * 1000:.2f} ms")
    print(f"Average Latency per Frame: {(avg_latency / T) * 1000:.3f} ms")
    print(f"Throughput               : {fps:.2f} frames/sec")

    # Measure MACs / FLOPs (Optional)
    try:
        from thop import profile
        macs, params = profile(model, inputs=(dummy_input, lengths), verbose=False)
        print(f"MACs                     : {macs / 1e9:.2f} G")
    except ImportError:
        print("MACs                     : N/A (Run 'pip install thop' to see MACs)")

    # Measure GPU Memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad():
            _ = model(dummy_input, lengths)
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)
        print(f"\n--- GPU Memory ---")
        print(f"Peak VRAM usage          : {peak_mem:.2f} MB")
    else:
        print("\n--- Memory ---")
        print("GPU Memory profiling is only available on CUDA devices.")

if __name__ == "__main__":
    main()
