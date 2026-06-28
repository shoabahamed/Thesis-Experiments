"""
THCT-Net: Two-stream Hybrid CNN-Transformer Network
Causal, per-frame variant for 750-frame sequences.

Changes from the original (30-frame, single-label) version
───────────────────────────────────────────────────────────
1.  T_FRAMES 30 → 750.

2.  Left-causal throughout (strictly no future frame leakage):
      • BatchNorm replaced with CausalLayerNorm (LN over channel dim
        per spatial position / per frame). BatchNorm and GroupNorm with
        groups < C both compute statistics over the time axis and leak.
        LayerNorm(C) applied per-position is causal-safe.
      • Transformer stream: causal attention mask (upper-triangle = −∞)
        so token t attends only to tokens 0…t.
      • CNN temporal convolutions: manual left-padding only
        (pad_left = (kernel−1)×dilation, pad_right = 0).
      • Motion branch: backward diff  M_t = S_t − S_{t−1}  (pads the
        first frame with zero) rather than forward diff (leaks S_{t+1}).
      • Residual-fusion 7×1 kernel: causal left-pad on the time axis.
      • Transformer temporal-aggregation Conv1d: causal left-pad.

3.  Per-frame output (B, T, num_classes):
      • Transformer: token-wise linear head; no GAP over time.
      • CNN: strided downsampling replaced by dilated causal convolutions
        (keeps T intact); last spatial dim pooled per-frame.
      • Late-fusion: element-wise weighted sum → (B, T, num_classes).

Pipeline contract (unchanged from original)
───────────────────────────────────────────
  DataLoader yields : (sequences, labels, lengths)
    sequences : (B, W=750, D=132)   palm-reference normalised
    labels    : (B, T)              per-frame class index
    lengths   : (B,)                accepted, not used
  Model call        : logits = model(sequences, lengths)
  logits            : (B, T=750, num_classes)
  Typical loss      : criterion(logits.reshape(-1, C), labels.reshape(-1))
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import DEVICE

# ─────────────────────────────────────────────────────────────────────
# Skeleton constants
# ─────────────────────────────────────────────────────────────────────
T_FRAMES     = 750   # ← was 30
NUM_FEATURES = 132
M_ENTITIES   = 2
V_PER_ENT    = 22
C_IN         = 3
NUM_JOINTS   = V_PER_ENT * M_ENTITIES   # 44


# ─────────────────────────────────────────────────────────────────────
# Causal-safe normalisation
# ─────────────────────────────────────────────────────────────────────
class CausalLN2d(nn.Module):
    """
    Causal LayerNorm for 2-D feature maps (B, C, T, V).

    Normalises over the channel dimension *per (B, t, v) position* so
    no statistics cross frame boundaries → strictly causal.

    Equivalent to LayerNorm applied on the last dim after permuting C last.
    Works on any (B, C, T, …) tensor with arbitrary trailing spatial dims.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln  = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        # x : (B, C, T, …)
        # Move C to last, apply LN, move back
        x = x.permute(0, 2, 3, 1)     # (B, T, V, C)  — works for 4-D
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)     # (B, C, T, V)
        return x


class CausalLN3d(nn.Module):
    """
    Causal LayerNorm for 3-D feature maps (B, C, T, H, W).
    Normalises over C per (B, t, h, w) position.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        # x : (B, C, T, H, W)
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        return x


class CausalLN1d(nn.Module):
    """
    Causal LayerNorm for 1-D sequences (B, C, T).
    Normalises over C per (B, t) position.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        # x : (B, C, T)
        x = x.permute(0, 2, 1)   # (B, T, C)
        x = self.ln(x)
        x = x.permute(0, 2, 1)   # (B, C, T)
        return x


# ─────────────────────────────────────────────────────────────────────
# Causal padding helpers
# ─────────────────────────────────────────────────────────────────────
def _causal_pad1d(x: Tensor, kernel_size: int, dilation: int = 1) -> Tensor:
    """Left-pad (B, C, T) for a causal Conv1d. No right pad."""
    return F.pad(x, ((kernel_size - 1) * dilation, 0))


def _causal_pad2d_time(x: Tensor, kernel_t: int, dilation_t: int = 1) -> Tensor:
    """
    Left-pad (B, C, T, V) along the time (height) axis for a causal Conv2d.
    Spatial dim (last) is not padded.
    F.pad order: (last_left, last_right, T_left, T_right)
    """
    return F.pad(x, (0, 0, (kernel_t - 1) * dilation_t, 0))


# ─────────────────────────────────────────────────────────────────────
# Helper: reshape flat window → skeleton tensor
# ─────────────────────────────────────────────────────────────────────
def _to_skeleton_tensor(x: Tensor) -> Tensor:
    """(B, T, 132) → (B, 3, T, 22, 2)"""
    B, T, D = x.shape
    x = x.reshape(B, T, M_ENTITIES, V_PER_ENT, C_IN)
    return x.permute(0, 4, 1, 3, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────
# 1.  Causal ISATA Self-Attention Block
# ─────────────────────────────────────────────────────────────────────
class _CausalISATABlock(nn.Module):
    """
    ISATA block (paper eq. 6) with a causal upper-triangular mask so
    token at position t attends only to positions 0…t.

    LayerNorm (standard, on last dim) is causal-safe here because tokens
    are shaped (B, T, d) and LN normalises over d only.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        # V = X (no projection, per paper eq. 5)
        self.v_proj = nn.Identity()

        self.alpha = nn.Parameter(torch.ones(1))
        self.A     = nn.Parameter(torch.zeros(num_heads, 1, 1))

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        # Standard LayerNorm over the last (channel) dim: causal-safe
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, T, d_model)"""
        B, U, d = x.shape
        H, Dh   = self.num_heads, self.head_dim

        residual = x
        x_n = self.norm1(x)

        Q = self.q_proj(x_n).reshape(B, U, H, Dh).transpose(1, 2)
        K = self.k_proj(x_n).reshape(B, U, H, Dh).transpose(1, 2)
        V = x_n.reshape(B, U, H, Dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
        scores = self.alpha * torch.tanh(scores) + self.A

        # Causal mask: upper triangle → −∞
        mask = torch.triu(torch.full((U, U), float("-inf"), device=x.device),
                          diagonal=1)
        scores = scores + mask

        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, V).transpose(1, 2).reshape(B, U, d) + residual
        return out + self.ffn(self.norm2(out))


# ─────────────────────────────────────────────────────────────────────
# 2.  Transformer Stream  (causal, per-frame output)
# ─────────────────────────────────────────────────────────────────────
class TransformerStream(nn.Module):
    """
    Input  : (B, 3, T=750, V=22, M=2)
    Output : (B, T, num_classes)

    One token per frame: a Conv3d with kernel (1, V, M) collapses the
    spatial (joint) dimensions but not the time axis, giving T tokens.
    Causal ISATABlocks then perform temporal self-attention.
    A causal Conv1d aggregates local context before the per-frame head.
    """

    def __init__(
        self,
        num_classes: int,
        d_model:     int   = 128,
        num_heads:   int   = 4,
        num_layers:  int   = 4,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Spatial embedding (one token per frame, no temporal mixing)
        self.frame_embed = nn.Sequential(
            nn.Conv3d(C_IN, d_model,
                      kernel_size=(1, V_PER_ENT, M_ENTITIES),
                      stride=(1, V_PER_ENT, M_ENTITIES), bias=False),
            CausalLN3d(d_model),   # safe: normalises over d per (B,t,1,1)
            nn.GELU(),
        )

        self.pos_enc = nn.Parameter(torch.zeros(1, T_FRAMES, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        self.blocks = nn.ModuleList([
            _CausalISATABlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Causal local temporal aggregation (Conv1d with left-pad only)
        self.temporal_agg = nn.Conv1d(d_model, d_model, kernel_size=5, bias=False)
        self.agg_norm     = CausalLN1d(d_model)

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """(B, 3, T, V=22, M=2) → (B, T, num_classes)"""
        B, C, T, V, M = x.shape

        tokens = self.frame_embed(x)              # (B, d, T, 1, 1)
        tokens = tokens.squeeze(-1).squeeze(-1)   # (B, d, T)
        tokens = tokens.transpose(1, 2)           # (B, T, d)
        tokens = tokens + self.pos_enc[:, :T, :]

        for blk in self.blocks:
            tokens = blk(tokens)                  # (B, T, d)

        # Causal temporal aggregation
        t = tokens.transpose(1, 2)                # (B, d, T)
        t = _causal_pad1d(t, kernel_size=5)       # (B, d, T+4)
        t = self.temporal_agg(t)                  # (B, d, T)
        t = F.gelu(self.agg_norm(t))
        tokens = t.transpose(1, 2)                # (B, T, d)

        return self.head(tokens)                  # (B, T, num_classes)


# ─────────────────────────────────────────────────────────────────────
# 3.  CNN Stream  (causal, per-frame output)
# ─────────────────────────────────────────────────────────────────────
class _CausalCNNBranch(nn.Module):
    """
    Single CNN branch — causal and stride-free in time.

    • All temporal convolutions use manual left-only padding.
    • Strided downsampling replaced with dilated causal convolutions
      (dilation 1 and 2) to preserve T.
    • CausalLN2d replaces BatchNorm2d.

    Input  : (B, C=3, T, 44)
    Output : (B, base_ch, T, base_ch)  — T unchanged
    """

    def __init__(self, base_ch: int = 64) -> None:
        super().__init__()

        # Stage 1: spatial 1×1 — no time axis involved, no padding needed
        self.enc1 = nn.Sequential(
            nn.Conv2d(C_IN, base_ch, kernel_size=(1, 1), bias=False),
            CausalLN2d(base_ch), nn.ReLU(inplace=True),
        )

        # Stage 2: causal temporal 3×1, dilation=1 → left-pad 2 on T
        self.enc2_conv = nn.Conv2d(base_ch, base_ch,
                                   kernel_size=(3, 1), padding=0, bias=False)
        self.enc2_norm = CausalLN2d(base_ch)

        # After joint→channel transpose: input is (B, NUM_JOINTS=44, T, base_ch)
        # Stage 3: causal dilated 3×3, dilation=(2,1) → left-pad 4 on T
        self.enc3_conv = nn.Conv2d(NUM_JOINTS, base_ch,
                                   kernel_size=(3, 3),
                                   padding=(0, 1), dilation=(2, 1), bias=False)
        self.enc3_norm = CausalLN2d(base_ch)

        # Stage 4: causal temporal 3×3, dilation=(1,1) → left-pad 2 on T
        self.enc4_conv = nn.Conv2d(base_ch, base_ch,
                                   kernel_size=(3, 3),
                                   padding=(0, 1), dilation=(1, 1), bias=False)
        self.enc4_norm = CausalLN2d(base_ch)

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, 3, T, 44)  →  (B, base_ch, T, base_ch)"""
        x = self.enc1(x)                                       # (B, ch, T, 44)

        x = _causal_pad2d_time(x, kernel_t=3, dilation_t=1)
        x = F.relu(self.enc2_norm(self.enc2_conv(x)))          # (B, ch, T, 44)

        x = x.permute(0, 3, 2, 1).contiguous()                # (B, 44, T, ch)

        x = _causal_pad2d_time(x, kernel_t=3, dilation_t=2)
        x = F.relu(self.enc3_norm(self.enc3_conv(x)))          # (B, ch, T, ch)

        x = _causal_pad2d_time(x, kernel_t=3, dilation_t=1)
        x = F.relu(self.enc4_norm(self.enc4_conv(x)))          # (B, ch, T, ch)
        return x


class _CausalResidualFusion(nn.Module):
    """
    Asymmetric 1×7 / 7×1 residual fusion (paper Section III-B).

    • 1×7 (spatial axis): symmetric padding — no temporal coupling.
    • 7×1 (time axis): causal left-pad so frame t sees only frames ≤ t.
    • CausalLN2d replaces BatchNorm2d throughout.
    """

    def __init__(self, in_ch: int, out_ch: int = 128) -> None:
        super().__init__()
        self.conv_s = nn.Conv2d(in_ch, out_ch,
                                kernel_size=(1, 7), padding=(0, 3), bias=False)
        self.norm_s = CausalLN2d(out_ch)

        self.conv_t = nn.Conv2d(out_ch, out_ch,
                                kernel_size=(7, 1), padding=0, bias=False)
        self.norm_t = CausalLN2d(out_ch)

        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                CausalLN2d(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        skip = self.shortcut(x)
        out  = F.relu(self.norm_s(self.conv_s(x)))      # spatial conv — safe
        out  = _causal_pad2d_time(out, kernel_t=7)      # causal pad for 7×1
        out  = self.norm_t(self.conv_t(out))
        return self.act(out + skip)


class CNNStream(nn.Module):
    """
    Causal CNN stream for per-frame prediction.

    Input  : (B, 3, T=750, V=22, M=2)
    Output : (B, T, num_classes)
    """

    def __init__(self, num_classes: int, base_ch: int = 64) -> None:
        super().__init__()
        self.branch_S = _CausalCNNBranch(base_ch)   # raw coordinates
        self.branch_M = _CausalCNNBranch(base_ch)   # causal motion features

        self.fusion = _CausalResidualFusion(base_ch * 2, out_ch=128)

        self.fc1  = nn.Linear(128, 256)
        self.drop = nn.Dropout(0.3)
        self.fc2  = nn.Linear(256, num_classes)

    @staticmethod
    def _backward_diff(x: Tensor) -> Tensor:
        """
        Causal (backward) temporal difference:  M_t = S_t − S_{t−1}.
        Pads the first frame with zero — no future frame is accessed.
        The original used forward diff S_{t+1}−S_t which leaks one frame.
        """
        diff = x[:, :, 1:, :] - x[:, :, :-1, :]    # (B, C, T-1, V)
        return F.pad(diff, (0, 0, 1, 0))             # → (B, C, T, V)

    def forward(self, x: Tensor) -> Tensor:
        """(B, 3, T, V=22, M=2) → (B, T, num_classes)"""
        B, C, T, V, M = x.shape

        x_flat   = x.reshape(B, C, T, V * M)          # (B, 3, T, 44)
        x_motion = self._backward_diff(x_flat)

        feat_S = self.branch_S(x_flat)                # (B, ch, T, ch)
        feat_M = self.branch_M(x_motion)              # (B, ch, T, ch)

        if feat_S.shape != feat_M.shape:
            feat_M = F.interpolate(feat_M, size=feat_S.shape[2:])

        fused = torch.cat([feat_S, feat_M], dim=1)    # (B, 2ch, T, ch)
        fused = self.fusion(fused)                    # (B, 128, T, ch)

        out = fused.mean(dim=-1).transpose(1, 2)      # (B, T, 128)
        out = self.drop(F.relu(self.fc1(out)))
        return self.fc2(out)                          # (B, T, num_classes)


# ─────────────────────────────────────────────────────────────────────
# 4.  THCT-Net  (top-level)
# ─────────────────────────────────────────────────────────────────────
class THCTNet(nn.Module):
    """
    Two-stream Hybrid CNN-Transformer Network — causal, per-frame edition.

    Call signature:  logits = model(sequences, lengths)
      sequences : (B, T=750, D=132)  palm-normalised flat features
      lengths   : (B,)               accepted, not used
    Returns     : (B, T=750, num_classes)

    Every output position t is computed strictly from frames 0…t.

    Parameters
    ----------
    num_classes : int
    d_model     : int   Transformer hidden dim  (default 128)
    num_heads   : int   Attention heads          (default 4)
    num_layers  : int   Transformer blocks       (default 4)
    base_ch     : int   CNN base channels        (default 64)
    """

    def __init__(
        self,
        num_classes: int,
        d_model:     int = 128,
        num_heads:   int = 4,
        num_layers:  int = 4,
        base_ch:     int = 64,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cnn_stream   = CNNStream(num_classes, base_ch)
        self.trans_stream = TransformerStream(
            num_classes, d_model, num_heads, num_layers
        )
        # Sigmoid-gated fusion weight; initialised at 0.5
        self._raw_w = nn.Parameter(torch.zeros(1))

    def _fusion_weight(self) -> Tensor:
        return torch.sigmoid(self._raw_w)

    def forward(
        self, sequences: Tensor, lengths: Optional[Tensor] = None
    ) -> Tensor:
        """
        sequences : (B, T, D=132)
        returns   : (B, T, num_classes)
        """
        x = _to_skeleton_tensor(sequences)         # (B, 3, T, 22, 2)
        logits_cnn   = self.cnn_stream(x)          # (B, T, num_classes)
        logits_trans = self.trans_stream(x)        # (B, T, num_classes)
        w = self._fusion_weight()
        return w * logits_cnn + (1.0 - w) * logits_trans

    def forward_streams(self, sequences: Tensor):
        x = _to_skeleton_tensor(sequences)
        return self.cnn_stream(x), self.trans_stream(x)


# ─────────────────────────────────────────────────────────────────────
# 5.  Sanity check  (python model.py)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("THCT-Net (causal, per-frame, 750 frames)")
    print("=" * 60)

    NUM_CLASSES = 21
    B = 2
    print(f"Device : {DEVICE}")

    model = THCTNet(num_classes=NUM_CLASSES).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n_p:,}")

    sequences = torch.randn(B, T_FRAMES, NUM_FEATURES).to(DEVICE)
    lengths   = torch.full((B,), T_FRAMES, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(sequences, lengths)

    assert logits.shape == (B, T_FRAMES, NUM_CLASSES), f"Bad shape: {logits.shape}"
    print(f"Input  : {tuple(sequences.shape)}     (B, T, D)           ✓")
    print(f"Output : {tuple(logits.shape)}  (B, T, num_classes) ✓")

    cnn_out, tr_out = model.forward_streams(sequences)
    assert cnn_out.shape == (B, T_FRAMES, NUM_CLASSES)
    assert tr_out.shape  == (B, T_FRAMES, NUM_CLASSES)
    print(f"CNN stream    : {tuple(cnn_out.shape)} ✓")
    print(f"Trans stream  : {tuple(tr_out.shape)} ✓")
    print(f"Fusion weight : {torch.sigmoid(model._raw_w).item():.4f}  (learnable)")

    # ── Causal leak check ─────────────────────────────────────────────
    # Corrupt frames after PROBE by 1e4. Output at frames ≤ PROBE must
    # be bit-identical (max diff < 1e-4).
    print("\nRunning causal leak check …")
    model.eval()
    seq_a = torch.randn(1, T_FRAMES, NUM_FEATURES).to(DEVICE)
    seq_b = seq_a.clone()
    PROBE = T_FRAMES // 2
    seq_b[:, PROBE + 1:, :] += 1e4

    with torch.no_grad():
        out_a = model(seq_a)
        out_b = model(seq_b)

    max_past   = (out_a[:, :PROBE + 1] - out_b[:, :PROBE + 1]).abs().max().item()
    max_future = (out_a[:, PROBE + 1:] - out_b[:, PROBE + 1:]).abs().max().item()

    print(f"  Max |Δ| frames ≤ {PROBE} (must be 0) : {max_past:.2e}")
    print(f"  Max |Δ| frames > {PROBE} (must be >0): {max_future:.2e}")
    assert max_past < 1e-4, "CAUSAL LEAK DETECTED — future frames affect past output!"
    assert max_future > 0,  "Future frames should differ after perturbation."
    print("  Causal check PASSED ✓\n")

    print("Plug into your notebook:")
    print("  model = THCTNet(num_classes=<N>)")
    print("  logits = model(sequences, lengths)   # (B, T, num_classes)")
    print("  loss = criterion(logits.reshape(-1, N), labels.reshape(-1))")
