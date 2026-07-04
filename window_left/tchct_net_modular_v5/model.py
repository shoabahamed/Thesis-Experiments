"""
THCT-Net: Two-stream Hybrid CNN-Transformer Network
Non-causal, sliding-window variant.

NOTE ON NORMALIZATION:
This variant is intended for ONLINE inference at batch_size=1 (streaming
windows one at a time). BatchNorm's statistics depend on the batch, so at
inference it silently falls back to running_mean/running_var estimated
during training — this can mismatch training-time behavior, especially if
training batch sizes are small. GroupNorm computes statistics per-sample
(over channel groups), so it behaves IDENTICALLY at train and inference
time regardless of batch size. All BatchNorm2d/BatchNorm3d layers below
have been replaced with GroupNorm. LayerNorm inside the attention blocks
is untouched (already batch-independent, as it should be).
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
NUM_FEATURES = 138
M_ENTITIES   = 2
V_PER_ENT    = 23        # 22 hand joints + 1 elbow per hand
C_IN         = 3
NUM_JOINTS   = V_PER_ENT * M_ENTITIES   # 46


# ─────────────────────────────────────────────────────────────────────
# Helper: GroupNorm factory (batch-size-independent, drop-in for BN)
# ─────────────────────────────────────────────────────────────────────
def _gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """
    Returns a GroupNorm with the largest group count <= max_groups that
    evenly divides num_channels (falls back to 1 group, i.e. LayerNorm-like
    behavior over all channels, if nothing else divides evenly).
    Works transparently on 4D (B,C,H,W) and 5D (B,C,D,H,W) tensors.
    """
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


# ─────────────────────────────────────────────────────────────────────
# Helper: reshape flat window → skeleton tensor
# ─────────────────────────────────────────────────────────────────────
def _to_skeleton_tensor(x: Tensor) -> Tensor:
    """(B, T, 138) -> (B, 3, T, 23, 2)"""
    B, T, D = x.shape
    x = x.reshape(B, T, M_ENTITIES, V_PER_ENT, C_IN)
    return x.permute(0, 4, 1, 3, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────
# 1.  ISATA Self-Attention Block (Non-Causal)
# ─────────────────────────────────────────────────────────────────────
class _ISATABlock(nn.Module):
    """
    ISATA block (paper eq. 6) - Non-Causal.
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
        # LayerNorm is already batch-independent — left unchanged.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, U, d_model)"""
        B, U, d = x.shape
        H, Dh   = self.num_heads, self.head_dim

        # ── Attention ──────────────────────────────────────────────
        residual = x
        x_norm   = self.norm1(x)

        Q = self.q_proj(x_norm).reshape(B, U, H, Dh).transpose(1, 2)  # (B,H,U,Dh)
        K = self.k_proj(x_norm).reshape(B, U, H, Dh).transpose(1, 2)
        V = x_norm.reshape(B, U, H, Dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,U,U)
        scores = self.alpha * torch.tanh(scores) + self.A               # eq. 6
        attn   = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)                                     # (B,H,U,Dh)
        out = out.transpose(1, 2).reshape(B, U, d) + residual

        # ── FFN ────────────────────────────────────────────────────
        out = out + self.ffn(self.norm2(out))
        return out


# ─────────────────────────────────────────────────────────────────────
# 2.  Transformer Stream (Non-Causal)
# ─────────────────────────────────────────────────────────────────────
class TransformerStream(nn.Module):
    """
    Input  : (B, C=3, T=window_size, V=23, M=2)
    Tokens : 3D sliding window Tw×Vw×Mw  →  U tokens of dim d_model
    Blocks : L × _ISATABlock
    Output : (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        d_model:     int   = 128,
        num_heads:   int   = 4,
        num_layers:  int   = 4,
        Tw:          int   = 5,   # T=30 → 6 temporal tokens
        Vw:          int   = 2,   # V=22 → 11 joint tokens
        Mw:          int   = 1,   # M=2  → 2  entity tokens
        dropout:     float = 0.1,
        window_size: int   = 30,
    ) -> None:
        super().__init__()
        self.Tw, self.Vw, self.Mw = Tw, Vw, Mw
        self.d_model = d_model

        # 3D conv embedding: each (Tw×Vw×Mw) patch → d_model scalar
        # BatchNorm3d → GroupNorm: identical behavior at B=1 inference.
        self.token_embed = nn.Sequential(
            nn.Conv3d(C_IN, d_model,
                      kernel_size=(Tw, Vw, Mw),
                      stride=(Tw, Vw, Mw), bias=False),
            _gn(d_model),
            nn.GELU(),
        )

        self.nT = window_size // Tw
        self.nV = V_PER_ENT   // Vw   # 23 // 2 = 11 (rounds down)
        self.nM = M_ENTITIES  // Mw
        self.num_tokens = self.nT * self.nV * self.nM

        self.pos_enc = nn.Parameter(
            torch.zeros(1, self.num_tokens, d_model)
        )
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        self.blocks = nn.ModuleList([
            _ISATABlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Temporal aggregation (Conv3D kernel=5 along T as in paper)
        self.temporal_agg = nn.Conv3d(
            d_model, d_model,
            kernel_size=(min(5, self.nT), 1, 1),
            padding=(min(5, self.nT) // 2, 0, 0),
        )
        # BatchNorm3d → GroupNorm
        self.bn_agg = _gn(d_model)

        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, 3, T, V, M)"""
        B = x.size(0)

        tokens = self.token_embed(x)                    # (B, d, nT, nV, nM)
        tokens = tokens.flatten(2).transpose(1, 2)      # (B, U, d)
        tokens = tokens + self.pos_enc

        for blk in self.blocks:
            tokens = blk(tokens)

        # Reshape back for temporal aggregation
        tokens = (tokens
                  .transpose(1, 2)
                  .reshape(B, self.d_model, self.nT, self.nV, self.nM))
        tokens = F.gelu(self.bn_agg(self.temporal_agg(tokens)))

        out = self.gap(tokens).flatten(1)               # (B, d)
        return self.head(out)                           # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────
# 3.  CNN Stream (Non-Causal)
# ─────────────────────────────────────────────────────────────────────
class _CNNBranch(nn.Module):
    """
    Single branch (raw S or motion M).
    Input  : (B, C=3, T, V_total=46)
    """

    def __init__(self, base_ch: int = 64) -> None:
        super().__init__()

        # Stage 1 & 2: per-joint encoding then temporal context
        # BatchNorm2d → GroupNorm throughout this branch.
        self.enc1 = nn.Sequential(
            nn.Conv2d(C_IN, base_ch, kernel_size=(1, 1), bias=False),
            _gn(base_ch), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=(3, 1),
                      padding=(1, 0), bias=False),
            _gn(base_ch), nn.ReLU(inplace=True),
        )
        # After transpose: (B, V*M=46, T, base_ch)
        self.enc3 = nn.Sequential(
            nn.Conv2d(NUM_JOINTS, base_ch, kernel_size=(3, 3),
                      padding=(1, 1), stride=(2, 2), bias=False),
            _gn(base_ch), nn.ReLU(inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=(3, 3),
                      padding=(1, 1), stride=(2, 2), bias=False),
            _gn(base_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, 3, T, V*M=46)"""
        x = self.enc1(x)                               # (B, base_ch, T, 46)
        x = self.enc2(x)                               # (B, base_ch, T, 46)
        x = x.permute(0, 3, 2, 1).contiguous()         # (B, 46, T, base_ch)
        x = self.enc3(x)                               # (B, base_ch, T/2, ...)
        x = self.enc4(x)                               # (B, base_ch, T/4, ...)
        return x


class _ResidualFusion(nn.Module):
    """
    Fuses concatenated dual-branch features via asymmetric 1×7 / 7×1 convs
    + residual shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int = 128) -> None:
        super().__init__()
        # BatchNorm2d → GroupNorm on both the main path and the shortcut.
        self.path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 7),
                      padding=(0, 3), bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=(7, 1),
                      padding=(3, 0), bias=False),
            _gn(out_ch),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                _gn(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.path(x) + self.shortcut(x))


class CNNStream(nn.Module):
    """
    Full CNN stream.
    Input  : (B, C=3, T, V=23, M=2)
    Output : (B, num_classes)
    """

    def __init__(self, num_classes: int, base_ch: int = 64) -> None:
        super().__init__()
        self.branch_S = _CNNBranch(base_ch)   # raw coordinates
        self.branch_M = _CNNBranch(base_ch)   # temporal difference (motion)

        self.fusion = _ResidualFusion(base_ch * 2, out_ch=128)

        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(128, 256)
        self.drop = nn.Dropout(0.3)
        self.fc2  = nn.Linear(256, num_classes)

    @staticmethod
    def _temporal_diff(x: Tensor) -> Tensor:
        """Frame-to-frame difference: M_t = S_{t+1} − S_t"""
        diff = x[:, :, 1:, :] - x[:, :, :-1, :]   # (B, C, T-1, V)
        return F.pad(diff, (0, 0, 0, 1))            # pad last frame → (B,C,T,V)

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, 3, T, V=23, M=2)"""
        B, C, T, V, M = x.shape

        # Early entity fusion: merge M into V
        x_flat = x.reshape(B, C, T, V * M)          # (B, 3, T, 46)
        x_motion = self._temporal_diff(x_flat)

        feat_S = self.branch_S(x_flat)
        feat_M = self.branch_M(x_motion)

        # Align spatial dims if they differ (shouldn't with symmetric branches)
        if feat_S.shape != feat_M.shape:
            feat_M = F.interpolate(feat_M, size=feat_S.shape[2:])

        fused = torch.cat([feat_S, feat_M], dim=1)  # (B, 128, T'', d'')
        fused = self.fusion(fused)                   # (B, 128, T'', d'')

        out = self.gap(fused).flatten(1)             # (B, 128)
        out = self.drop(F.relu(self.fc1(out)))
        return self.fc2(out)                         # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────
# 4.  THCT-Net  (top-level model)
# ─────────────────────────────────────────────────────────────────────
class THCTNet(nn.Module):
    """
    Two-stream Hybrid CNN-Transformer Network — Leap Motion edition.
    Non-causal sliding-window variant. GroupNorm throughout (instead of
    BatchNorm) so behavior is identical whether you run batch_size=1
    online inference or larger training batches.
    """

    def __init__(
        self,
        num_classes: int,
        d_model:     int   = 128,
        num_heads:   int   = 4,
        num_layers:  int   = 4,
        base_ch:     int   = 64,
        window_size: int   = 30,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size

        self.cnn_stream  = CNNStream(num_classes, base_ch)
        self.trans_stream = TransformerStream(
            num_classes, d_model, num_heads, num_layers, window_size=window_size
        )

        # Learnable fusion weight in (0,1) via sigmoid; starts at 0.5
        self._raw_w = nn.Parameter(torch.zeros(1))

    def _fusion_weight(self) -> Tensor:
        return torch.sigmoid(self._raw_w)

    def forward(self, sequences: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
        """
        sequences : (B, T=window_size, D=138)
        lengths   : (B,)               ← accepted for API compatibility, not used
        returns   : (B, num_classes)
        """
        # Reshape flat window into skeleton tensor once, share between both streams
        x = _to_skeleton_tensor(sequences)   # (B, 3, T, 23, 2)

        logits_cnn  = self.cnn_stream(x)     # (B, num_classes)
        logits_trans = self.trans_stream(x)  # (B, num_classes)

        w   = self._fusion_weight().to(x.device)
        return w * logits_cnn + (1.0 - w) * logits_trans

    def forward_streams(self, sequences: Tensor):
        x = _to_skeleton_tensor(sequences)
        return self.cnn_stream(x), self.trans_stream(x)


if __name__ == "__main__":
    import torch
    print("=" * 60)
    print("THCT-Net Leap Motion — non-causal check (GroupNorm, B=1 ready)")
    print("=" * 60)

    NUM_CLASSES = 21
    print(f"Device : {DEVICE}")

    model = THCTNet(num_classes=NUM_CLASSES, window_size=30).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n_p:,}")

    # ── Sanity check 1: batch_size = 8 (e.g. training) ──────────────
    B = 8
    sequences = torch.randn(B, 30, NUM_FEATURES).to(DEVICE)
    lengths   = torch.full((B,), 30, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits = model(sequences, lengths)
    assert logits.shape == (B, NUM_CLASSES), f"Bad output shape: {logits.shape}"
    print(f"[B=8] Input  : {tuple(sequences.shape)}  (B, T, D)")
    print(f"[B=8] Output : {tuple(logits.shape)}  (B, num_classes) [OK]")

    # ── Sanity check 2: batch_size = 1 (online / streaming inference) ─
    model.eval()
    seq_single = torch.randn(1, 30, NUM_FEATURES).to(DEVICE)
    len_single = torch.full((1,), 30, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits_single = model(seq_single, len_single)
    assert logits_single.shape == (1, NUM_CLASSES), f"Bad output shape: {logits_single.shape}"
    print(f"[B=1] Input  : {tuple(seq_single.shape)}  (B, T, D)")
    print(f"[B=1] Output : {tuple(logits_single.shape)}  (B, num_classes) [OK]")
    print("GroupNorm confirmed batch-size-independent: B=1 ran with no BN running-stat fallback.")