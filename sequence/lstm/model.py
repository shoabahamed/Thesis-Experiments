"""
LSTM model for full-sequence sign language recognition.

Provides:
  - FullSequenceLSTM : frame encoder → unidirectional LSTM → per-frame classifier
                       Supports both batch forward() and online step() inference
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    DROPOUT,
    FEAT_DIM,
    HIDDEN_SIZE,
    INPUT_DIM,
    NUM_LSTM_LAYERS,
)


class FullSequenceLSTM(nn.Module):
    """Frame encoder → unidirectional LSTM → per-frame classifier."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        feat_dim: int = FEAT_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LSTM_LAYERS,
        num_classes: int = 21,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.frame_enc = nn.Sequential(
            nn.Linear(input_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def _encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        enc = self.frame_enc(x.reshape(b * t, d))
        return enc.reshape(b, t, self.feat_dim)

    def forward(
        self, sequences: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        enc = self._encode_frames(sequences)
        packed = nn.utils.rnn.pack_padded_sequence(
            enc, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        b, t, h = out.shape
        logits = self.classifier(out.reshape(b * t, h))
        return logits.reshape(b, t, self.num_classes)

    def step(self, frame: torch.Tensor, hidden=None):
        """Online inference: one normalized frame (1, 132) → logits (1, C)."""
        # Ensure shape is (batch=1, seq_len=1, features)
        if frame.ndim == 1:
            # (132,) → (1, 1, 132)
            frame = frame.unsqueeze(0).unsqueeze(0)
        elif frame.ndim == 2:
            # (1, 132) → (1, 1, 132)
            frame = frame.unsqueeze(1)
        # if already (1, 1, 132), pass through

        enc = self.frame_enc(frame)              # (1, 1, feat_dim)
        out, hidden = self.lstm(enc, hidden)     # out: (1, 1, hidden_size)
        logits = self.classifier(out[:, -1, :])  # (1, hidden_size) → (1, num_classes)
        return logits, hidden
