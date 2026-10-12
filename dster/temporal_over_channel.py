"""
Project ECHO — Temporal-over-Channel Stream (Stream 1)
=======================================================
Processes the rhythmic latency patterns of keystroke dynamics.

This stream treats each keystroke feature channel (press duration,
inter-key interval, flight time, etc.) as a spatial dimension, and
applies self-attention ACROSS TIME within each channel. This captures
the temporal rhythm of a user's typing — the cadence and flow.

Architecture:
    Input: (batch, seq_len, num_channels)
    → Permute to (batch * num_channels, seq_len, 1)
    → Positional encoding
    → Transformer encoder layers (attention over time)
    → Pool across time → (batch, num_channels, embed_dim)
    → Flatten → Stream 1 embedding
"""

import torch
import torch.nn as nn
import math

from dster.gaussian_range_encoding import GaussianRangeEncoder


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalOverChannelStream(nn.Module):
    """
    Stream 1: Attention over the temporal axis for each feature channel.
    Captures rhythmic keystroke latency patterns.
    """

    def __init__(
        self,
        num_channels: int = 4,
        seq_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_gaussians: int = 16,
    ):
        """
        Args:
            num_channels: Number of keystroke feature channels
                          (e.g., press_duration, iki, flight_time, hold_time).
            seq_len: Length of the input sequence (number of keystroke events).
            embed_dim: Transformer embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
            num_gaussians: Number of Gaussian basis functions for GRE.
        """
        super().__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Phase 1: Gaussian Range Encoding replaces raw Linear projection
        self.input_projection = GaussianRangeEncoder(
            num_gaussians=num_gaussians,
            embed_dim=embed_dim,
        )

        self.pos_encoding = PositionalEncoding(embed_dim, max_len=seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: one embedding per channel, flattened
        self.output_dim = num_channels * embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_channels) — raw keystroke features.

        Returns:
            (batch, num_channels * embed_dim) — Stream 1 embedding.
        """
        batch_size = x.size(0)

        # Permute to process each channel independently over time
        # (batch, seq_len, channels) → (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        # Reshape: (batch * channels, seq_len, 1)
        x = x.reshape(batch_size * self.num_channels, self.seq_len, 1)

        # Project to embedding dim
        x = self.input_projection(x)  # (B*C, seq_len, embed_dim)
        x = self.pos_encoding(x)

        # Temporal self-attention
        x = self.transformer(x)  # (B*C, seq_len, embed_dim)

        # Pool across time (mean pooling)
        x = x.mean(dim=1)  # (B*C, embed_dim)

        # Reshape back: (batch, channels * embed_dim)
        x = x.reshape(batch_size, self.num_channels * self.embed_dim)

        return x
