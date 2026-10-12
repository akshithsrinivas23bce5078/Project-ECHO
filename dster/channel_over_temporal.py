"""
Project ECHO — Channel-over-Temporal Stream (Stream 2)
=======================================================
Processes contextual relationships between keystroke feature channels.

This stream treats time steps as the spatial dimension and applies
self-attention ACROSS CHANNELS at each time step. This captures
cross-feature dependencies — e.g., the relationship between press
duration and inter-key interval at the same moment in time.

Architecture:
    Input: (batch, seq_len, num_channels)
    → Permute to (batch * seq_len, num_channels, 1)
    → Positional encoding (over channels)
    → Transformer encoder layers (attention over channels)
    → Pool across channels → (batch, seq_len, embed_dim)
    → Pool across time → Stream 2 embedding
"""

import torch
import torch.nn as nn
import math

from dster.gaussian_range_encoding import GaussianRangeEncoder


class ChannelPositionalEncoding(nn.Module):
    """Positional encoding for the channel dimension."""

    def __init__(self, d_model: int, max_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_channels, d_model)
        position = torch.arange(0, max_channels, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ChannelOverTemporalStream(nn.Module):
    """
    Stream 2: Attention over the channel axis at each time step.
    Captures contextual relationships between keystroke features.
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
            num_channels: Number of keystroke feature channels.
            seq_len: Length of input sequence.
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

        self.pos_encoding = ChannelPositionalEncoding(
            embed_dim, max_channels=num_channels, dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output dimension matches Stream 1 for fusion
        self.output_dim = num_channels * embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_channels) — raw keystroke features.

        Returns:
            (batch, num_channels * embed_dim) — Stream 2 embedding.
        """
        batch_size = x.size(0)

        # Reshape to process each time step independently over channels
        # (batch, seq_len, channels) → (batch * seq_len, channels, 1)
        x = x.reshape(batch_size * self.seq_len, self.num_channels, 1)

        # Project to embedding dim
        x = self.input_projection(x)  # (B*T, channels, embed_dim)
        x = self.pos_encoding(x)

        # Channel self-attention
        x = self.transformer(x)  # (B*T, channels, embed_dim)

        # Pool across channels
        x = x.mean(dim=1)  # (B*T, embed_dim)

        # Reshape: (batch, seq_len, embed_dim)
        x = x.reshape(batch_size, self.seq_len, self.embed_dim)

        # Pool across time
        x = x.mean(dim=1)  # (batch, embed_dim)

        # Expand to match Stream 1 output dimension
        x = x.unsqueeze(1).expand(-1, self.num_channels, -1)
        x = x.reshape(batch_size, self.num_channels * self.embed_dim)

        return x
