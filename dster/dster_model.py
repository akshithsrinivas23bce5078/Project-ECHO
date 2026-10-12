"""
Project ECHO — DSTER Model (Dual-Stream Transformer Fusion)
=============================================================
Fuses the Temporal-over-Channel and Channel-over-Temporal streams
into a unified emotion embedding.

Architecture:
    Stream 1 (Temporal-over-Channel) → embedding_1
    Stream 2 (Channel-over-Temporal) → embedding_2
    Concatenate → [embedding_1 ; embedding_2]
    → Fusion MLP → unified embedding
    → Classification head (see emotion_classifier.py)

Phase 1: GRE encoding replaces raw linear projections.
Phase 2: Optional CNSA multi-scale spectral attention and
         TaRoPE task-aware position encoding.

The dual-stream design captures BOTH:
    1. Rhythmic typing patterns within each feature over time (Stream 1)
    2. Cross-feature relationships at each moment (Stream 2)

This allows the model to detect emotions from keystroke dynamics alone,
without optical or auditory sensors.
"""

import torch
import torch.nn as nn
from typing import Optional
from dster.temporal_over_channel import TemporalOverChannelStream
from dster.channel_over_temporal import ChannelOverTemporalStream
from dster.cnsa import CrossNeedletSpectralAttention
from dster.tarope import TaRoPE


class DSTERModel(nn.Module):
    """
    Dual-Stream Transformer-based Emotion Recognition.

    Fuses two parallel attention streams into a single emotion embedding.

    Phase 1 Upgrade: Both streams now use Gaussian Range Encoding (GRE)
    to create continuous probabilistic representations of discrete
    timing data before Transformer processing.

    Phase 2 Upgrade: Optional CNSA for multi-scale temporal analysis
    and TaRoPE for task-context-aware position encoding.
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
        enable_cnsa: bool = False,
        cnsa_bands: int = 4,
        enable_tarope: bool = False,
        num_task_contexts: int = 8,
    ):
        super().__init__()

        self.enable_cnsa = enable_cnsa
        self.enable_tarope = enable_tarope

        # Stream 1: Temporal-over-Channel
        self.stream_temporal = TemporalOverChannelStream(
            num_channels=num_channels,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_gaussians=num_gaussians,
        )

        # Stream 2: Channel-over-Temporal
        self.stream_channel = ChannelOverTemporalStream(
            num_channels=num_channels,
            seq_len=seq_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_gaussians=num_gaussians,
        )

        # Phase 2: Cross-Needlet Spectral Attention (optional)
        self.cnsa = None
        if enable_cnsa:
            self.cnsa = CrossNeedletSpectralAttention(
                dim=embed_dim,
                num_bands=cnsa_bands,
                num_heads=num_heads,
                dropout=dropout,
            )

        # Phase 2: Task-Aware Rotary Position Embeddings (optional)
        self.tarope = None
        if enable_tarope:
            self.tarope = TaRoPE(
                dim=embed_dim,
                max_seq_len=seq_len * 2,
                num_task_contexts=num_task_contexts,
            )

        # Fusion: concatenated streams → unified embedding
        fused_dim = self.stream_temporal.output_dim + self.stream_channel.output_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        self.output_dim = embed_dim

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_channels) — raw keystroke features.
            task_id: (batch,) — optional task context for TaRoPE.

        Returns:
            (batch, embed_dim) — unified emotion embedding.
        """
        # Phase 2: Apply TaRoPE to input if enabled
        if self.tarope is not None and x.shape[-1] == self.tarope.dim:
            x = self.tarope(x, task_id=task_id)

        # Parallel stream processing
        emb_temporal = self.stream_temporal(x)   # (batch, C * E)
        emb_channel = self.stream_channel(x)     # (batch, C * E)

        # Concatenate and fuse
        fused = torch.cat([emb_temporal, emb_channel], dim=-1)
        embedding = self.fusion(fused)  # (batch, embed_dim)

        # Phase 2: Apply CNSA for multi-scale analysis if enabled
        if self.cnsa is not None:
            # Expand embedding to sequence for CNSA processing
            emb_seq = embedding.unsqueeze(1)  # (batch, 1, embed_dim)
            emb_seq = self.cnsa(emb_seq)
            embedding = emb_seq.squeeze(1)

        return embedding
