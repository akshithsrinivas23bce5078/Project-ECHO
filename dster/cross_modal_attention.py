"""
Project ECHO — Audio-Visual-Text Cross-Modal Attention (AVT-CA)
=================================================================
Phase 2: Multimodal Fusion

Implements cross-modal attention that enables each modality stream
(behavioral/keystroke, visual/webcam, contextual/game-state) to
attend to every other, creating a unified multimodal representation.

Architecture:
    Each modality produces an embedding sequence. AVT-CA applies
    pairwise cross-attention between all modality pairs, then
    fuses via gated aggregation to produce a single unified
    embedding that captures inter-modal correlations.

Example inter-modal correlations captured:
    - Keystroke hesitation + face tension → frustration
    - High mouse speed + high optical flow → flow state
    - Low keystrokes + chin contact → thinking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ModalityEncoder(nn.Module):
    """
    Lightweight encoder that projects a raw modality feature vector
    into a shared embedding space for cross-modal attention.
    """

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
        Returns:
            (batch, seq_len, embed_dim) or (batch, 1, embed_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.proj(x)


class CrossModalAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention between two modality streams.

    Given modality A and modality B:
        A_out = A + CrossAttn(Q=A, K=B, V=B)
        B_out = B + CrossAttn(Q=B, K=A, V=A)
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # A attends to B
        self.attn_a_to_b = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_a = nn.LayerNorm(embed_dim)

        # B attends to A
        self.attn_b_to_a = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_b = nn.LayerNorm(embed_dim)

    def forward(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            a: (batch, seq_a, dim) — modality A embeddings.
            b: (batch, seq_b, dim) — modality B embeddings.
        Returns:
            Tuple of cross-attended (a_out, b_out).
        """
        # A attends to B
        a_cross, _ = self.attn_a_to_b(a, b, b)
        a_out = self.norm_a(a + a_cross)

        # B attends to A
        b_cross, _ = self.attn_b_to_a(b, a, a)
        b_out = self.norm_b(b + b_cross)

        return a_out, b_out


class AVTCrossModalAttention(nn.Module):
    """
    Audio-Visual-Text Cross-Modal Attention Fusion.

    Fuses three modality streams via pairwise cross-attention:
        1. Behavioral (keystroke/mouse kinematics)
        2. Visual (webcam face/body features)
        3. Contextual (game state and environment)

    Parameters:
        behavioral_dim: Input dimension for behavioral features.
        visual_dim: Input dimension for visual features.
        context_dim: Input dimension for context features.
        embed_dim: Shared embedding dimension.
        num_heads: Attention heads per cross-attention block.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        behavioral_dim: int = 64,
        visual_dim: int = 16,
        context_dim: int = 8,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Per-modality encoders
        self.behavioral_encoder = ModalityEncoder(behavioral_dim, embed_dim, dropout)
        self.visual_encoder = ModalityEncoder(visual_dim, embed_dim, dropout)
        self.context_encoder = ModalityEncoder(context_dim, embed_dim, dropout)

        # Pairwise cross-attention blocks
        self.cross_bv = CrossModalAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_bc = CrossModalAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_vc = CrossModalAttentionBlock(embed_dim, num_heads, dropout)

        # Gated fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 3),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.output_dim = embed_dim

    def forward(
        self,
        behavioral: torch.Tensor,
        visual: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse three modality streams via pairwise cross-attention.

        Args:
            behavioral: (batch, seq_b, behavioral_dim) — DSTER embeddings.
            visual: (batch, visual_dim) or (batch, 1, visual_dim).
            context: (batch, context_dim) or (batch, 1, context_dim).

        Returns:
            Fused multimodal embedding (batch, embed_dim).
        """
        # Encode into shared space
        b_emb = self.behavioral_encoder(behavioral)  # (batch, seq_b, dim)
        v_emb = self.visual_encoder(visual)           # (batch, 1, dim)
        c_emb = self.context_encoder(context)         # (batch, 1, dim)

        # Pairwise cross-attention
        b_fused, v_fused = self.cross_bv(b_emb, v_emb)
        b_fused, c_fused = self.cross_bc(b_fused, c_emb)
        v_fused, c_fused = self.cross_vc(v_fused, c_fused)

        # Pool to single vectors
        b_pooled = b_fused.mean(dim=1)  # (batch, dim)
        v_pooled = v_fused.mean(dim=1)
        c_pooled = c_fused.mean(dim=1)

        # Gated fusion
        concat = torch.cat([b_pooled, v_pooled, c_pooled], dim=-1)
        gate = self.fusion_gate(concat)
        gated = concat * gate

        # Project to output
        fused = self.fusion_proj(gated)

        return fused

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}"
