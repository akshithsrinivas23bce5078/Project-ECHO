"""
Project ECHO — Task-Aware Rotary Position Embeddings (TaRoPE)
================================================================
Phase 2: Multimodal Fusion

Extends standard Rotary Position Embeddings (RoPE) with task-context
conditioning. Instead of fixed sinusoidal frequency bases, TaRoPE
modulates the rotation frequencies based on the current game genre
and difficulty state, enabling the Transformer to adapt its temporal
attention patterns to different gameplay contexts.

Reference:
    Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary
    Position Embedding. arXiv:2104.09864.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class TaRoPE(nn.Module):
    """
    Task-Aware Rotary Position Embeddings.

    Standard RoPE applies fixed-frequency rotations to query/key vectors.
    TaRoPE adds a learnable task-context projection that modulates these
    frequencies, allowing the model to widen or narrow its temporal
    attention span depending on the game's pacing and complexity.

    Parameters:
        dim: Embedding dimension (must be even).
        max_seq_len: Maximum sequence length.
        num_task_contexts: Number of discrete task contexts (e.g., game genres).
        base: RoPE base frequency (default 10000.0).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        num_task_contexts: int = 8,
        base: float = 10000.0,
    ):
        super().__init__()
        assert dim % 2 == 0, f"TaRoPE dim must be even, got {dim}"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Standard RoPE inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Task-aware frequency modulation
        self.task_embedding = nn.Embedding(num_task_contexts, dim // 2)
        self.freq_gate = nn.Sequential(
            nn.Linear(dim // 2, dim // 2),
            nn.Sigmoid(),
        )

        # Pre-compute position indices
        self.register_buffer(
            "pos_idx", torch.arange(max_seq_len).float().unsqueeze(1)
        )

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply TaRoPE to input tensor.

        Args:
            x: (batch, seq_len, dim) — query or key tensor.
            task_id: (batch,) — integer task context IDs. If None,
                     uses standard RoPE without task modulation.

        Returns:
            Rotated tensor of same shape as input.
        """
        batch, seq_len, dim = x.shape
        assert dim == self.dim, f"Expected dim={self.dim}, got {dim}"

        # Base frequencies
        freqs = self.inv_freq  # (dim//2,)

        # Task-aware modulation
        if task_id is not None:
            task_emb = self.task_embedding(task_id)  # (batch, dim//2)
            gate = self.freq_gate(task_emb)  # (batch, dim//2)
            # Modulate: scale frequencies between 0.5x and 2x
            freqs = freqs.unsqueeze(0) * (0.5 + 1.5 * gate.unsqueeze(1))
            # freqs: (batch, 1, dim//2)
            pos = self.pos_idx[:seq_len]  # (seq_len, 1)
            angles = pos.unsqueeze(0) * freqs  # (batch, seq_len, dim//2)
        else:
            pos = self.pos_idx[:seq_len]  # (seq_len, 1)
            angles = pos * freqs  # (seq_len, dim//2)
            angles = angles.unsqueeze(0).expand(batch, -1, -1)

        # Split input into pairs and apply rotation
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices

        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # Rotary transformation
        out1 = x1 * cos_angles - x2 * sin_angles
        out2 = x1 * sin_angles + x2 * cos_angles

        # Interleave back
        out = torch.stack([out1, out2], dim=-1).flatten(-2)
        return out

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, max_seq_len={self.max_seq_len}, "
            f"base={self.base}, "
            f"num_task_contexts={self.task_embedding.num_embeddings}"
        )
