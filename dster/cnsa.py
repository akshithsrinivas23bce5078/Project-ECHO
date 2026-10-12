"""
Project ECHO — Cross-Needlet Spectral Attention (CNSA)
========================================================
Phase 2: Multimodal Fusion

Implements multi-scale spectral analysis via needlet-inspired
decomposition. Captures behavioral patterns at multiple temporal
granularities simultaneously — from micro-hesitations (50ms) to
macro-trends (minutes) — using learnable spectral band filters.

Reference:
    Baldi, P. et al. (2009). Needlets and CMB Analysis.
    Adapted for temporal behavioral signal processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SpectralBandFilter(nn.Module):
    """
    Learnable band-pass filter for a specific temporal scale.

    Each filter isolates behavioral patterns within a frequency
    band via windowed DFT → learnable mask → inverse DFT.
    """

    def __init__(self, dim: int, center_freq: float, bandwidth: float):
        super().__init__()
        self.dim = dim
        self.center_freq = center_freq
        self.bandwidth = bandwidth

        # Learnable frequency mask parameters
        self.mask_weight = nn.Parameter(torch.ones(1))
        self.mask_sharpness = nn.Parameter(torch.tensor(5.0))

        # Per-band projection
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply band-pass filtering in spectral domain.

        Args:
            x: (batch, seq_len, dim)

        Returns:
            Filtered signal (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape

        # DFT along sequence dimension
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, freq_bins, dim)
        n_freqs = x_fft.shape[1]

        # Create frequency axis [0, 1) normalized
        freq_axis = torch.linspace(0, 1, n_freqs, device=x.device)

        # Gaussian band-pass mask centered at center_freq
        mask = torch.exp(
            -self.mask_sharpness * (freq_axis - self.center_freq) ** 2
            / (2 * self.bandwidth ** 2 + 1e-6)
        )
        mask = mask * self.mask_weight
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, freq_bins, 1)

        # Apply mask
        x_filtered = x_fft * mask

        # Inverse DFT
        x_reconstructed = torch.fft.irfft(x_filtered, n=seq_len, dim=1)

        # Project and normalize
        out = self.proj(x_reconstructed)
        out = self.norm(out)

        return out


class CrossNeedletSpectralAttention(nn.Module):
    """
    Multi-scale spectral attention via needlet-inspired decomposition.

    Decomposes temporal sequences into K frequency bands, processes
    each band independently, and fuses via cross-scale attention.

    Parameters:
        dim: Feature dimension.
        num_bands: Number of spectral decomposition bands.
        num_heads: Attention heads for cross-scale fusion.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_bands: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_bands = num_bands

        # Create band filters at logarithmically spaced frequencies
        self.band_filters = nn.ModuleList()
        for k in range(num_bands):
            center = (k + 1) / (num_bands + 1)  # Evenly spaced in [0, 1]
            bw = 1.0 / (num_bands * 2)
            self.band_filters.append(
                SpectralBandFilter(dim, center_freq=center, bandwidth=bw)
            )

        # Cross-scale attention: each band attends to all others
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Band fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_bands, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

        # Residual gate
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale spectral decomposition and cross-scale fusion.

        Args:
            x: (batch, seq_len, dim)

        Returns:
            Fused multi-scale representation (batch, seq_len, dim)
        """
        batch, seq_len, dim = x.shape

        # Decompose into spectral bands
        band_outputs = []
        for band_filter in self.band_filters:
            filtered = band_filter(x)  # (batch, seq_len, dim)
            band_outputs.append(filtered)

        # Stack bands: (batch, num_bands * seq_len, dim)
        stacked = torch.cat(band_outputs, dim=1)

        # Cross-scale attention
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # Reshape back: (batch, num_bands, seq_len, dim)
        attended = attended.view(batch, self.num_bands, seq_len, dim)

        # Pool across bands per time step
        # Concat all band representations at each timestep
        band_concat = attended.permute(0, 2, 1, 3).reshape(
            batch, seq_len, self.num_bands * dim
        )

        # Fuse
        fused = self.fusion(band_concat)  # (batch, seq_len, dim)

        # Residual gating
        gate_input = torch.cat([x, fused], dim=-1)
        gate_weight = self.gate(gate_input)
        output = gate_weight * fused + (1 - gate_weight) * x

        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_bands={self.num_bands}"
