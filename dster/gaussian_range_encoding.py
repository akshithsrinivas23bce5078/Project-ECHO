"""
Project ECHO — Gaussian Range Encoding (GRE) Layer
=====================================================
Replaces raw linear projections in the DSTER dual-stream architecture
with a learned Gaussian basis function encoding.

For each raw timing value x, the GRE computes:
    GRE(x) = [exp(-(x - μ_1)² / (2σ_1²)), ..., exp(-(x - μ_K)² / (2σ_K²))]

where K = num_gaussians, and μ_k, σ_k are learnable parameters.

This smooths the discrete, highly variable timing data inherent in
human typing, creating a continuous probabilistic space that the
subsequent Transformer layers can process more efficiently.

The GRE output is projected to the target embedding dimension via
a learned linear layer.

Reference:
    Li, M. et al. (2023). Gaussian Range Encoding for keystroke
    dynamics in behavioral biometrics.
"""

import torch
import torch.nn as nn
import math


class GaussianRangeEncoder(nn.Module):
    """
    Gaussian Range Encoding layer for behavioral biometric features.

    Converts raw scalar timing values into a soft histogram-like
    representation using learned Gaussian basis functions, then
    projects to the target embedding dimension.

    Architecture:
        Input: (*, 1) — single scalar per position
        → Gaussian expansion: (*, num_gaussians)
        → Linear projection: (*, embed_dim)
        → LayerNorm + GELU activation
    """

    def __init__(
        self,
        num_gaussians: int = 16,
        embed_dim: int = 64,
        mu_init_range: tuple = (-2.0, 8.0),
        sigma_init: float = 0.5,
        learnable: bool = True,
    ):
        """
        Args:
            num_gaussians: Number of Gaussian basis functions (K).
            embed_dim: Output embedding dimension.
            mu_init_range: (min, max) for initial center positions.
                           Default covers log-scaled IKI range.
            sigma_init: Initial spread for all Gaussians.
            learnable: If True, μ and σ are trainable parameters.
        """
        super().__init__()
        self.num_gaussians = num_gaussians
        self.embed_dim = embed_dim

        # Initialize Gaussian centers uniformly across the expected range
        mu_values = torch.linspace(mu_init_range[0], mu_init_range[1], num_gaussians)
        sigma_values = torch.full((num_gaussians,), sigma_init)

        if learnable:
            self.mu = nn.Parameter(mu_values)
            self.sigma = nn.Parameter(sigma_values)
        else:
            self.register_buffer("mu", mu_values)
            self.register_buffer("sigma", sigma_values)

        # Project Gaussian activations to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(num_gaussians, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights with scaled Xavier."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight,
                    gain=1.0 / math.sqrt(3.0)
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian Range Encoding.

        Args:
            x: (*, 1) — raw scalar values (e.g., IKI, press duration).

        Returns:
            (*, embed_dim) — encoded representations.
        """
        # Ensure sigma is positive (use softplus to avoid negative spreads)
        sigma_pos = torch.nn.functional.softplus(self.sigma) + 1e-6

        # Expand x for broadcasting: (*, 1) → (*, K)
        # mu and sigma are (K,), broadcast over batch dims
        diff = x - self.mu  # (*, K) via broadcasting
        exponent = -(diff ** 2) / (2.0 * sigma_pos ** 2)
        gaussians = torch.exp(exponent)  # (*, K)

        # Project to embedding dimension
        encoded = self.projection(gaussians)  # (*, embed_dim)

        return encoded

    def extra_repr(self) -> str:
        return (
            f"num_gaussians={self.num_gaussians}, "
            f"embed_dim={self.embed_dim}, "
            f"learnable={isinstance(self.mu, nn.Parameter)}"
        )
