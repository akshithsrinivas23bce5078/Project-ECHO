"""
Project ECHO — Multimodal Fusion Tests (Phase 2)
===================================================
Tests for TaRoPE, CNSA, and AVT Cross-Modal Attention.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

from dster.tarope import TaRoPE
from dster.cnsa import CrossNeedletSpectralAttention, SpectralBandFilter
from dster.cross_modal_attention import (
    AVTCrossModalAttention, CrossModalAttentionBlock, ModalityEncoder,
)


class TestTaRoPE:
    """Tests for Task-Aware Rotary Position Embeddings."""

    def test_output_shape(self):
        rope = TaRoPE(dim=64, max_seq_len=100, num_task_contexts=4)
        x = torch.randn(4, 50, 64)
        out = rope(x)
        assert out.shape == (4, 50, 64)

    def test_task_conditioned_output(self):
        rope = TaRoPE(dim=64, max_seq_len=100, num_task_contexts=4)
        x = torch.randn(4, 50, 64)
        task_id = torch.tensor([0, 1, 2, 3])
        out = rope(x, task_id=task_id)
        assert out.shape == (4, 50, 64)

    def test_different_tasks_produce_different_outputs(self):
        rope = TaRoPE(dim=64, max_seq_len=100, num_task_contexts=4)
        x = torch.randn(1, 50, 64)
        x_dup = x.expand(2, -1, -1)
        tasks = torch.tensor([0, 1])
        out = rope(x_dup, task_id=tasks)
        assert not torch.allclose(out[0], out[1], atol=1e-5)

    def test_no_task_equals_standard_rope(self):
        rope = TaRoPE(dim=64, max_seq_len=100)
        x = torch.randn(2, 30, 64)
        out = rope(x, task_id=None)
        assert out.shape == (2, 30, 64)

    def test_gradient_flow(self):
        rope = TaRoPE(dim=32, max_seq_len=50, num_task_contexts=4)
        x = torch.randn(2, 20, 32, requires_grad=True)
        task_id = torch.tensor([0, 1])
        out = rope(x, task_id=task_id)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_single_sample(self):
        rope = TaRoPE(dim=32, max_seq_len=50)
        x = torch.randn(1, 10, 32)
        out = rope(x)
        assert out.shape == (1, 10, 32)


class TestCNSA:
    """Tests for Cross-Needlet Spectral Attention."""

    def test_output_shape(self):
        cnsa = CrossNeedletSpectralAttention(dim=64, num_bands=4, num_heads=4)
        x = torch.randn(4, 50, 64)
        out = cnsa(x)
        assert out.shape == (4, 50, 64)

    def test_single_band(self):
        cnsa = CrossNeedletSpectralAttention(dim=32, num_bands=1, num_heads=2)
        x = torch.randn(2, 20, 32)
        out = cnsa(x)
        assert out.shape == (2, 20, 32)

    def test_residual_connection(self):
        """Output should not be identical to input (fusion changes it)."""
        cnsa = CrossNeedletSpectralAttention(dim=32, num_bands=2, num_heads=2)
        x = torch.randn(2, 20, 32)
        out = cnsa(x)
        assert not torch.allclose(x, out, atol=1e-3)

    def test_gradient_flow(self):
        cnsa = CrossNeedletSpectralAttention(dim=32, num_bands=3, num_heads=2)
        x = torch.randn(2, 20, 32, requires_grad=True)
        out = cnsa(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_spectral_band_filter_shape(self):
        f = SpectralBandFilter(dim=32, center_freq=0.25, bandwidth=0.1)
        x = torch.randn(2, 50, 32)
        out = f(x)
        assert out.shape == (2, 50, 32)


class TestAVTCrossModalAttention:
    """Tests for Audio-Visual-Text Cross-Modal Attention."""

    def test_output_shape(self):
        avt = AVTCrossModalAttention(
            behavioral_dim=64, visual_dim=16, context_dim=8,
            embed_dim=64, num_heads=4,
        )
        behavioral = torch.randn(4, 50, 64)
        visual = torch.randn(4, 16)
        context = torch.randn(4, 8)
        out = avt(behavioral, visual, context)
        assert out.shape == (4, 64)

    def test_single_sample(self):
        avt = AVTCrossModalAttention(
            behavioral_dim=32, visual_dim=8, context_dim=4,
            embed_dim=32, num_heads=2,
        )
        behavioral = torch.randn(1, 20, 32)
        visual = torch.randn(1, 8)
        context = torch.randn(1, 4)
        out = avt(behavioral, visual, context)
        assert out.shape == (1, 32)

    def test_gradient_flow(self):
        avt = AVTCrossModalAttention(
            behavioral_dim=32, visual_dim=8, context_dim=4,
            embed_dim=32, num_heads=2,
        )
        b = torch.randn(2, 20, 32, requires_grad=True)
        v = torch.randn(2, 8, requires_grad=True)
        c = torch.randn(2, 4, requires_grad=True)
        out = avt(b, v, c)
        loss = out.sum()
        loss.backward()
        assert b.grad is not None
        assert v.grad is not None
        assert c.grad is not None

    def test_cross_modal_block_shapes(self):
        block = CrossModalAttentionBlock(embed_dim=32, num_heads=2)
        a = torch.randn(2, 10, 32)
        b = torch.randn(2, 5, 32)
        a_out, b_out = block(a, b)
        assert a_out.shape == (2, 10, 32)
        assert b_out.shape == (2, 5, 32)

    def test_modality_encoder_2d_input(self):
        enc = ModalityEncoder(input_dim=16, embed_dim=32)
        x = torch.randn(4, 16)
        out = enc(x)
        assert out.shape == (4, 1, 32)

    def test_modality_encoder_3d_input(self):
        enc = ModalityEncoder(input_dim=16, embed_dim=32)
        x = torch.randn(4, 10, 16)
        out = enc(x)
        assert out.shape == (4, 10, 32)

    def test_visual_as_sequence(self):
        """Visual input can also be a sequence (multi-frame)."""
        avt = AVTCrossModalAttention(
            behavioral_dim=32, visual_dim=8, context_dim=4,
            embed_dim=32, num_heads=2,
        )
        behavioral = torch.randn(2, 20, 32)
        visual = torch.randn(2, 5, 8)  # 5 frames
        context = torch.randn(2, 4)
        out = avt(behavioral, visual, context)
        assert out.shape == (2, 32)
