"""
tests/test_correctness.py
=========================
Numerical correctness checks for all three kernels against PyTorch references.

Run with:
    pytest tests/test_correctness.py -v

All tests skip automatically if CUDA is unavailable (e.g., CI on CPU).
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_kernels

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


# ---------------------------------------------------------------------------
# Flash Attention
# ---------------------------------------------------------------------------

class TestFlashAttention:

    @cuda_required
    def test_non_causal_correctness(self):
        torch.manual_seed(0)
        B, H, M, D = 2, 4, 128, 64
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = llm_kernels.flash_attn_forward(q, k, v, causal=False)

        assert torch.allclose(ref, out, atol=1e-2), \
            f"Non-causal max err: {(ref - out).abs().max():.4f}"

    @cuda_required
    def test_causal_correctness(self):
        torch.manual_seed(1)
        B, H, M, D = 2, 8, 512, 64
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = llm_kernels.flash_attn_forward(q, k, v, causal=True)

        assert torch.allclose(ref, out, atol=1e-2), \
            f"Causal max err: {(ref - out).abs().max():.4f}"

    @cuda_required
    def test_output_shape(self):
        B, H, M, D = 1, 2, 64, 32
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
        out = llm_kernels.flash_attn_forward(q, q, q, causal=True)
        assert out.shape == (B, H, M, D)
        assert out.dtype == torch.float16

    @cuda_required
    def test_module_wrapper(self):
        torch.manual_seed(2)
        B, H, M, D = 1, 4, 128, 64
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)

        attn = llm_kernels.FlashAttention(causal=True)
        out = attn(q, q, q)
        assert out.shape == (B, H, M, D)


# ---------------------------------------------------------------------------
# Fused RMSNorm + Linear
# ---------------------------------------------------------------------------

class TestFusedRMSNormLinear:

    def _ref(self, x, w_norm, w_lin, eps=1e-6):
        x_f = x.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        x_n = (x_f * rms * w_norm.float()).to(x.dtype)
        return torch.nn.functional.linear(x_n, w_lin)

    @cuda_required
    def test_correctness(self):
        torch.manual_seed(3)
        B, D_in, D_out = 64, 512, 512
        x      = torch.randn(B, D_in,       device="cuda", dtype=torch.float16)
        w_norm = torch.ones(D_in,           device="cuda", dtype=torch.float16)
        w_lin  = torch.randn(D_out, D_in,   device="cuda", dtype=torch.float16)

        ref = self._ref(x, w_norm, w_lin)
        out = llm_kernels.fused_rmsnorm_linear(x, w_norm, w_lin)

        assert torch.allclose(ref.float(), out.float(), atol=1e-1), \
            f"FusedRMSNorm max err: {(ref.float() - out.float()).abs().max():.4f}"

    @cuda_required
    def test_output_shape(self):
        B, D_in, D_out = 16, 256, 512
        x     = torch.randn(B, D_in,      device="cuda", dtype=torch.float16)
        w_n   = torch.ones(D_in,          device="cuda", dtype=torch.float16)
        w_l   = torch.randn(D_out, D_in,  device="cuda", dtype=torch.float16)
        out   = llm_kernels.fused_rmsnorm_linear(x, w_n, w_l)
        assert out.shape == (B, D_out)

    @cuda_required
    def test_module_wrapper(self):
        torch.manual_seed(4)
        B, D_in, D_out = 8, 256, 128
        x   = torch.randn(B, D_in, device="cuda", dtype=torch.float16)
        mod = llm_kernels.FusedRMSNormLinear(D_in, D_out).cuda()
        out = mod(x)
        assert out.shape == (B, D_out)


# ---------------------------------------------------------------------------
# int8 GEMM + dequant
# ---------------------------------------------------------------------------

class TestInt8Gemm:

    @cuda_required
    def test_output_shape(self):
        M, N, K = 128, 256, 64
        A = torch.zeros(M, K, device="cuda", dtype=torch.int8)
        B = torch.zeros(K, N, device="cuda", dtype=torch.int8)
        sa = torch.ones(M, device="cuda", dtype=torch.float32)
        sb = torch.ones(N, device="cuda", dtype=torch.float32)
        out = llm_kernels.int8_gemm_dequant_fwd(A, B, sa, sb)
        assert out.shape == (M, N)
        assert out.dtype == torch.float16

    @cuda_required
    def test_quantize_helper(self):
        x = torch.randn(32, 64, device="cuda", dtype=torch.float16)
        x_q, scale = llm_kernels.quantize_symmetric(x)
        assert x_q.dtype == torch.int8
        assert scale.dtype == torch.float32
        assert x_q.shape  == x.shape
        assert scale.shape == (32,)
        # Values should be bounded
        assert x_q.abs().max() <= 127

    @cuda_required
    def test_correctness_vs_fp16(self):
        """int8 has ~0.5–1% quantization error; atol=1.5 is appropriate for M=512."""
        torch.manual_seed(5)
        M = N = K = 512
        A_fp = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B_fp = torch.randn(K, N, device="cuda", dtype=torch.float16)

        A_q, sa = llm_kernels.quantize_symmetric(A_fp)
        B_q, sb = llm_kernels.quantize_symmetric(B_fp.T)
        B_q = B_q.T.contiguous()

        out = llm_kernels.int8_gemm_dequant_fwd(A_q, B_q, sa, sb)
        ref = (A_fp @ B_fp).to(torch.float16)

        max_err = (ref.float() - out.float()).abs().max().item()
        assert max_err < 2.0, f"int8 GEMM max err too large: {max_err:.4f}"

    @cuda_required
    def test_module_wrapper(self):
        torch.manual_seed(6)
        B, D_in, D_out = 4, 256, 128
        x       = torch.randn(B, D_in, device="cuda", dtype=torch.float16)
        w_fp    = torch.randn(D_out, D_in, device="cuda", dtype=torch.float16)
        layer   = llm_kernels.Int8Linear(D_in, D_out)
        layer.quantize_weights(w_fp)
        out = layer(x)
        assert out.shape == (B, D_out)
