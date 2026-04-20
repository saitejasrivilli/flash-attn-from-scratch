"""
llm_kernels
===========
Unified package exposing all three custom GPU kernels as both
bare functions and drop-in nn.Module wrappers.

Quick-start
-----------
>>> import llm_kernels
>>> out = llm_kernels.flash_attn_forward(q, k, v, causal=True)
>>> out = llm_kernels.int8_gemm_dequant_fwd(A_q, B_q, sa, sb)
>>> out = llm_kernels.fused_rmsnorm_linear(x, w_norm, w_lin)

# Or as nn.Modules:
>>> attn = llm_kernels.FlashAttention(causal=True)
>>> norm_proj = llm_kernels.FusedRMSNormLinear(d_in=4096, d_out=4096)
"""

import torch
import torch.nn as nn

# ── bare kernel functions ──────────────────────────────────────────────────
from kernels.flash_attn import flash_attn_forward
from kernels.int8_gemm import int8_gemm_dequant_fwd, quantize_symmetric
from kernels.fused_rmsnorm_linear import fused_rmsnorm_linear


# ── nn.Module wrappers ─────────────────────────────────────────────────────

class FlashAttention(nn.Module):
    """
    Drop-in replacement for scaled_dot_product_attention.

    >>> attn = FlashAttention(causal=True)
    >>> out = attn(q, k, v)   # [B, H, M, D] fp16
    """

    def __init__(self, causal: bool = True, block_m: int = 64, block_n: int = 64):
        super().__init__()
        self.causal  = causal
        self.block_m = block_m
        self.block_n = block_n

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return flash_attn_forward(q, k, v, causal=self.causal,
                                  block_m=self.block_m, block_n=self.block_n)

    def extra_repr(self) -> str:
        return f"causal={self.causal}, block_m={self.block_m}, block_n={self.block_n}"


class FusedRMSNormLinear(nn.Module):
    """
    Fused RMSNorm + Linear layer.

    Equivalent to:  nn.Linear(d_in, d_out, bias=False)(nn.RMSNorm(d_in)(x))
    but with a single kernel launch and no HBM round-trip.

    >>> layer = FusedRMSNormLinear(d_in=4096, d_out=4096)
    >>> y = layer(x)   # [B, D_out] fp16
    """

    def __init__(self, d_in: int, d_out: int, eps: float = 1e-6):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        self.eps   = eps
        self.w_norm = nn.Parameter(torch.ones(d_in,         dtype=torch.float16))
        self.w_lin  = nn.Parameter(torch.empty(d_out, d_in, dtype=torch.float16))
        nn.init.kaiming_uniform_(self.w_lin, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm_linear(x, self.w_norm, self.w_lin, self.eps)

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, d_out={self.d_out}, eps={self.eps}"


class Int8Linear(nn.Module):
    """
    Weight-only int8 quantized linear layer with fused dequantization.

    Weights are stored as int8; activations are quantized on the fly.
    Output is fp16.

    >>> layer = Int8Linear(d_in=4096, d_out=4096)
    >>> layer.quantize_weights(w_fp16)
    >>> y = layer(x)   # [B, D_out] fp16
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        # Buffers aren't parameters — won't appear in optimizer state.
        self.register_buffer("w_q",     torch.zeros(d_out, d_in, dtype=torch.int8))
        self.register_buffer("scale_w", torch.ones(d_out,        dtype=torch.float32))

    @torch.no_grad()
    def quantize_weights(self, w: torch.Tensor):
        """Quantize fp16/fp32 weight [D_out, D_in] and store as int8."""
        self.w_q, self.scale_w = quantize_symmetric(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, D_in] fp16
        """
        x_q, scale_x = quantize_symmetric(x)
        # A=[B, D_in], B=[D_in, D_out] — need B in [K, N] layout
        return int8_gemm_dequant_fwd(
            x_q,
            self.w_q.T.contiguous(),   # [D_in, D_out]
            scale_x,                    # [B]    per-row of activations
            self.scale_w,               # [D_out] per-row of W (= per-col of W^T)
        )

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, d_out={self.d_out}"


# ── public API ─────────────────────────────────────────────────────────────
__all__ = [
    # functions
    "flash_attn_forward",
    "int8_gemm_dequant_fwd",
    "quantize_symmetric",
    "fused_rmsnorm_linear",
    # modules
    "FlashAttention",
    "FusedRMSNormLinear",
    "Int8Linear",
]
