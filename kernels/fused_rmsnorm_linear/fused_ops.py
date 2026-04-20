"""
fused_ops.py
============
Python interface to the CUDA extension for fused RMSNorm + Linear.

At import time we try to load the pre-built extension (llm_kernels_cuda).
If it isn't built yet we fall back to a pure-PyTorch reference so that
the rest of the codebase keeps working.
"""

import torch
import torch.nn.functional as F

try:
    import llm_kernels_cuda as _ext  # compiled via setup.py
    _HAS_CUDA_EXT = True
except ImportError:
    _HAS_CUDA_EXT = False


# ---------------------------------------------------------------------------
# PyTorch reference implementation (always available)
# ---------------------------------------------------------------------------

def _rmsnorm_linear_ref(
    x: torch.Tensor,
    w_norm: torch.Tensor,
    w_lin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Unfused reference: RMSNorm then Linear."""
    x_f = x.float()
    rms = x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    x_norm = (x_f * rms * w_norm.float()).to(x.dtype)
    return F.linear(x_norm, w_lin)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_rmsnorm_linear(
    x: torch.Tensor,
    w_norm: torch.Tensor,
    w_lin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm + Linear forward pass.

    Uses the CUDA kernel when the extension is compiled; falls back to
    PyTorch otherwise (useful on CPU or before first build).

    Args:
        x      : [B, D_in]      fp16, CUDA
        w_norm : [D_in]          fp16, CUDA  (RMSNorm gamma)
        w_lin  : [D_out, D_in]   fp16, CUDA  (Linear weight)
        eps    : stability epsilon

    Returns:
        y      : [B, D_out] fp16
    """
    if _HAS_CUDA_EXT and x.is_cuda and x.dtype == torch.float16:
        return _ext.fused_rmsnorm_linear(x, w_norm, w_lin, eps)
    return _rmsnorm_linear_ref(x, w_norm, w_lin, eps)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, D_in, D_out = 16, 512, 512
    x      = torch.randn(B, D_in,       device="cuda", dtype=torch.float16)
    w_norm = torch.ones(D_in,           device="cuda", dtype=torch.float16)
    w_lin  = torch.randn(D_out, D_in,   device="cuda", dtype=torch.float16)

    ref = _rmsnorm_linear_ref(x, w_norm, w_lin)
    out = fused_rmsnorm_linear(x, w_norm, w_lin)

    match = torch.allclose(ref.float(), out.float(), atol=1e-1)
    print(f"[fused_ops] correctness={match}  cuda_ext={'yes' if _HAS_CUDA_EXT else 'no (fallback)'}")
