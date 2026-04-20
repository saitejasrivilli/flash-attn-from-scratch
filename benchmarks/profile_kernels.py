"""
benchmarks/profile_flash_attn.py
=================================
Target script for Nsight Compute profiling of the Flash Attention kernel.

Usage (from terminal on an A30 / A100 machine — NOT Colab):
------------------------------------------------------------
    ncu --set full \\
        --target-processes all \\
        -o flash_attn_profile \\
        python benchmarks/profile_flash_attn.py

    # Open the .ncu-rep file in Nsight Compute GUI.
    # Key metrics to examine:
    #   - Roofline chart: memory-bound or compute-bound?
    #   - L2 hit rate: should be high for the K/V tile reuse
    #   - SM utilization: target >80%
    #   - Achieved occupancy: limited by registers or shared mem?

Tips
----
- If roofline shows memory-bound: increase BLOCK_M / BLOCK_N tile sizes.
- If compute-bound: look at instruction-level throughput (good position).
- -lineinfo in nvcc flags (set in setup.py) enables source correlation.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_kernels


def profile_flash_attn():
    B, H, M, D = 2, 16, 2048, 64
    q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warm up JIT compilation before profiling
    for _ in range(5):
        _ = llm_kernels.flash_attn_forward(q, k, v, causal=True)
    torch.cuda.synchronize()

    # Single timed iteration — ncu captures the CUDA kernel launch
    out = llm_kernels.flash_attn_forward(q, k, v, causal=True)
    torch.cuda.synchronize()
    print(f"Output shape: {out.shape}  dtype: {out.dtype}")


def profile_int8_gemm():
    M = N = K = 4096
    A_fp = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B_fp = torch.randn(K, N, device="cuda", dtype=torch.float16)

    A_q, sa = llm_kernels.quantize_symmetric(A_fp)
    B_q, sb = llm_kernels.quantize_symmetric(B_fp.T)
    B_q = B_q.T.contiguous()

    for _ in range(5):
        _ = llm_kernels.int8_gemm_dequant_fwd(A_q, B_q, sa, sb)
    torch.cuda.synchronize()

    out = llm_kernels.int8_gemm_dequant_fwd(A_q, B_q, sa, sb)
    torch.cuda.synchronize()
    print(f"Output shape: {out.shape}  dtype: {out.dtype}")


if __name__ == "__main__":
    profile_flash_attn()
    profile_int8_gemm()
