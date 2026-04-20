"""
int8 Symmetric GEMM + Fused Dequantization — Triton Implementation
===================================================================
Computes:  C = dequant(A_int8 @ B_int8, scale_a, scale_b)

Key design decisions
--------------------
- A30 supports DP4A (int8 dot-product-4-accumulate) — Triton's tl.dot on
  int8 inputs uses this automatically, giving ~4× throughput vs fp16 GEMM.
- Accumulation is int32 (prevents overflow for K up to ~16 384).
- Dequantization is fused in the epilogue: no HBM round-trip for int32 output.
- Per-channel symmetric quantization (absmax / 127).

Quantization scheme
-------------------
  For a matrix X of shape [M, K]:
    scale[i] = max(|X[i, :]|) / 127          (per-row scale)
    X_q[i, j] = round(X[i, j] / scale[i])   clamped to [-128, 127]

  Dequant in epilogue:
    C_fp32[i, j] = int32_acc[i, j] * scale_a[i] * scale_b[j]
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def int8_gemm_dequant_kernel(
    A,
    B,
    C,
    scale_a,
    scale_b,  # [M] and [N] fp32 dequant scales
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Grid: (ceil(M / BLOCK_M), ceil(N / BLOCK_N))
    Each CTA computes a [BLOCK_M, BLOCK_N] output tile.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # int32 accumulator — never overflows for K ≤ 16 384
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)

    # ── main K loop ────────────────────────────────────────────────────────
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        a = tl.load(
            A + offs_m[:, None] * K + k_offs[None, :],
            mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
            other=0,
        )  # [BLOCK_M, BLOCK_K] int8

        b = tl.load(
            B + k_offs[:, None] * N + offs_n[None, :],
            mask=(k_offs[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        )  # [BLOCK_K, BLOCK_N] int8

        # int8 × int8 → int32 via DP4A
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)

    # ── fused dequantization epilogue ──────────────────────────────────────
    sa = tl.load(scale_a + offs_m, mask=offs_m < M)  # [BLOCK_M] fp32
    sb = tl.load(scale_b + offs_n, mask=offs_n < N)  # [BLOCK_N] fp32

    out = acc.to(tl.float32) * sa[:, None] * sb[None, :]  # [BLOCK_M, BLOCK_N]

    tl.store(
        C + offs_m[:, None] * N + offs_n[None, :],
        out.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def int8_gemm_dequant_fwd(
    A: torch.Tensor,  # [M, K] int8, CUDA
    B: torch.Tensor,  # [K, N] int8, CUDA
    scale_a: torch.Tensor,  # [M]    fp32, CUDA  (per-row of A)
    scale_b: torch.Tensor,  # [N]    fp32, CUDA  (per-col of B)
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
) -> torch.Tensor:
    """
    int8 GEMM with fused per-channel dequantization.

    Returns:
        C : [M, N] fp16
    """
    assert A.dtype == torch.int8, "A must be int8"
    assert B.dtype == torch.int8, "B must be int8"
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.is_contiguous() and B.is_contiguous(), "Inputs must be contiguous"

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimension mismatch: A has K={K}, B has K={K2}"

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    int8_gemm_dequant_kernel[grid](
        A,
        B,
        C,
        scale_a,
        scale_b,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return C


# ---------------------------------------------------------------------------
# Quantization helper
# ---------------------------------------------------------------------------


def quantize_symmetric(x: torch.Tensor):
    """
    Per-row symmetric absmax quantization.

    Args:
        x : [M, K]  fp16 or fp32

    Returns:
        x_q   : [M, K]  int8
        scale : [M]     fp32  (absmax / 127)
    """
    x_f32 = x.float()
    scale = x_f32.abs().amax(dim=-1) / 127.0  # [M]
    x_q = (x_f32 / scale.unsqueeze(-1)).round()
    x_q = x_q.clamp(-128, 127).to(torch.int8)
    return x_q, scale.to(torch.float32)


# ---------------------------------------------------------------------------
# Quick correctness smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    M, N, K = 512, 512, 512

    A_fp = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B_fp = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Quantize A (row-major) and B transposed then re-transposed so
    # scale_b is per-column of B.
    A_q, scale_a = quantize_symmetric(A_fp)
    B_q, scale_b = quantize_symmetric(B_fp.T)  # quantize rows of B^T
    B_q = B_q.T.contiguous()  # back to [K, N]

    out = int8_gemm_dequant_fwd(A_q, B_q, scale_a, scale_b)

    # Reference: fp16 matmul
    ref = (A_fp @ B_fp).to(torch.float16)

    # int8 has ~0.5% quantization error — atol=1.0 is appropriate
    match = torch.allclose(ref, out, atol=1.0)
    max_err = (ref.float() - out.float()).abs().max().item()
    print(f"[int8_gemm] correctness={match}  max_abs_err={max_err:.4f}")


def int8_gemm_dequant_auto(A, B, scale_a, scale_b):
    M = A.shape[0]
    if M <= 1024:
        bm, bn, bk = 64, 64, 64
    else:
        bm, bn, bk = 128, 128, 64
    return int8_gemm_dequant_fwd(
        A, B, scale_a, scale_b, block_m=bm, block_n=bn, block_k=bk
    )
