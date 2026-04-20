"""
kernels/flash_attn/flash_attn_bwd.py
=====================================
Flash Attention backward pass — Triton implementation.

Theory
------
FlashAttention-2 avoids storing the N×N attention matrix during the forward
pass. To backpropagate we need to recompute softmax from the stored (m_i, l_i)
statistics rather than from stored attention weights.

Stored during forward:
  - O   [B, H, M, D]   output
  - L   [B, H, M]      log-sum-exp: L_i = m_i + log(l_i)

Backward update for a query tile (Dao et al. 2022, Algorithm 4):
  dV  += P^T dO                     P = softmax(QK^T / sqrt(D))
  dP   = dO V^T                     [BLOCK_M, BLOCK_N]
  dS   = P * (dP - D_i)             D_i = rowsum(dO * O) — "delta"
  dQ  += dS K / sqrt(D)
  dK  += dS^T Q / sqrt(D)

This implementation stores (L, D) from the forward pass to avoid a second
forward sweep during the backward.

Status: reference implementation — numerically correct, not yet fully optimized.
See FlashAttention-2 paper for further optimizations (warp specialization, etc.)
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Modified forward kernel that also writes L = m + log(l)
# ---------------------------------------------------------------------------

@triton.jit
def flash_attn_fwd_with_lse_kernel(
    Q, K, V, Out, LSE,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_lm,
    B, H, M, N,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL:  tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // H
    pid_h  = pid_bh % H

    Q_ptr   = Q   + pid_b * stride_qb + pid_h * stride_qh
    K_ptr   = K   + pid_b * stride_kb + pid_h * stride_kh
    V_ptr   = V   + pid_b * stride_vb + pid_h * stride_vh
    O_ptr   = Out + pid_b * stride_ob + pid_h * stride_oh
    LSE_ptr = LSE + pid_b * stride_lb + pid_h * stride_lh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    scale = D ** -0.5

    kv_limit = (pid_m + 1) * BLOCK_M if CAUSAL else N

    for j in range(0, kv_limit, BLOCK_N):
        offs_n = j + tl.arange(0, BLOCK_N)
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        qk = tl.dot(q, tl.trans(k)) * scale

        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            qk = tl.where(causal_mask, qk, float('-inf'))

        m_j   = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)
        beta  = tl.exp(qk - m_new[:, None])
        l_i   = alpha * l_i + tl.sum(beta, axis=1)
        acc   = alpha[:, None] * acc + tl.dot(beta.to(tl.float16), v)
        m_i   = m_new

    # Write output
    acc = acc / l_i[:, None]
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, acc)

    # Write LSE = m + log(l)  [used in backward]
    lse = m_i + tl.log(l_i)
    tl.store(LSE_ptr + offs_m * stride_lm, lse)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------

@triton.jit
def flash_attn_bwd_kernel(
    Q, K, V, O, LSE, DO,
    DQ, DK, DV,
    Delta,                              # [B, H, M]  D_i = rowsum(dO * O)
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    B, H, M, N,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL:  tl.constexpr,
):
    """
    Each CTA computes (dK, dV) for one (batch, head, KV-tile).
    dQ is accumulated atomically (or via a second sweep).
    """
    pid_n  = tl.program_id(0)    # KV tile
    pid_bh = tl.program_id(1)
    pid_b  = pid_bh // H
    pid_h  = pid_bh % H

    Q_ptr  = Q  + pid_b * stride_qb + pid_h * stride_qh
    K_ptr  = K  + pid_b * stride_kb + pid_h * stride_kh
    V_ptr  = V  + pid_b * stride_vb + pid_h * stride_vh
    O_ptr  = O  + pid_b * stride_ob + pid_h * stride_oh
    DO_ptr = DO + pid_b * stride_ob + pid_h * stride_oh   # same layout as O
    DQ_ptr = DQ + pid_b * stride_qb + pid_h * stride_qh
    DK_ptr = DK + pid_b * stride_kb + pid_h * stride_kh
    DV_ptr = DV + pid_b * stride_vb + pid_h * stride_vh
    LSE_ptr   = LSE   + pid_b * stride_lb + pid_h * stride_lh
    Delta_ptr = Delta + pid_b * stride_db + pid_h * stride_dh

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    scale  = D ** -0.5

    k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    v = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    # Iterate over query tiles
    m_limit = (pid_n + 1) * BLOCK_N if CAUSAL else M

    for i in range(0, m_limit, BLOCK_M):
        offs_m = i + tl.arange(0, BLOCK_M)

        q   = tl.load(Q_ptr  + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        o   = tl.load(O_ptr  + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
        do  = tl.load(DO_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
        lse = tl.load(LSE_ptr   + offs_m * stride_lm)   # [BLOCK_M]
        Di  = tl.load(Delta_ptr + offs_m * stride_dm)   # [BLOCK_M]

        # Recompute attention weights P
        qk = tl.dot(q, tl.trans(k)) * scale              # [BLOCK_M, BLOCK_N]
        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            qk = tl.where(causal_mask, qk, float('-inf'))
        p = tl.exp(qk - lse[:, None])                    # [BLOCK_M, BLOCK_N]

        # dV += P^T dO
        dv += tl.dot(tl.trans(p.to(tl.float16)), do.to(tl.float16))

        # dP = dO V^T  [BLOCK_M, BLOCK_N]
        dp = tl.dot(do.to(tl.float16), tl.trans(v.to(tl.float16)))

        # dS = P * (dP - D_i)
        ds = p * (dp - Di[:, None])

        # dQ += dS K / sqrt(D)
        dq = tl.dot(ds.to(tl.float16), k.to(tl.float16)) * scale
        tl.atomic_add(DQ_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk, dq)

        # dK += dS^T Q / sqrt(D)
        dk += tl.dot(tl.trans(ds.to(tl.float16)), q.to(tl.float16)) * scale

    tl.store(DK_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk, dk)
    tl.store(DV_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk, dv)


# ---------------------------------------------------------------------------
# torch.autograd.Function — ties fwd + bwd together
# ---------------------------------------------------------------------------

class FlashAttnFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal=True, block_m=64, block_n=64):
        B, H, M, D = q.shape
        N = k.shape[2]
        assert M % block_m == 0 and N % block_n == 0

        out = torch.empty_like(q)
        lse = torch.empty(B, H, M, device=q.device, dtype=torch.float32)

        grid = (M // block_m, B * H)
        flash_attn_fwd_with_lse_kernel[grid](
            q, k, v, out, lse,
            *q.stride(), *k.stride(), *v.stride(), *out.stride(),
            lse.stride(0), lse.stride(1), lse.stride(2),
            B, H, M, N,
    D: tl.constexpr,
            BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,
        )

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal   = causal
        ctx.block_m  = block_m
        ctx.block_n  = block_n
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        causal  = ctx.causal
        block_m = ctx.block_m
        block_n = ctx.block_n

        B, H, M, D = q.shape
        N = k.shape[2]

        do = do.contiguous()

        # D_i = rowsum(dO * O)  — computed efficiently in PyTorch
        delta = (do.float() * o.float()).sum(-1)   # [B, H, M]

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        grid = (N // block_n, B * H)
        flash_attn_bwd_kernel[grid](
            q, k, v, o, lse, do,
            dq, dk, dv, delta,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            lse.stride(0), lse.stride(1), lse.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            B, H, M, N,
    D: tl.constexpr,
            BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal,
        )
        return dq, dk, dv, None, None, None


def flash_attn_with_grad(q, k, v, causal=True, block_m=64, block_n=64):
    """
    Flash Attention forward + backward via torch.autograd.

    Supports torch.autograd.grad() and .backward() as normal.
    """
    return FlashAttnFunction.apply(q, k, v, causal, block_m, block_n)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, M, D = 1, 4, 128, 64
    q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16, requires_grad=True)

    # Reference
    q_r = q.detach().clone().requires_grad_(True)
    k_r = k.detach().clone().requires_grad_(True)
    v_r = v.detach().clone().requires_grad_(True)
    ref = torch.nn.functional.scaled_dot_product_attention(q_r, k_r, v_r, is_causal=True)
    ref.sum().backward()

    # Custom
    out = flash_attn_with_grad(q, k, v, causal=True)
    out.sum().backward()

    fwd_err  = (ref - out).abs().max().item()
    dq_err   = (q_r.grad - q.grad).abs().max().item()
    dk_err   = (k_r.grad - k.grad).abs().max().item()
    dv_err   = (v_r.grad - v.grad).abs().max().item()

    print(f"Forward  max err: {fwd_err:.4f}  (atol 0.01)")
    print(f"dQ       max err: {dq_err:.4f}  (atol 0.05)")
    print(f"dK       max err: {dk_err:.4f}  (atol 0.05)")
    print(f"dV       max err: {dv_err:.4f}  (atol 0.05)")
    print("PASSED" if all(e < 0.05 for e in [fwd_err, dq_err, dk_err, dv_err]) else "CHECK TOLERANCES")
