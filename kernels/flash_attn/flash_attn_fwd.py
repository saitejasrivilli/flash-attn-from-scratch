import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd_kernel(
    Q,
    K,
    V,
    Out,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_ob,
    stride_oh,
    stride_om,
    stride_ok,
    B,
    H,
    M,
    N,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    O_ptr = Out + pid_b * stride_ob + pid_h * stride_oh
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    scale = D**-0.5
    for j in range(0, N, BLOCK_N):
        offs_n = j + tl.arange(0, BLOCK_N)
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        qk = tl.dot(q, tl.trans(k)) * scale
        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(beta, axis=1)
        acc = alpha[:, None] * acc + tl.dot(beta.to(tl.float16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, acc)


@triton.jit
def flash_attn_fwd_causal_kernel(
    Q,
    K,
    V,
    Out,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_ob,
    stride_oh,
    stride_om,
    stride_ok,
    B,
    H,
    M,
    N,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    Q_ptr = Q + pid_b * stride_qb + pid_h * stride_qh
    K_ptr = K + pid_b * stride_kb + pid_h * stride_kh
    V_ptr = V + pid_b * stride_vb + pid_h * stride_vh
    O_ptr = Out + pid_b * stride_ob + pid_h * stride_oh
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    scale = D**-0.5
    max_n = (pid_m + 1) * BLOCK_M
    for j in range(0, max_n, BLOCK_N):
        offs_n = j + tl.arange(0, BLOCK_N)
        k = tl.load(K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
        v = tl.load(V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        qk = tl.dot(q, tl.trans(k)) * scale
        causal_mask = offs_n[None, :] <= offs_m[:, None]
        qk = tl.where(causal_mask, qk, float("-inf"))
        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(beta, axis=1)
        acc = alpha[:, None] * acc + tl.dot(beta.to(tl.float16), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, acc)


def flash_attn_forward(q, k, v, causal=True, block_m=64, block_n=64):
    assert q.dtype == torch.float16 and q.is_cuda
    B, H, M, D = q.shape
    N = k.shape[2]
    assert M % block_m == 0 and N % block_n == 0
    out = torch.empty_like(q)
    grid = (M // block_m, B * H)
    kernel = flash_attn_fwd_causal_kernel if causal else flash_attn_fwd_kernel
    kernel[grid](
        q,
        k,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        B,
        H,
        M,
        N,
        D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, M, D = 2, 8, 512, 64
    q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = flash_attn_forward(q, k, v, causal=True)
    match = torch.allclose(ref, out, atol=1e-2)
    print(f"[flash_attn] correctness={match}  max_abs_err={(ref-out).abs().max():.4f}")
    assert match
