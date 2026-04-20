# Interview Q&A — llm-kernel-lib

Answers to the questions NVIDIA, Meta, and inference-focused startups
actually ask about this project. Each answer is written for spoken delivery
(2–3 minutes) not as a wall of text.

---

## Flash Attention

**Q: Walk me through why FlashAttention is faster than standard attention.**

Standard attention materialises the full N×N attention matrix in HBM (GPU DRAM).
For sequence length 4096, that's 4096² × 2 bytes = 32 MB *per head*. Reading and
writing that 32 MB dominates runtime — the kernel is memory-bound, not
compute-bound.

FlashAttention avoids writing the N×N matrix entirely. It tiles the computation:
for each query tile of size BLOCK_M, we stream all K/V tiles through a loop,
maintaining running (max, sum, output-accumulator) statistics. At the end of the
loop we have the correct output without ever storing the full matrix. The key
identity is the online softmax recurrence:

    m_new = max(m_old, tile_max)
    l_new = exp(m_old - m_new) * l_old + sum(exp(x - m_new))
    O_new = diag(exp(m_old - m_new)) * O_old + exp(QK^T - m_new) * V

The HBM reads become O(N·D) instead of O(N²), which moves the kernel from
memory-bound to compute-bound at long sequences.

---

**Q: What's the arithmetic intensity of Flash Attention and where does it sit
on the roofline?**

FLOPs ≈ 4·B·H·M²·D (QK^T + softmax·V, two matmuls).
HBM bytes ≈ 2·(Q + K + V + O) = 8·B·H·M·D·2 bytes.

Arithmetic intensity ≈ 4M²D / (8·2·MD) = M/4 FLOP/byte.

On an A30 (933 GB/s, 165 TFLOP/s fp16), the ridge point is at
165e12 / 933e9 ≈ 177 FLOP/byte. So for M < 700 we're memory-bound;
for M > 700 we're compute-bound. At M=2048, AI ≈ 512, which is deep
in the compute-bound regime — exactly where you want to be.

---

**Q: What's the difference between FlashAttention-1 and FlashAttention-2?**

FA-1 (Dao et al. 2022) introduced online softmax and tiled HBM access.
FA-2 (2023) added:
1. **Better work partitioning** — splits work across thread blocks over the query
   dimension instead of K/V, reducing inter-warp communication.
2. **Fewer rescaling operations** — the output rescaling `alpha * acc` was moved
   to happen once per K/V tile instead of once per inner iteration.
3. **Causal masking optimisation** — skips entire KV tiles that are fully masked,
   reducing FLOPs by ~50% for causal attention.

My implementation includes the causal skip optimisation (the `max_n` trick in
the loop bound) but not FA-2's warp partitioning — that's the main remaining gap
to close.

---

**Q: You cast `beta` to fp16 before `tl.dot`. Why?**

Triton's `tl.dot` requires both operands to have the same dtype. `beta` is
computed as `exp(qk - m_new)` in fp32 (to avoid underflow). `v` is fp16.
Without the cast, Triton throws a dtype error at JIT time. The cast is safe
because the values in `beta` are in (0, 1] — no precision is lost going fp32→fp16.

---

## Fused RMSNorm + Linear

**Q: Why is fusing RMSNorm + Linear faster than calling them separately?**

When you call them separately, PyTorch writes the normalised activations back to
HBM after RMSNorm, then reads them again for the linear layer. That's a full
B×D read + write in HBM — at B=512, D=4096, that's 512×4096×2 = 4 MB of wasted
traffic each way.

In the fused kernel, the normalised activations live in shared memory (on-chip,
~100× higher bandwidth than HBM). The CUDA block loads a row, normalises it in
shared mem, and immediately computes the dot product against W_lin — zero HBM
round-trip between the two operations.

The speedup depends on how memory-bound the unfused version is. For small batch
sizes (where the GEMM isn't compute-bound), we see 1.5–2× speedup. For larger
batches the linear layer becomes compute-bound and the fusion benefit shrinks.

---

**Q: Why use one CUDA block per token?**

Two reasons:
1. All threads in a block share on-chip shared memory. Loading the token row
   (D=4096 elements) into shared mem lets every thread in the block access it
   at SMEM bandwidth (~19 TB/s on A30) rather than HBM bandwidth (933 GB/s).
2. The reduction for RMS is naturally intra-block — all threads cooperate to
   sum D elements, then a single warp reduces across warps via `__shfl_xor_sync`.
   A cross-block reduction would require a second kernel launch.

The constraint is D×4 bytes ≤ max shared memory (49 152 bytes on A30), so D ≤ 12 288.
For larger D we'd need to tile across the hidden dimension.

---

**Q: What's the difference between RMSNorm and LayerNorm?**

LayerNorm: normalise then subtract mean, divide by std, apply γ and β.
RMSNorm: skip the mean subtraction — just divide by the root-mean-square.
RMSNorm saves ~30% of the computation (no mean accumulation, no β parameter)
and Zhang & Sennrich (2019) showed it achieves similar training quality.
LLaMA, Mistral, and Gemma all use RMSNorm.

---

## int8 GEMM

**Q: Explain the quantization scheme you chose and its tradeoffs.**

Per-channel symmetric absmax: `scale[i] = max(|x[i,:]|) / 127`.

*Why symmetric?* Zero-point = 0 simplifies the dequant: `x ≈ x_q * scale`.
Asymmetric quantization is more accurate but adds a zero-point term that costs
extra multiplications in the epilogue.

*Why per-channel?* Per-tensor would use a single scale for the entire matrix.
LLM activations have large per-channel variance (the "outlier" problem, see
LLM.int8 by Dettmers et al.), so per-tensor scale would clip most channels to
near-zero. Per-channel is the minimum granularity that handles outliers without
mixed-precision fallback.

*Tradeoff:* per-channel requires storing M scales (one per row of A). For very
large batch sizes this is negligible; for batch=1 inference it's a few KB.

---

**Q: What is DP4A and how does Triton use it?**

DP4A (dot-product of 4 int8 values, accumulate to int32) is a hardware
instruction on SM61+ (Pascal and newer). It computes:
`d += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]`
in a single cycle, giving ~4× the throughput of fp16 for matrix multiplication.

In Triton, you get DP4A automatically by calling `tl.dot(a, b)` where `a` and
`b` are `int8` tensors. Triton's PTX lowering emits `dp4a` instructions.
The accumulator must be `int32` to prevent overflow — this is enforced by
declaring `acc = tl.zeros([...], dtype=tl.int32)`.

---

**Q: Why not just use cuBLAS int8 GEMM?**

cuBLAS `cublasGemmEx` with `CUDA_R_8I` is faster for large, square, well-aligned
matrices (M=N=K a multiple of 128). But it has constraints:
- No custom epilogues — you get int32 output and must dequantize separately.
- No per-channel scale fusion — that's a separate kernel call.
- Requires cuBLAS handle management.

The Triton kernel fuses the dequantization, saving one HBM write (int32 output)
and one HBM read (scales + int32 input). For small or irregular shapes Triton
also tends to be faster because cuBLAS has large kernel selection overhead.

In production (vLLM, TRT-LLM) you'd use cuBLAS or CUTLASS for the largest GEMMs
and custom Triton kernels for fused epilogues. The real value of this kernel is
demonstrating you understand the dequant fusion, not replacing cuBLAS.

---

## Systems / General

**Q: If this kernel is 80% of FlashAttention-2's throughput, what's the other 20%?**

Three main gaps:
1. **Warp specialisation** — FA-2 assigns different warps to producer/consumer
   roles (one warp loads, another computes). This hides load latency behind
   compute. My implementation doesn't pipeline loads and MMA.
2. **Register blocking** — FA-2 uses larger register tiles to amortise loop
   overhead. I'm at BLOCK_M=64 which isn't always optimal.
3. **Swizzled shared memory layout** — avoids bank conflicts in the V transpose.
   My implementation uses a default layout which can cause 2-way bank conflicts
   on the V tile.

The autotuner catches some of this by finding the best BLOCK_M/BLOCK_N, but
warp specialisation requires rewriting the kernel structure.

---

**Q: How would you extend this to multi-GPU or tensor parallelism?**

Flash Attention parallelises naturally along the head dimension — each GPU
handles H/N_gpu heads independently. This is how Megatron-LM and vLLM do it.

For sequence parallelism (long sequences split across GPUs), you need a Ring
Attention pattern: each GPU processes a slice of the query, and the KV tiles
are rotated around the ring via peer-to-peer NCCL sends. The online softmax
accumulation still works because the (m, l) statistics are commutative.

The fused RMSNorm+Linear maps onto tensor parallelism via column-parallel linear
(shard W_lin along D_out across GPUs, AllReduce after the linear).
