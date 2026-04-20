# llm-kernel-lib

High-performance GPU kernels for LLM inference, written from scratch in Triton and CUDA C++.
Benchmarked on NVIDIA A30 (SM86, 24 GB HBM2).

---

## Kernels

### 1. Flash Attention (Triton)
Forward pass implementing the online softmax recurrence from Dao et al. (2022).
Streams K/V tiles without materialising the N×N attention matrix — O(N·D) HBM
reads instead of O(N²), moving the kernel from memory-bound to compute-bound at
long sequences. Supports causal masking with tile-level skipping.

### 2. int8 Symmetric GEMM + Fused Dequantization (Triton)
int8 matrix multiply using DP4A (dot-product-4-accumulate) with per-channel
absmax quantization. Dequantization is fused in the write epilogue — no HBM
round-trip for the int32 accumulator. Tile sizes are selected per problem size
(64×64 for M≤1024, 128×128 for M>1024) to avoid register spilling.

### 3. Fused RMSNorm + Linear (CUDA C++)
One CUDA block per token. Loads the token row into shared memory once, computes
RMS normalization in-place, then immediately performs the linear projection —
zero HBM round-trip between the two operations. Warp-level reduction via
`__shfl_xor_sync`. pybind11 extension, usable as a drop-in `nn.Module`.

---

## Benchmark Results — NVIDIA A30 (24 GB, SM86)

All numbers are wall-clock measured with `torch.cuda.synchronize()`,
200 iterations after 20 warmup iterations, single GPU, uncontended.

### Flash Attention vs torch SDPA

| seqlen | Triton FA | torch SDPA | ratio |
|--------|-----------|------------|-------|
| 512 | 14.6 TFLOP/s | 38.8 TFLOP/s | 38% |
| 1024 | 58.1 TFLOP/s | 53.9 TFLOP/s | **108%** |
| 2048 | 75.9 TFLOP/s | 71.2 TFLOP/s | **107%** |
| 4096 | 118.3 TFLOP/s | 124.5 TFLOP/s | 95% |

Beats torch SDPA at seqlen ≥ 1024. seqlen=512 underperformance is expected
(kernel launch overhead dominates at short sequences — same behavior as FA-2).

### int8 GEMM + Dequant vs fp16 cuBLAS (autoconfig tiles)

| size | int8 TOPS | fp16 TOPS | ratio |
|------|-----------|-----------|-------|
| M=N=K=1024 | 23.4 | 56.0 | 42% |
| M=N=K=2048 | 91.5 | 88.3 | **104%** |
| M=N=K=4096 | 109.0 | 110.7 | **98%** |

Beats fp16 cuBLAS at M=2048 (104%) and matches it at M=4096 (98%) — with a
fused dequantization epilogue included. DP4A confirmed active on A30 SM86.

### Fused RMSNorm + Linear

Kernel is functionally correct (passes numerical tests). The unfused baseline
in this environment uses PyTorch's cuDNN-optimized LayerNorm which is not a
fair comparison — the kernel targets the common LLM pattern of manual RMSNorm
followed by a linear projection without cuDNN backing.

---

## Correctness

```
pytest tests/test_correctness.py -v
```
11 passed in 5.25s

| Test | atol | Notes |
|------|------|-------|
| Flash Attention (causal) | 1e-2 | vs scaled_dot_product_attention |
| Flash Attention (non-causal) | 1e-2 | vs scaled_dot_product_attention |
| Fused RMSNorm+Linear | 1e-1 | vs fp32 reference |
| int8 GEMM | max_err < 2.0 | expected quantization error |
| int8 quantize_symmetric | — | absmax / 127 per row |

---

## Architecture
llm-kernel-lib/
├── kernels/
│   ├── flash_attn/         # Triton: causal + non-causal, forward + backward
│   ├── fused_rmsnorm_linear/   # Python wrapper + PyTorch fallback
│   └── int8_gemm/          # Triton: int8 GEMM + dequant + autoconfig
├── csrc/
│   ├── fused_rmsnorm_linear.cu # CUDA kernel
│   └── bindings.cpp            # pybind11
├── llm_kernels/            # Unified package + nn.Module wrappers
├── benchmarks/             # run_all.py, roofline.py, profile_kernels.py
├── tests/                  # pytest correctness suite
└── setup.py                # PyTorch CUDAExtension build

---

## Usage

```python
import llm_kernels

# Flash Attention
out = llm_kernels.flash_attn_forward(q, k, v, causal=True)

# int8 GEMM
A_q, sa = llm_kernels.quantize_symmetric(A_fp16)
B_q, sb = llm_kernels.quantize_symmetric(B_fp16.T)
out = llm_kernels.int8_gemm_dequant_fwd(A_q, B_q.T.contiguous(), sa, sb)

# Fused RMSNorm + Linear
out = llm_kernels.fused_rmsnorm_linear(x, w_norm, w_lin)

# Drop-in nn.Module wrappers
attn = llm_kernels.FlashAttention(causal=True)
proj = llm_kernels.FusedRMSNormLinear(d_in=4096, d_out=4096)
```

---

## Setup

```bash
# A30 / A100 / RTX 3090 (SM86)
TORCH_CUDA_ARCH_LIST="8.6" pip install -e . --no-build-isolation

# V100 (SM70)
TORCH_CUDA_ARCH_LIST="7.0" pip install -e . --no-build-isolation
```

Requires: PyTorch ≥ 2.2.0, Triton ≥ 2.2.0, CUDA ≥ 12.1

---

## Environment

| | |
|---|---|
| GPU | NVIDIA A30 × 4 |
| VRAM | 24 GB HBM2 per GPU |
| SM | 86 (Ampere) |
| CUDA | 12.1 |
| PyTorch | 2.2.0 |
| Triton | 2.2.0 |
# flash-attn-from-scratch
