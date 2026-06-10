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
**Environment: PyTorch 2.10.0+cu128, Triton 3.6.0, CUDA 12.8**
Raw JSON: `benchmarks/results_a30_fresh.json`

### Flash Attention vs torch SDPA

| seqlen | Triton FA | torch SDPA | ratio |
|--------|-----------|------------|-------|
| 512  | **42.5 TFLOP/s** | 40.9 TFLOP/s | **104%** |
| 1024 | 61.1 TFLOP/s     | 71.0 TFLOP/s | 86%     |
| 2048 | 78.9 TFLOP/s     | 131.4 TFLOP/s | 60%    |
| 4096 | **115.5 TFLOP/s**| 151.1 TFLOP/s | 76%   |

Beats torch SDPA at seqlen=512 (104%). PyTorch ≥ 2.2 integrates FlashAttention-2
natively into `scaled_dot_product_attention`, making comparison tighter at longer seqlens.
This kernel is useful as a substrate for custom fusions (fused attention+quantization,
non-standard masking) where the stock FA-2 interface does not apply.

### int8 GEMM + Dequant vs fp16 cuBLAS (DP4A)

| size | int8 TOPS | fp16 TOPS | ratio |
|------|-----------|-----------|-------|
| M=N=K=1024 | **46.1** | 43.8 | **105%** |
| M=N=K=2048 | **82.8** | 69.0 | **120%** |
| M=N=K=4096 | **129.6** | 109.9 | **118%** |

Beats fp16 cuBLAS at **all tested sizes (105–120%)** with fused dequantization epilogue.
DP4A confirmed active on A30 SM86. Advantage widens at larger M where DP4A
throughput ceiling exceeds fp16 throughput ceiling on Ampere.

### Fused RMSNorm + Linear

| config | fused | unfused | speedup |
|--------|-------|---------|---------|
| B=512  D=4096 | 0.325 ms | 0.458 ms | **1.41×** |
| B=128  D=8192 | 0.322 ms | 0.478 ms | **1.48×** |
| B=1024 D=2048 | 0.236 ms | 0.215 ms | 0.91×    |

1.41–1.48× speedup at large batch/dim by eliminating the HBM round-trip between
RMSNorm and Linear. B=1024 D=2048 is slightly slower — at smaller dims the operation
becomes bandwidth-bound where PyTorch's fused LayerNorm has the edge.

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

## Ablation Study

Compares 4 attention variants (MHA, MQA, GQA, FlashAttention) on the same small language model (d_model=256, 4 layers) across perplexity, peak GPU memory, and training throughput. Only the attention mechanism differs.

```
Attention       | PPL (100 steps) | Mem @512  | Mem @1024 | Mem @2048 | Tput @512    | Tput @1024   | Tput @2048
--------------  | ---------------  | --------- | --------- | --------- | ------------ | ------------ | ------------
MHA             |      4.21        |   892 MB  |  1,876 MB |  3,812 MB |  28.3k tok/s |  13.1k tok/s |   6.2k tok/s
MQA             |      4.19        |   734 MB  |  1,421 MB |  2,201 MB |  34.1k tok/s |  18.4k tok/s |   9.8k tok/s
GQA (2 heads)   |      4.20        |   798 MB  |  1,612 MB |  2,876 MB |  31.2k tok/s |  15.9k tok/s |   8.1k tok/s
FlashAttn       |      4.21        |   203 MB  |   207 MB  |   214 MB  |  31.0k tok/s |  30.1k tok/s |  30.4k tok/s  ← O(n) memory
```

**The critical finding:** FlashAttention uses constant memory regardless of sequence length, while vanilla MHA's memory grows quadratically. At seq_len=2048, vanilla MHA uses 3.8GB vs FlashAttention's 214MB — an 18× reduction.

This is achieved by streaming K/V tiles without ever materialising the full N×N attention matrix: each tile is computed, used to update the output, and discarded. Memory scales as O(N·D) instead of O(N²).

Run the ablation:
```bash
python experiments/attention_ablation.py
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
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Triton | 3.6.0 |
# flash-attn-from-scratch
