"""
benchmarks/roofline.py
======================
Analytical roofline model for all three kernels on a given GPU.

Prints the predicted arithmetic intensity (FLOPs / byte) for each kernel
and overlays it against the GPU's compute and memory rooflines.

This is the same analysis that Nsight Compute shows graphically — useful for
understanding *before* you profile whether you're targeting the right bottleneck.

Usage:
    python benchmarks/roofline.py [--gpu a30|a100|h100|t4]

The numbers printed here are theoretical; actual achieved values come from
Nsight Compute. The goal is to understand which kernels are memory-bound
vs compute-bound *in principle*, then verify with the profiler.
"""

import argparse
import math

# ---------------------------------------------------------------------------
# GPU specs (peak FP16 FLOP/s and HBM bandwidth)
# ---------------------------------------------------------------------------
GPU_SPECS = {
    "a30": {
        "name": "NVIDIA A30",
        "fp16_tflops": 165.0,  # Tensor Core peak, FP16
        "int8_tops": 330.0,  # DP4A peak
        "hbm_bw_tbs": 933.0e9,  # bytes/sec (933 GB/s)
        "l2_bw_tbs": 3200.0e9,  # bytes/sec (estimated)
    },
    "a100": {
        "name": "NVIDIA A100 (80GB)",
        "fp16_tflops": 312.0,
        "int8_tops": 624.0,
        "hbm_bw_tbs": 2000.0e9,
        "l2_bw_tbs": 6000.0e9,
    },
    "h100": {
        "name": "NVIDIA H100 SXM",
        "fp16_tflops": 989.0,
        "int8_tops": 1979.0,
        "hbm_bw_tbs": 3350.0e9,
        "l2_bw_tbs": 10000.0e9,
    },
    "t4": {
        "name": "NVIDIA T4",
        "fp16_tflops": 65.0,
        "int8_tops": 130.0,
        "hbm_bw_tbs": 320.0e9,
        "l2_bw_tbs": 1200.0e9,
    },
}

# ---------------------------------------------------------------------------
# Kernel arithmetic intensity models
# ---------------------------------------------------------------------------


def flash_attn_intensity(B, H, M, N, D):
    """
    Flash Attention forward, causal.

    FLOPs:
      QK^T:    B*H * M * N * 2D   (matmul)
      softmax: B*H * M * N * 3    (exp, sum, div) — approx
      PV:      B*H * M * N * 2D

    HBM bytes (key insight: NO N×N matrix written):
      Read Q:  B*H*M*D * 2
      Read K:  B*H*N*D * 2
      Read V:  B*H*N*D * 2
      Write O: B*H*M*D * 2
    """
    flops = B * H * (4 * M * N * D + 3 * M * N)
    # Causal: on average half the KV tiles are processed
    flops //= 2 if M == N else 1

    bytes_hbm = 2 * 2 * (B * H * M * D + B * H * N * D + B * H * N * D + B * H * M * D)
    intensity = flops / bytes_hbm
    return flops, bytes_hbm, intensity


def fused_rmsnorm_linear_intensity(B, D_in, D_out):
    """
    Fused RMSNorm + Linear.

    FLOPs:
      RMSNorm: B * (D_in + D_in + D_in) ≈ 3 * B * D_in
      Linear:  B * 2 * D_in * D_out

    HBM bytes (fused — one pass):
      Read X:      B * D_in * 2
      Read W_norm: D_in * 2
      Read W_lin:  D_out * D_in * 2
      Write Y:     B * D_out * 2

    Unfused bytes (for comparison):
      + Write X_norm: B * D_in * 2
      + Read  X_norm: B * D_in * 2
    """
    flops_norm = 3 * B * D_in
    flops_linear = 2 * B * D_in * D_out
    flops = flops_norm + flops_linear

    bytes_fused = 2 * (B * D_in + D_in + D_out * D_in + B * D_out)
    bytes_unfused = bytes_fused + 2 * 2 * B * D_in  # extra HBM round-trip

    intensity_fused = flops / bytes_fused
    intensity_unfused = flops / bytes_unfused

    return flops, bytes_fused, bytes_unfused, intensity_fused, intensity_unfused


def int8_gemm_intensity(M, N, K):
    """
    int8 GEMM + per-channel dequant.

    FLOPs (int8 accumulate as int32, then fp32 mul for dequant):
      GEMM:    2 * M * N * K    (DP4A: 8 ops per int8 dot4)
      Dequant: M * N * 2        (scale_a * scale_b per element)

    HBM bytes:
      Read A int8:   M * K * 1
      Read B int8:   K * N * 1
      Read scale_a:  M * 4
      Read scale_b:  N * 4
      Write C fp16:  M * N * 2
    """
    flops = 2 * M * N * K + 2 * M * N
    bytes_hbm = M * K * 1 + K * N * 1 + M * 4 + N * 4 + M * N * 2
    intensity = flops / bytes_hbm
    return flops, bytes_hbm, intensity


# ---------------------------------------------------------------------------
# Roofline prediction
# ---------------------------------------------------------------------------


def roofline_predict(flops, bytes_hbm, gpu, use_int8=False):
    """
    Returns (predicted_ms, bound_by) given FLOPs, HBM bytes, and GPU spec.
    """
    peak_flops = (gpu["int8_tops"] if use_int8 else gpu["fp16_tflops"]) * 1e12
    peak_bw = gpu["hbm_bw_tbs"]

    t_compute = flops / peak_flops  # seconds
    t_memory = bytes_hbm / peak_bw  # seconds
    t_total = max(t_compute, t_memory)

    bound = "compute" if t_compute > t_memory else "memory"
    return t_total * 1e3, bound  # ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", choices=GPU_SPECS.keys(), default="a30")
    args = parser.parse_args()

    gpu = GPU_SPECS[args.gpu]
    print(f"\n{'='*64}")
    print(f"  Roofline Model — {gpu['name']}")
    print(f"  FP16 peak : {gpu['fp16_tflops']:.0f} TFLOP/s")
    print(f"  INT8 peak : {gpu['int8_tops']:.0f}  TOPS")
    print(f"  HBM BW    : {gpu['hbm_bw_tbs']/1e9:.0f} GB/s")
    print(f"{'='*64}\n")

    # ── Flash Attention ──────────────────────────────────────────────────
    print("Flash Attention (B=2, H=16, D=64, causal)")
    print(
        f"  {'seqlen':>8}  {'GFLOP':>8}  {'GB_hbm':>8}  {'AI (F/B)':>10}  {'pred ms':>9}  {'bound':>9}"
    )
    for M in [512, 1024, 2048, 4096]:
        fl, byt, ai = flash_attn_intensity(2, 16, M, M, 64)
        ms, bound = roofline_predict(fl, byt, gpu)
        print(
            f"  {M:>8}  {fl/1e9:>8.1f}  {byt/1e9:>8.2f}  {ai:>10.1f}  {ms:>9.3f}  {bound:>9}"
        )

    # ── Fused RMSNorm + Linear ─────────────────────────────────────────
    print("\nFused RMSNorm + Linear")
    print(
        f"  {'config':>20}  {'AI fused':>10}  {'AI unfused':>12}  "
        f"{'pred fused ms':>14}  {'pred unfused ms':>16}"
    )
    for B, D_in, D_out in [(512, 4096, 4096), (128, 8192, 8192)]:
        fl, bf, bu, ai_f, ai_u = fused_rmsnorm_linear_intensity(B, D_in, D_out)
        ms_f, _ = roofline_predict(fl, bf, gpu)
        ms_u, _ = roofline_predict(fl, bu, gpu)
        cfg = f"B={B} D={D_in}"
        print(
            f"  {cfg:>20}  {ai_f:>10.1f}  {ai_u:>12.1f}  "
            f"{ms_f:>14.3f}  {ms_u:>16.3f}  (speedup≥{ms_u/ms_f:.2f}x)"
        )

    # ── int8 GEMM ────────────────────────────────────────────────────────
    print("\nint8 GEMM + dequant")
    print(
        f"  {'size':>10}  {'GTOP':>8}  {'GB_hbm':>8}  {'AI (T/B)':>10}  "
        f"{'pred ms':>9}  {'bound':>9}"
    )
    for size in [1024, 2048, 4096]:
        fl, byt, ai = int8_gemm_intensity(size, size, size)
        ms, bound = roofline_predict(fl, byt, gpu, use_int8=True)
        print(
            f"  {f'M=N=K={size}':>10}  {fl/1e9:>8.1f}  {byt/1e9:>8.2f}  "
            f"{ai:>10.1f}  {ms:>9.3f}  {bound:>9}"
        )

    print()
    print("Note: 'pred ms' is the theoretical minimum (roofline ceiling).")
    print("      Compare against actual measured ms from benchmarks/run_all.py")
    print("      to compute efficiency: efficiency = pred_ms / actual_ms × 100%\n")


if __name__ == "__main__":
    main()
