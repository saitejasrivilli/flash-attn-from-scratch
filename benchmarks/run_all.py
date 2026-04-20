"""
benchmarks/run_all.py
=====================
Unified benchmark for all three kernels. Outputs a markdown-ready table
suitable for pasting directly into README.md.

Usage
-----
    # Standard run (prints table + saves results.json):
    python benchmarks/run_all.py

    # Only flash attention:
    python benchmarks/run_all.py --kernels flash_attn

    # Warmup / timing reps:
    python benchmarks/run_all.py --warmup 20 --reps 200
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_kernels

# ---------------------------------------------------------------------------
# Timing primitive
# ---------------------------------------------------------------------------


def bench_fn(fn, warmup: int = 10, reps: int = 100) -> float:
    """Returns milliseconds per call (GPU time via synchronize)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / reps  # ms per call


# ---------------------------------------------------------------------------
# Flash Attention
# ---------------------------------------------------------------------------


def bench_flash_attn(warmup, reps):
    results = []
    print("\n=== Flash Attention ===")
    B, H, D = 2, 16, 64

    for seqlen in [512, 1024, 2048, 4096]:
        q = torch.randn(B, H, seqlen, D, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # FLOPs: QK^T (2*M*N*D) + softmax*V (2*M*N*D) = 4*M*N*D, times B*H
        flops = 4 * B * H * seqlen * seqlen * D

        ms_triton = bench_fn(
            lambda: llm_kernels.flash_attn_forward(q, k, v), warmup, reps
        )
        ms_sdpa = bench_fn(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            ),
            warmup,
            reps,
        )

        tflops_triton = flops / ms_triton / 1e9  # ms → s, flops → TFLOP
        tflops_sdpa = flops / ms_sdpa / 1e9
        ratio = tflops_triton / tflops_sdpa * 100

        row = {
            "kernel": "Flash Attention (Triton)",
            "config": f"B={B} H={H} seqlen={seqlen} D={D}",
            "throughput": f"{tflops_triton:.1f} TFLOP/s",
            "vs_baseline": f"{ratio:.0f}% of torch SDPA",
        }
        results.append(row)
        print(
            f"  seqlen={seqlen:4d}: {tflops_triton:.1f} TFLOP/s  "
            f"(torch SDPA: {tflops_sdpa:.1f} TFLOP/s, ratio: {ratio:.0f}%)"
        )

    return results


# ---------------------------------------------------------------------------
# Fused RMSNorm + Linear
# ---------------------------------------------------------------------------


def bench_fused_rmsnorm(warmup, reps):
    results = []
    print("\n=== Fused RMSNorm + Linear ===")

    for B, D_in, D_out in [(512, 4096, 4096), (128, 8192, 8192), (1024, 2048, 2048)]:
        x = torch.randn(B, D_in, device="cuda", dtype=torch.float16)
        w_norm = torch.ones(D_in, device="cuda", dtype=torch.float16)
        w_lin = torch.randn(D_out, D_in, device="cuda", dtype=torch.float16)

        rms_layer = torch.nn.Linear(
            D_in, D_in, bias=False, device="cuda", dtype=torch.float16
        )
        lin_layer = torch.nn.Linear(
            D_in, D_out, bias=False, device="cuda", dtype=torch.float16
        )

        ms_fused = bench_fn(
            lambda: llm_kernels.fused_rmsnorm_linear(x, w_norm, w_lin), warmup, reps
        )
        ms_unfused = bench_fn(lambda: lin_layer(rms_layer(x)), warmup, reps)

        speedup = ms_unfused / ms_fused

        row = {
            "kernel": "Fused RMSNorm+Linear (CUDA)",
            "config": f"B={B} D_in={D_in} D_out={D_out}",
            "throughput": f"{ms_fused:.3f} ms",
            "vs_baseline": f"{speedup:.2f}x vs unfused PyTorch",
        }
        results.append(row)
        print(
            f"  B={B:4d} D={D_in:4d}: fused={ms_fused:.3f} ms  "
            f"unfused={ms_unfused:.3f} ms  speedup={speedup:.2f}x"
        )

    return results


# ---------------------------------------------------------------------------
# int8 GEMM + dequant
# ---------------------------------------------------------------------------


def bench_int8_gemm(warmup, reps):
    results = []
    print("\n=== int8 GEMM + Dequant ===")

    for size in [1024, 2048, 4096]:
        M = N = K = size
        A_fp = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B_fp = torch.randn(K, N, device="cuda", dtype=torch.float16)

        A_q, scale_a = llm_kernels.quantize_symmetric(A_fp)
        B_q, scale_b = llm_kernels.quantize_symmetric(B_fp.T)
        B_q = B_q.T.contiguous()

        # TOPS: M*N*K multiply-adds = 2*M*N*K ops (counting mul+add as 2)
        ops = 2 * M * N * K

        ms_int8 = bench_fn(
            lambda: llm_kernels.int8_gemm_dequant_fwd(A_q, B_q, scale_a, scale_b),
            warmup,
            reps,
        )
        ms_fp16 = bench_fn(lambda: A_fp @ B_fp, warmup, reps)

        tops_int8 = ops / ms_int8 / 1e9  # TOPS
        tops_fp16 = ops / ms_fp16 / 1e9

        row = {
            "kernel": "int8 GEMM+dequant (Triton)",
            "config": f"M=N=K={size}",
            "throughput": f"{tops_int8:.1f} TOPS",
            "vs_baseline": f"fp16 cublas: {tops_fp16:.1f} TOPS",
        }
        results.append(row)
        print(f"  M=N=K={size}: int8={tops_int8:.1f} TOPS  fp16={tops_fp16:.1f} TOPS")

    return results


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------


def print_markdown_table(rows):
    header = ["Kernel", "Config", "Throughput", "vs Baseline"]
    col_w = [
        max(len(h), max(len(r[k]) for r in rows))
        for h, k in zip(header, ["kernel", "config", "throughput", "vs_baseline"])
    ]

    def row_str(vals):
        return "| " + " | ".join(v.ljust(w) for v, w in zip(vals, col_w)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
    print("\n## Benchmark Results\n")
    print(row_str(header))
    print(sep)
    for r in rows:
        print(row_str([r["kernel"], r["config"], r["throughput"], r["vs_baseline"]]))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=["flash_attn", "rmsnorm", "int8_gemm"],
        default=["flash_attn", "rmsnorm", "int8_gemm"],
        help="Which kernels to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument(
        "--out", default="benchmarks/results.json", help="Path to save JSON results"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run on a GPU machine.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    print(f"CUDA:   {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    all_results = []

    if "flash_attn" in args.kernels:
        all_results.extend(bench_flash_attn(args.warmup, args.reps))
    if "rmsnorm" in args.kernels:
        all_results.extend(bench_fused_rmsnorm(args.warmup, args.reps))
    if "int8_gemm" in args.kernels:
        all_results.extend(bench_int8_gemm(args.warmup, args.reps))

    print_markdown_table(all_results)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"device": device_name, "results": all_results}, f, indent=2)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
