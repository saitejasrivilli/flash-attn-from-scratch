"""
kernels/flash_attn/autotune.py
==============================
Grid-search BLOCK_M × BLOCK_N tile sizes for Flash Attention and cache
the best configuration per (GPU, D, seqlen) triple.

Why bother?
-----------
BLOCK_M and BLOCK_N control the register / shared-memory / occupancy trade-off.
The optimal sizes differ between T4 (SM75), A30 (SM86), and A100 (SM80).
Hard-coding 64×64 is fine for development but leaves 10–25% on the table.

Usage
-----
    # Find and cache best config for current GPU:
    python kernels/flash_attn/autotune.py

    # The best configs are printed and saved to kernels/flash_attn/configs.json
    # flash_attn_forward() loads configs.json at import time if it exists.
"""

import itertools
import json
import os
import time

import torch

# The configs we sweep
BLOCK_SIZES = [32, 64, 128]
WARMUP = 5
REPS = 30


def _bench_config(q, k, v, block_m, block_n, causal):
    """Return ms/call for a given tile config, or inf if it errors."""
    from kernels.flash_attn.flash_attn_fwd import (
        flash_attn_fwd_causal_kernel, flash_attn_fwd_kernel)

    B, H, M, D = q.shape
    N = k.shape[2]
    if M % block_m != 0 or N % block_n != 0:
        return float("inf")

    out = torch.empty_like(q)
    grid = (M // block_m, B * H)
    kernel = flash_attn_fwd_causal_kernel if causal else flash_attn_fwd_kernel

    try:
        # Warmup: also triggers Triton JIT compilation
        for _ in range(WARMUP):
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
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
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
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / REPS
    except Exception:
        return float("inf")


def autotune(
    B: int = 2,
    H: int = 16,
    seq_lens: list = None,
    D: int = 64,
    causal: bool = True,
    save_path: str = None,
) -> dict:
    """
    Sweep BLOCK_M × BLOCK_N for each seqlen and return best configs.

    Returns:
        dict mapping seqlen → {"block_m": int, "block_n": int, "ms": float}
    """
    if seq_lens is None:
        seq_lens = [512, 1024, 2048, 4096]

    device_name = torch.cuda.get_device_name(0)
    print(f"Autotuning Flash Attention on {device_name}")
    print(f"B={B} H={H} D={D} causal={causal}")
    print()

    results = {}

    for M in seq_lens:
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        best_ms = float("inf")
        best_bm = best_bn = None

        print(f"  seqlen={M}:")
        for bm, bn in itertools.product(BLOCK_SIZES, BLOCK_SIZES):
            ms = _bench_config(q, k, v, bm, bn, causal)
            marker = " ← best" if ms < best_ms else ""
            if ms < float("inf"):
                print(f"    BLOCK_M={bm:3d} BLOCK_N={bn:3d}  {ms:.3f} ms{marker}")
            if ms < best_ms:
                best_ms = ms
                best_bm, best_bn = bm, bn

        results[M] = {"block_m": best_bm, "block_n": best_bn, "ms": best_ms}
        print(f"  → Best: BLOCK_M={best_bm} BLOCK_N={best_bn} ({best_ms:.3f} ms)\n")

    # Save to JSON
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), "configs.json")

    payload = {
        "device": device_name,
        "D": D,
        "causal": causal,
        "configs": {str(k): v for k, v in results.items()},
    }
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved configs to {save_path}")

    return results


def load_best_config(seqlen: int, D: int = 64, causal: bool = True):
    """
    Load the cached best (BLOCK_M, BLOCK_N) for the current GPU.
    Falls back to (64, 64) if configs.json doesn't exist or has no entry.
    """
    config_path = os.path.join(os.path.dirname(__file__), "configs.json")
    if not os.path.exists(config_path):
        return 64, 64

    with open(config_path) as f:
        data = json.load(f)

    # Check the config was built for the same GPU
    if data.get("device") != torch.cuda.get_device_name(0):
        return 64, 64

    configs = data.get("configs", {})
    # Find the closest cached seqlen
    cached_lens = [int(k) for k in configs]
    if not cached_lens:
        return 64, 64
    closest = min(cached_lens, key=lambda x: abs(x - seqlen))
    entry = configs[str(closest)]
    return entry["block_m"], entry["block_n"]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    autotune()
