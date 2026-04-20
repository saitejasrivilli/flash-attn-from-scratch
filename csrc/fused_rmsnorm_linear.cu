/*
 * fused_rmsnorm_linear.cu
 * =======================
 * One CUDA block handles one token (row).
 *
 * Pipeline (all in shared memory — zero HBM round-trip between norm and linear):
 *   1. Load row into shared mem, accumulate sum-of-squares.
 *   2. Warp-reduce → RMS → normalize + apply learnable weight in-place.
 *   3. Compute linear projection: each thread owns a slice of D_out.
 *
 * Constraints:
 *   - D_in must fit in shared mem (≤ 49 152 bytes on A30 → max 12 288 fp32 elems).
 *   - fp16 I/O, fp32 accumulation.
 *   - For D_in > 8 192 add an inner tile loop across the hidden dim.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Full-warp reduction (assumes 32 threads per warp, one warp per call site).
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int THREADS>
__global__ void fused_rmsnorm_linear_kernel(
    const half* __restrict__ X,        // [B, D_in]      input
    const half* __restrict__ W_norm,   // [D_in]          RMSNorm weight (gamma)
    const half* __restrict__ W_lin,    // [D_out, D_in]   linear weight (row-major)
    half*       __restrict__ Y,        // [B, D_out]      output
    float eps,
    int D_in,
    int D_out
) {
    int row = blockIdx.x;   // one block per token
    int tid = threadIdx.x;

    // Shared memory layout: [D_in] fp32 normalized activations.
    extern __shared__ float smem[];
    float* x_shared = smem;            // size = D_in floats

    // ── Step 1: load row & accumulate sum-of-squares ─────────────────────
    float ss = 0.f;
    for (int i = tid; i < D_in; i += THREADS) {
        float xi    = __half2float(X[row * D_in + i]);
        x_shared[i] = xi;
        ss          += xi * xi;
    }

    // Reduce ss across all threads in the block.
    // For THREADS <= 32 a single warp suffices; for larger blocks we need
    // shared memory staging too.
    __shared__ float ss_stage[32];     // one slot per warp

    int lane   = tid & 31;
    int warp_id = tid >> 5;
    ss = warp_reduce_sum(ss);
    if (lane == 0) ss_stage[warp_id] = ss;
    __syncthreads();

    // Let thread 0 do the final reduction across warps.
    if (tid == 0) {
        float total = 0.f;
        int n_warps = (THREADS + 31) / 32;
        for (int w = 0; w < n_warps; w++) total += ss_stage[w];
        ss_stage[0] = rsqrtf(total / D_in + eps);   // store rms_inv
    }
    __syncthreads();
    float rms_inv = ss_stage[0];

    // ── Step 2: normalize + apply learnable weight ───────────────────────
    for (int i = tid; i < D_in; i += THREADS) {
        float w     = __half2float(W_norm[i]);
        x_shared[i] = x_shared[i] * rms_inv * w;
    }
    __syncthreads();

    // ── Step 3: linear projection — each thread computes a slice of D_out ─
    //   y[j] = dot(x_normalized, W_lin[j, :])
    for (int j = tid; j < D_out; j += THREADS) {
        float acc = 0.f;
        const half* w_row = W_lin + (long long)j * D_in;
        for (int i = 0; i < D_in; i++)
            acc += x_shared[i] * __half2float(w_row[i]);
        Y[row * D_out + j] = __float2half(acc);
    }
}

// ---------------------------------------------------------------------------
// Public launcher — called from bindings.cpp
// ---------------------------------------------------------------------------

void launch_fused_rmsnorm_linear(
    const half* X,
    const half* W_norm,
    const half* W_lin,
    half*       Y,
    float eps,
    int B, int D_in, int D_out,
    cudaStream_t stream
) {
    const int THREADS = 256;
    size_t smem_bytes = (size_t)D_in * sizeof(float);

    fused_rmsnorm_linear_kernel<THREADS><<<B, THREADS, smem_bytes, stream>>>(
        X, W_norm, W_lin, Y, eps, D_in, D_out
    );
}
