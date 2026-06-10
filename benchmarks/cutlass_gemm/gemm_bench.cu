/*
 * CUTLASS fp16 GEMM vs cuBLAS benchmark — NVIDIA A30 (SM86, Ampere)
 * Measures GFLOPS at multiple GEMM sizes representative of LLM weight matrices.
 * Reference: cutlass/examples/00_basic_gemm/basic_gemm.cu
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#define CUDA_CHECK(x) do { \
  cudaError_t e = (x); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while(0)

#define CUBLAS_CHECK(x) do { \
  cublasStatus_t s = (x); \
  if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
    exit(1); \
  } \
} while(0)

#define CUTLASS_CHECK(x) do { \
  cutlass::Status s = (x); \
  if (s != cutlass::Status::kSuccess) { \
    fprintf(stderr, "CUTLASS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
    exit(1); \
  } \
} while(0)

// Ampere (SM80/SM86) CUTLASS fp16 GEMM configuration
// ThreadblockShape: 128x256x32, WarpShape: 64x64x32, Stages: 3
using ElementA    = cutlass::half_t;
using ElementB    = cutlass::half_t;
using ElementC    = cutlass::half_t;
using ElementAccum = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,   // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,      // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,       // InstructionShape (Ampere TensorCore)
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 8, ElementAccum, ElementAccum>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3   // Stages
>;

// --------------------------------------------------------------------------
// Host helpers
// --------------------------------------------------------------------------

__half *alloc_fp16(int n) {
    __half *p;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(__half)));
    return p;
}

void fill_random_fp16(__half *dev, int n, float scale = 0.1f) {
    std::vector<__half> h(n);
    for (int i = 0; i < n; ++i)
        h[i] = __float2half(scale * ((float)rand() / RAND_MAX - 0.5f));
    CUDA_CHECK(cudaMemcpy(dev, h.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

// --------------------------------------------------------------------------
// Benchmark one GEMM size
// --------------------------------------------------------------------------

struct Result {
    int M, N, K;
    double cutlass_gflops;
    double cublas_gflops;
    double speedup;
    bool   correct;
};

Result bench(int M, int N, int K, cublasHandle_t cublas_handle,
             int warmup = 20, int iters = 100) {

    long long flops = 2LL * M * N * K;

    __half *dA = alloc_fp16(M * K);
    __half *dB = alloc_fp16(K * N);
    __half *dC_cutlass = alloc_fp16(M * N);
    __half *dC_cublas  = alloc_fp16(M * N);

    fill_random_fp16(dA, M * K);
    fill_random_fp16(dB, K * N);
    CUDA_CHECK(cudaMemset(dC_cutlass, 0, M * N * sizeof(__half)));
    CUDA_CHECK(cudaMemset(dC_cublas,  0, M * N * sizeof(__half)));

    // ── CUTLASS setup ──────────────────────────────────────────────────
    Gemm::Arguments args(
        {M, N, K},
        {(cutlass::half_t*)dA, K},
        {(cutlass::half_t*)dB, K},
        {(cutlass::half_t*)dC_cutlass, N},
        {(cutlass::half_t*)dC_cutlass, N},
        {ElementAccum(1), ElementAccum(0)}
    );

    Gemm gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(args));

    size_t workspace_size = Gemm::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    // warmup CUTLASS
    for (int i = 0; i < warmup; ++i) {
        CUTLASS_CHECK(gemm_op(args, workspace));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // time CUTLASS
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iters; ++i)
        CUTLASS_CHECK(gemm_op(args, workspace));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float cutlass_ms;
    CUDA_CHECK(cudaEventElapsedTime(&cutlass_ms, ev_start, ev_stop));
    double cutlass_gflops = (double)flops * iters / (cutlass_ms * 1e6);

    // ── cuBLAS setup ───────────────────────────────────────────────────
    // C = A*B  (row-major A [M,K], row-major B [K,N], row-major C [M,N])
    // cuBLAS is column-major: C^T = B^T * A^T
    __half alpha_h = __float2half(1.0f);
    __half beta_h  = __float2half(0.0f);

    // warmup cuBLAS
    for (int i = 0; i < warmup; ++i) {
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha_h,
            dB, CUDA_R_16F, N,
            dA, CUDA_R_16F, K,
            &beta_h,
            dC_cublas, CUDA_R_16F, N,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iters; ++i) {
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha_h,
            dB, CUDA_R_16F, N,
            dA, CUDA_R_16F, K,
            &beta_h,
            dC_cublas, CUDA_R_16F, N,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float cublas_ms;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, ev_start, ev_stop));
    double cublas_gflops = (double)flops * iters / (cublas_ms * 1e6);

    // ── Correctness check (sample 256 elements) ────────────────────────
    std::vector<__half> h_cutlass(M * N), h_cublas(M * N);
    CUDA_CHECK(cudaMemcpy(h_cutlass.data(), dC_cutlass, M*N*sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cublas.data(),  dC_cublas,  M*N*sizeof(__half), cudaMemcpyDeviceToHost));
    int check_n = std::min(256, M * N);
    float max_err = 0.0f;
    for (int i = 0; i < check_n; ++i) {
        float a = __half2float(h_cutlass[i]);
        float b = __half2float(h_cublas[i]);
        max_err = std::max(max_err, std::abs(a - b));
    }
    bool correct = (max_err < 0.5f);  // fp16 accumulation tolerance

    // cleanup
    CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC_cutlass)); CUDA_CHECK(cudaFree(dC_cublas));
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaEventDestroy(ev_start)); CUDA_CHECK(cudaEventDestroy(ev_stop));

    Result r;
    r.M = M; r.N = N; r.K = K;
    r.cutlass_gflops = cutlass_gflops;
    r.cublas_gflops  = cublas_gflops;
    r.speedup = cutlass_gflops / cublas_gflops;
    r.correct = correct;
    return r;
}

int main() {
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM%d%d, %.1f GB HBM)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1e9);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // LLM-representative GEMM sizes (M=batch, N=hidden, K=hidden/ffn)
    // Focus on shapes that appear in Transformer forward pass
    struct Shape { int M, N, K; const char* label; };
    std::vector<Shape> shapes = {
        {1,    4096, 4096,  "decode B=1 (4096x4096)"},
        {8,    4096, 4096,  "decode B=8 (4096x4096)"},
        {32,   4096, 4096,  "small batch B=32"},
        {128,  4096, 4096,  "prefill B=128"},
        {512,  4096, 4096,  "large prefill B=512"},
        {128,  4096, 11008, "FFN gate/up (LLaMA-2-7B)"},
        {128,  11008,4096,  "FFN down (LLaMA-2-7B)"},
    };

    printf("\n%-35s  %10s  %10s  %8s  %8s\n",
           "Shape", "CUTLASS", "cuBLAS", "Speedup", "Pass?");
    printf("%-35s  %10s  %10s  %8s  %8s\n",
           "-----------------------------------",
           "GFLOPS", "GFLOPS", "ratio", "");
    printf("─────────────────────────────────────────────────────────────────────────────\n");

    std::vector<Result> results;
    for (auto& s : shapes) {
        Result r = bench(s.M, s.N, s.K, handle);
        results.push_back(r);
        printf("%-35s  %10.1f  %10.1f  %7.2fx  %s\n",
               s.label,
               r.cutlass_gflops, r.cublas_gflops,
               r.speedup,
               r.correct ? "PASS" : "FAIL");
        fflush(stdout);
    }

    // JSON output
    FILE* fj = fopen("results.json", "w");
    if (fj) {
        fprintf(fj, "{\n  \"device\": \"%s\",\n  \"sm\": \"%d%d\",\n  \"warmup\": 20,\n  \"iters\": 100,\n  \"results\": [\n", prop.name, prop.major, prop.minor);
        for (size_t i = 0; i < results.size(); ++i) {
            auto& r = results[i];
            fprintf(fj, "    {\"M\":%d,\"N\":%d,\"K\":%d,"
                    "\"cutlass_gflops\":%.1f,"
                    "\"cublas_gflops\":%.1f,"
                    "\"speedup\":%.4f,"
                    "\"correct\":%s}%s\n",
                    r.M, r.N, r.K,
                    r.cutlass_gflops, r.cublas_gflops,
                    r.speedup,
                    r.correct ? "true" : "false",
                    i + 1 < results.size() ? "," : "");
        }
        fprintf(fj, "  ]\n}\n");
        fclose(fj);
        printf("\nResults saved to results.json\n");
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
