/*
 * csrc/bindings.cpp
 * =================
 * Thin pybind11 glue between Python / PyTorch and the CUDA kernels.
 * Argument validation lives here so the .cu files stay pure CUDA.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Forward declaration (defined in fused_rmsnorm_linear.cu).
void launch_fused_rmsnorm_linear(
    const at::Half* X,
    const at::Half* W_norm,
    const at::Half* W_lin,
    at::Half*       Y,
    float eps,
    int B, int D_in, int D_out,
    cudaStream_t stream
);

// ---------------------------------------------------------------------------
// fused_rmsnorm_linear
// ---------------------------------------------------------------------------

torch::Tensor fused_rmsnorm_linear(
    torch::Tensor X,        // [B, D_in]      fp16, CUDA
    torch::Tensor W_norm,   // [D_in]          fp16, CUDA
    torch::Tensor W_lin,    // [D_out, D_in]   fp16, CUDA
    float eps
) {
    // ── validation ────────────────────────────────────────────────────────
    TORCH_CHECK(X.is_cuda(),     "X must be a CUDA tensor");
    TORCH_CHECK(W_norm.is_cuda(),"W_norm must be a CUDA tensor");
    TORCH_CHECK(W_lin.is_cuda(), "W_lin must be a CUDA tensor");
    TORCH_CHECK(X.dtype()     == torch::kFloat16, "X must be fp16");
    TORCH_CHECK(W_norm.dtype()== torch::kFloat16, "W_norm must be fp16");
    TORCH_CHECK(W_lin.dtype() == torch::kFloat16, "W_lin must be fp16");
    TORCH_CHECK(X.dim() == 2,    "X must be 2D [B, D_in]");
    TORCH_CHECK(W_norm.dim()==1, "W_norm must be 1D [D_in]");
    TORCH_CHECK(W_lin.dim() == 2,"W_lin must be 2D [D_out, D_in]");

    int B    = X.size(0);
    int D_in = X.size(1);
    int D_out= W_lin.size(0);

    TORCH_CHECK(W_norm.size(0) == D_in, "W_norm size mismatch");
    TORCH_CHECK(W_lin.size(1)  == D_in, "W_lin inner-dim mismatch");
    TORCH_CHECK(D_in * 4 <= 49152,
        "D_in too large for shared memory — add tiling across the hidden dim");

    // ── output allocation ─────────────────────────────────────────────────
    auto Y = torch::empty({B, D_out}, X.options());

    // ── launch ────────────────────────────────────────────────────────────
    launch_fused_rmsnorm_linear(
        X.data_ptr<at::Half>(),
        W_norm.data_ptr<at::Half>(),
        W_lin.data_ptr<at::Half>(),
        Y.data_ptr<at::Half>(),
        eps,
        B, D_in, D_out,
        c10::cuda::getCurrentCUDAStream()
    );

    return Y;
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom LLM CUDA kernels";

    m.def(
        "fused_rmsnorm_linear",
        &fused_rmsnorm_linear,
        "Fused RMSNorm + Linear forward pass (fp16, CUDA)",
        py::arg("X"),
        py::arg("W_norm"),
        py::arg("W_lin"),
        py::arg("eps") = 1e-6f
    );
}
