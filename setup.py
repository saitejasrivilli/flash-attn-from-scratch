"""
setup.py
========
Builds the CUDA extension (fused_rmsnorm_linear) and packages all kernels.

Build commands
--------------
# Development install (editable, recompiles on change):
    pip install -e . --no-build-isolation

# Production wheel:
    pip wheel . --no-build-isolation

# Rebuild just the extension after editing .cu files:
    python setup.py build_ext --inplace

Environment notes
-----------------
- A30 / A100 / 3090 → SM86:  set TORCH_CUDA_ARCH_LIST="8.6"
- V100               → SM70:  set TORCH_CUDA_ARCH_LIST="7.0"
- T4  (Colab)        → SM75:  set TORCH_CUDA_ARCH_LIST="7.5"
  e.g.: TORCH_CUDA_ARCH_LIST="7.5" pip install -e . --no-build-isolation

Add -lineinfo for Nsight Compute source correlation during development.
"""

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ── detect target arch ─────────────────────────────────────────────────────
# Falls back to SM86 (A30 / RTX 3090) if not set.
arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "8.6")

nvcc_flags = [
    "-O3",
    f"-arch=sm_{arch.replace('.', '')}",
    "--use_fast_math",
    "-lineinfo",          # source correlation in Nsight Compute
    "--expt-relaxed-constexpr",
]

setup(
    name="llm_kernels",
    version="0.1.0",
    description="Custom LLM GPU kernels: Flash Attention, fused RMSNorm+Linear, int8 GEMM",
    author="Your Name",
    python_requires=">=3.9",
    packages=find_packages(exclude=["benchmarks", "tests"]),
    ext_modules=[
        CUDAExtension(
            name="llm_kernels_cuda",          # import name for the .so
            sources=[
                "csrc/bindings.cpp",
                "csrc/fused_rmsnorm_linear.cu",
            ],
            extra_compile_args={
                "cxx":  ["-O3", "-std=c++17"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.2.0",
        "triton>=2.2.0",
    ],
    extras_require={
        "bench": ["tabulate", "matplotlib"],
        "dev":   ["pytest", "black", "isort"],
    },
)
