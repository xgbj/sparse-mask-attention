"""
setup.py - 编译 CUDA 扩展
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# 检查 CUDA 是否可用
assert torch.cuda.is_available(), "需要 CUDA 环境来编译"

# CUDA 编译选项
nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    # A100 架构
    "-gencode=arch=compute_80,code=sm_80",
    # 允许更多寄存器（提升 kernel 性能）
    "--maxrregcount=64",
]

cxx_flags = ["-O3", "-std=c++17"]

setup(
    name="sparse_mask_attention",
    version="0.1.0",
    description="高性能短序列稀疏 Mask Attention CUDA 算子",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name="sparse_attn_cuda",
            sources=["csrc/sparse_attention.cu"],
            include_dirs=["csrc"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
