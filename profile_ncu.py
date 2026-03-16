# -*- coding: utf-8 -*-
"""Minimal ncu profiling script: single kernel launch"""
import math, os, torch
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.abspath(__file__))
props = torch.cuda.get_device_properties(0)
arch = f"{props.major}{props.minor}"

sparse_attn_cuda = load(
    name="sparse_attn_cuda",
    sources=[
        os.path.join(ROOT, "csrc/binding.cpp"),
        os.path.join(ROOT, "csrc/sparse_attention.cu"),
    ],
    extra_include_paths=[os.path.join(ROOT, "csrc")],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
    ],
    extra_cflags=["-O3"],
    verbose=False,
)

B, H, N, D = 16, 12, 512, 64
dtype = torch.float16
device = "cuda"

q = torch.randn(B, H, N, D, device=device, dtype=dtype)
k = torch.randn(B, H, N, D, device=device, dtype=dtype)
v = torch.randn(B, H, N, D, device=device, dtype=dtype)
mask = torch.rand(B, H, N, N, device=device) > 0.75

mask_packed = sparse_attn_cuda.pack_mask(mask)
out = torch.empty_like(q)
lse = torch.empty(B, H, N, device=device, dtype=torch.float32)
scale = 1.0 / math.sqrt(D)

# warmup
for _ in range(3):
    sparse_attn_cuda.forward(q, k, v, mask_packed, out, lse, scale, False)
torch.cuda.synchronize()

# single launch for profiling
sparse_attn_cuda.forward(q, k, v, mask_packed, out, lse, scale, False)
torch.cuda.synchronize()
print("done")
