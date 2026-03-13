"""
MFU 分析脚本：详细分析各序列长度下的 MFU 表现

用法：
    python benchmarks/profile_mfu.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.sparse_attention import sparse_attention, generate_random_sparse_mask
from python.benchmark import measure_mfu, attention_flops, sparse_attention_flops


def profile_all_configs():
    device = "cuda"
    dtype = torch.bfloat16
    sparsity = 0.75

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"稀疏度: {sparsity*100:.0f}%  精度: {dtype}")
    print(f"\n{'序列长度':>8} {'Batch':>6} {'延迟(ms)':>10} {'TFLOPS':>8} {'MFU%':>7}")
    print("-" * 50)

    configs = [
        (128, 64, 12, 64),
        (256, 32, 12, 64),
        (512, 16, 12, 64),
        (512, 32, 12, 64),
        (1024, 8, 12, 64),
    ]

    for N, B, H, D in configs:
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        mask = generate_random_sparse_mask(B, H, N, sparsity=sparsity, device=device)

        def fn():
            return sparse_attention(q, k, v, mask)

        m = measure_mfu(fn, B, H, N, D, sparsity=sparsity, dtype=dtype)
        print(f"{N:>8} {B:>6} {m['latency_ms']:>10.3f} {m['tflops']:>8.1f} {m['mfu_percent']:>7.1f}%")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    profile_all_configs()
