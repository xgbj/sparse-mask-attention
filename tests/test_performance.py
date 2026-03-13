"""
性能测试：验证各配置下的延迟和吞吐量

用法：
    python tests/test_performance.py
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.sparse_attention import sparse_attention, generate_random_sparse_mask
from python.benchmark import measure_mfu


def test_latency_scaling():
    """测试延迟随序列长度的缩放关系"""
    device = "cuda"
    dtype = torch.bfloat16
    B, H, D = 16, 12, 64
    sparsity = 0.75

    print("=== 延迟缩放测试 ===")
    print(f"{'N':>6} {'latency(ms)':>12} {'MFU%':>8}")
    print("-" * 30)

    for N in [64, 128, 256, 512, 1024]:
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        mask = generate_random_sparse_mask(B, H, N, sparsity=sparsity, device=device)

        m = measure_mfu(
            lambda: sparse_attention(q, k, v, mask),
            B, H, N, D, sparsity=sparsity, dtype=dtype
        )
        print(f"{N:>6} {m['latency_ms']:>12.3f} {m['mfu_percent']:>8.1f}%")


def test_batch_scaling():
    """测试延迟随 batch size 的缩放关系"""
    device = "cuda"
    dtype = torch.bfloat16
    H, N, D = 12, 512, 64
    sparsity = 0.75

    print("\n=== Batch Size 缩放测试（N=512）===")
    print(f"{'B':>6} {'latency(ms)':>12} {'MFU%':>8} {'throughput(seq/s)':>18}")
    print("-" * 50)

    for B in [1, 4, 8, 16, 32, 64]:
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        mask = generate_random_sparse_mask(B, H, N, sparsity=sparsity, device=device)

        m = measure_mfu(
            lambda: sparse_attention(q, k, v, mask),
            B, H, N, D, sparsity=sparsity, dtype=dtype
        )
        throughput = B / (m['latency_ms'] / 1000.0)
        print(f"{B:>6} {m['latency_ms']:>12.3f} {m['mfu_percent']:>8.1f}% {throughput:>18.1f}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "需要 CUDA GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    test_latency_scaling()
    test_batch_scaling()
