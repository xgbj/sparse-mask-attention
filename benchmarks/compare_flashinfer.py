"""
对比测试：我们的实现 vs FlashInfer

用法：
    python benchmarks/compare_flashinfer.py --seq_len 512 --batch_size 16
"""

import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.sparse_attention import sparse_attention, generate_random_sparse_mask
from python.benchmark import measure_mfu

try:
    import flashinfer
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False
    print("[Warning] flashinfer 未安装，跳过对比")


def run_benchmark(B, H, N, D, sparsity=0.75, dtype=torch.bfloat16):
    device = "cuda"
    print(f"\n{'='*60}")
    print(f"配置: B={B}, H={H}, N={N}, D={D}, sparsity={sparsity}")
    print(f"{'='*60}")

    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    mask = generate_random_sparse_mask(B, H, N, sparsity=sparsity, device=device)

    # ---- 我们的实现 ----
    def our_fn():
        return sparse_attention(q, k, v, mask)

    our_metrics = measure_mfu(our_fn, B, H, N, D, sparsity=sparsity, dtype=dtype)
    print(f"[Ours]          latency={our_metrics['latency_ms']:.3f}ms  "
          f"MFU={our_metrics['mfu_percent']:.1f}%")

    # ---- FlashInfer（batch prefill，dense baseline）----
    if HAS_FLASHINFER:
        # FlashInfer 使用 [B*N, H, D] 格式
        q_fi = q.reshape(B * N, H, D).contiguous()
        k_fi = k.reshape(B * N, H, D).contiguous()
        v_fi = v.reshape(B * N, H, D).contiguous()

        # 构建 qo_indptr 和 kv_indptr
        qo_indptr = torch.arange(0, (B + 1) * N, N, device=device, dtype=torch.int32)
        kv_indptr = torch.arange(0, (B + 1) * N, N, device=device, dtype=torch.int32)

        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        )

        def fi_fn():
            wrapper.begin_forward(qo_indptr, kv_indptr, H, H, D)
            return wrapper.forward(q_fi, k_fi, v_fi, causal=False)

        fi_metrics = measure_mfu(fi_fn, B, H, N, D, sparsity=0.0, dtype=dtype)
        print(f"[FlashInfer]    latency={fi_metrics['latency_ms']:.3f}ms  "
              f"MFU={fi_metrics['mfu_percent']:.1f}%")

        speedup = fi_metrics["latency_ms"] / our_metrics["latency_ms"]
        print(f"相对 FlashInfer 加速比: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--sparsity", type=float, default=0.75)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA GPU"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    for N in [128, 256, 512]:
        run_benchmark(args.batch_size, args.num_heads, N, args.head_dim, args.sparsity)


if __name__ == "__main__":
    main()
