"""
对比测试：我们的实现 vs Flash Attention

用法：
    python benchmarks/compare_flash.py --seq_len 512 --batch_size 16
"""

import argparse
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python.sparse_attention import sparse_attention, sparse_attention_ref, generate_random_sparse_mask
from python.benchmark import measure_mfu

try:
    from flash_attn import flash_attn_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False
    print("[Warning] flash_attn 未安装，跳过对比")


def run_benchmark(B, H, N, D, sparsity=0.75, dtype=torch.bfloat16):
    device = "cuda"
    print(f"\n{'='*60}")
    print(f"配置: B={B}, H={H}, N={N}, D={D}, sparsity={sparsity}, dtype={dtype}")
    print(f"{'='*60}")

    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    mask = generate_random_sparse_mask(B, H, N, sparsity=sparsity, device=device)

    results = {}

    # ---- 我们的实现 ----
    def our_fn():
        return sparse_attention(q, k, v, mask)

    our_metrics = measure_mfu(our_fn, B, H, N, D, sparsity=sparsity, dtype=dtype)
    results["Ours"] = our_metrics
    print(f"[Ours]          latency={our_metrics['latency_ms']:.3f}ms  "
          f"MFU={our_metrics['mfu_percent']:.1f}%  "
          f"TFLOPS={our_metrics['tflops']:.1f}")

    # ---- Flash Attention（dense，不支持任意 mask）----
    if HAS_FLASH:
        # Flash Attention 用 causal=False，不传 mask（dense baseline）
        q_fa = q.transpose(1, 2).contiguous()  # [B, N, H, D]
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()

        def flash_fn():
            return flash_attn_func(q_fa, k_fa, v_fa, causal=False)

        flash_metrics = measure_mfu(flash_fn, B, H, N, D, sparsity=0.0, dtype=dtype)
        results["FlashAttn (dense)"] = flash_metrics
        print(f"[FlashAttn]     latency={flash_metrics['latency_ms']:.3f}ms  "
              f"MFU={flash_metrics['mfu_percent']:.1f}%  "
              f"TFLOPS={flash_metrics['tflops']:.1f}")

    # ---- PyTorch 参考实现 ----
    def ref_fn():
        return sparse_attention_ref(q.float(), k.float(), v.float(), mask)

    ref_metrics = measure_mfu(ref_fn, B, H, N, D, sparsity=sparsity, dtype=torch.float32,
                               warmup=3, repeat=10)
    results["PyTorch Ref"] = ref_metrics
    print(f"[PyTorch Ref]   latency={ref_metrics['latency_ms']:.3f}ms  "
          f"MFU={ref_metrics['mfu_percent']:.1f}%  "
          f"TFLOPS={ref_metrics['tflops']:.1f}")

    # ---- 加速比 ----
    if HAS_FLASH:
        speedup = flash_metrics["latency_ms"] / our_metrics["latency_ms"]
        print(f"\n相对 Flash Attention 加速比: {speedup:.2f}x")

    ref_speedup = ref_metrics["latency_ms"] / our_metrics["latency_ms"]
    print(f"相对 PyTorch Ref 加速比: {ref_speedup:.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--sparsity", type=float, default=0.75)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "需要 CUDA GPU"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")

    # 测试多个序列长度
    for N in [128, 256, 512]:
        run_benchmark(
            B=args.batch_size,
            H=args.num_heads,
            N=N,
            D=args.head_dim,
            sparsity=args.sparsity,
        )


if __name__ == "__main__":
    main()
