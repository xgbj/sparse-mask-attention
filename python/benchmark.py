"""
MFU 计算工具

计算 Model FLOPs Utilization，用于评估 GPU 利用率。
"""

import torch
import time
import math


def get_a100_peak_tflops(dtype=torch.bfloat16):
    """返回 A100 的理论峰值算力（TFLOPS）"""
    if dtype in (torch.float16, torch.bfloat16):
        return 312.0  # A100 BF16/FP16 tensor core TFLOPS
    elif dtype == torch.float32:
        return 19.5   # A100 FP32 TFLOPS
    else:
        return 312.0


def attention_flops(B, H, N, D, is_training=False):
    """
    计算注意力的理论 FLOPs。

    前向：
      - QK^T: 2 * B * H * N * N * D
      - softmax: ~5 * B * H * N * N（近似）
      - AV:  2 * B * H * N * N * D
    反向约为前向的 2.5x。
    """
    fwd_flops = 4 * B * H * N * N * D  # QK^T + AV，忽略 softmax
    if is_training:
        return fwd_flops * 3.5  # 前向 + 反向
    return fwd_flops


def sparse_attention_flops(B, H, N, D, sparsity=0.75, is_training=False):
    """
    稀疏注意力的有效 FLOPs（跳过 masked 块后）。
    sparsity=0.75 表示 75% 的计算被跳过。
    """
    dense_flops = attention_flops(B, H, N, D, is_training)
    return dense_flops * (1.0 - sparsity)


def measure_mfu(fn, B, H, N, D, sparsity=0.75, is_training=False,
                warmup=10, repeat=50, dtype=torch.bfloat16):
    """
    测量函数的 MFU。

    参数：
        fn: 要测量的函数，无参数调用
        B, H, N, D: 注意力参数
        sparsity: mask 稀疏度
        is_training: 是否包含反向传播
        warmup: 预热次数
        repeat: 测量次数
    返回：
        dict: {latency_ms, tflops, mfu}
    """
    # 预热
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / repeat
    elapsed_s = elapsed_ms / 1000.0

    # 计算 FLOPs
    flops = sparse_attention_flops(B, H, N, D, sparsity, is_training)
    tflops = flops / elapsed_s / 1e12

    peak = get_a100_peak_tflops(dtype)
    mfu = tflops / peak * 100.0

    return {
        "latency_ms": elapsed_ms,
        "tflops": tflops,
        "mfu_percent": mfu,
        "peak_tflops": peak,
    }
