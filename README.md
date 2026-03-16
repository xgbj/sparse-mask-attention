# Sparse Mask Attention — High-Performance CUDA Kernel

针对**短序列（<1K tokens）+ 高稀疏度（~75%）** 场景深度优化的稀疏掩码注意力 CUDA kernel。

经过 23 轮系统性优化，从 16.571ms 优化至 **0.512ms**（**32x 总提速**），在 RTX 3080 上达到 **25.17 TFLOPS / 84.4% MFU**。

## 性能对比

测试配置：B=16, H=12, N=512, D=64, FP16, sparsity=0.75, RTX 3080

| 实现 | 延迟 (ms) | TFLOPS | 加速比 |
|------|-----------|--------|--------|
| **Ours** | **0.512** | **25.17** | **baseline** |
| Triton | 0.751 | 17.15 | 本项目快 1.59x |
| cuDNN SDPA | 0.892 | 14.45 | 本项目快 1.74x |
| FlashInfer | 1.082 | 11.91 | 本项目快 2.11x |
| PyTorch Ref | 2.091 | 6.16 | 本项目快 4.42x |
| flash-attn (dense) | 0.403 | 31.96 | 差距 1.17x |

> flash-attn 为 dense 实现（无 mask），为理论上界参考。

![性能优化图](notes/perf_chart.png)

## 核心优化技术

- **Bit-Pack 掩码压缩** — bool mask `[B,H,N,N]` → uint32 `[B,H,N,N/32]`，8x 内存节省
- **WMMA Tensor Core** — m16n16k16 同时加速 QK^T 和 PV 累加
- **寄存器内 Softmax** — 利用 fragment layout（4 lanes/row）+ `__shfl_xor_sync` 跨 lane 归约，消除 smem round-trip
- **Shared Memory Padding** — 内维度 +8 halves，将 bank conflict 从 16-way 降至 2-way
- **cp.async 异步加载** — K/V 通过 `cp.async.cg` 走 L2 bypass 路径
- **Mask 预加载** — 128 线程协作将 mask bits 预读到 smem，消除全局内存逐元素访问

## 项目结构

```
sparse-mask-attention/
├── csrc/                          # CUDA 核心
│   ├── sparse_attention.cu        # WMMA FP16 kernel + BF16 fallback
│   ├── sparse_attention.h         # 参数结构体 + 函数声明
│   ├── binding.cpp                # pybind11 绑定
│   └── utils.cuh                  # warp reduce, mask helpers
├── python/                        # Python 接口
│   ├── sparse_attention.py        # sparse_attention() API + pack_mask()
│   └── __init__.py
├── notes/                         # 优化文档
│   ├── perf_log.md                # 23 轮逐轮优化记录
│   ├── optimization_rules.md      # 优化方法论
│   ├── gen_chart.py               # 生成性能图表
│   └── perf_chart.png             # 性能优化图
├── run.py                         # 统一测试入口（正确性/性能/横向对比）
├── setup.py                       # 编译安装
├── summary.md                     # 详细优化报告
└── CLAUDE.md                      # AI 编码辅助指引
```

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

## 使用

### Python API

```python
import torch
from python.sparse_attention import sparse_attention

B, H, N, D = 16, 12, 512, 64
q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
mask = torch.rand(B, H, N, N, device='cuda') > 0.75  # 75% sparsity

output = sparse_attention(q, k, v, mask)
```

### 测试与基准

```bash
python run.py                       # 正确性 + 性能（默认）
python run.py --mode correctness    # 仅正确性验证
python run.py --mode perf           # 性能测试 + baseline 对比
python run.py --mode compare        # 全面横向对比所有后端
```

## 数据流

```
Q, K, V [B,H,N,D] + bool mask [B,H,N,N]
  → pack_mask_kernel → uint32 mask [B,H,N,N/32]
  → sparse_attn_wmma_full_fp16 (register softmax, tile skipping)
  → output [B,H,N,D]
```

## Kernel 参数

| 参数 | 值 |
|------|-----|
| BM (Q tile rows) | 16 |
| BN (K/V tile cols) | 64 |
| Head dim | 64 |
| Warps/block | 4 (128 threads) |
| Smem/block | ~37 KB |
| Occupancy | 2 blocks/SM |
| I/O 精度 | FP16 |
| 累加精度 | FP32 |

## License

MIT License
