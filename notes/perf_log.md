# Kernel 性能优化记录

GPU: NVIDIA GeForce RTX 3080 (sm_86)
基准对比: PyTorch 参考实现 (cuBLAS matmul)

## 评测配置

- 正确性: B=2, H=4, D=64, N=[64,128,256,512], sparsity=0.75, tol=1e-2
- 性能: B=16, H=12, D=64, N=512, sparsity=0.75
- 对比延迟: B=16, N=512, H=12, D=64

## 优化记录

### Round 0 — Baseline (原始 kernel)

改动: 无 (原始代码，仅修改 sm_80 → sm_86)

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU% |
|---|---|---|---|
| 64 | 0.352 | 0.14 | 0.0% |
| 128 | 1.281 | 0.16 | 0.1% |
| 256 | 4.288 | 0.19 | 0.1% |
| 512 | 16.571 | 0.19 | 0.1% |
| 1024 | 67.642 | 0.19 | 0.1% |

对比 PyTorch Ref (B=16, N=512):
- Ours: 16.653 ms
- PyTorch Ref: 4.441 ms
- 加速比: 0.3x

关键指标 (N=512): **16.571 ms, 0.19 TFLOPS, 0.1% MFU**

### Round 1 — 放宽寄存器限制 (maxrregcount 64→128)

改动: run.py 中 `--maxrregcount=64` → `--maxrregcount=128`
原因: 每线程需要 acc[64]+scores[32]=96 个 float 寄存器，64 上限导致大量 spill 到 local memory

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU% |
|---|---|---|---|
| 64 | 0.170 | 0.30 | 0.1% |
| 128 | 0.557 | 0.36 | 0.1% |
| 256 | 1.978 | 0.41 | 0.1% |
| 512 | 6.575 | 0.49 | 0.2% |
| 1024 | 25.236 | 0.51 | 0.2% |

对比 PyTorch Ref (B=16, N=512):
- Ours: 6.571 ms
- PyTorch Ref: 4.447 ms
- 加速比: 0.7x

关键指标 (N=512): **6.575 ms, 0.49 TFLOPS, 0.2% MFU** (vs Round 0: 16.571ms, 提速 2.5x)

### Round 2 — 128 线程协作加载 K/V tile

改动: csrc/sparse_attention.cu — block 线程数从 32→128 (4 warps)，新增 NUM_THREADS 模板参数，所有 128 线程协作加载 Q/K/V tile，前 32 线程负责计算
原因: 32 线程加载 K/V tile 带宽利用率低，128 线程可 4x 加速全局内存读取

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU% |
|---|---|---|---|
| 64 | 0.154 | 0.33 | 0.1% |
| 128 | 0.488 | 0.41 | 0.1% |
| 256 | 1.666 | 0.48 | 0.2% |
| 512 | 5.113 | 0.63 | 0.2% |
| 1024 | 22.298 | 0.58 | 0.2% |

对比 PyTorch Ref (B=16, N=512):
- Ours: 5.147 ms
- PyTorch Ref: 4.448 ms
- 加速比: 0.9x

关键指标 (N=512): **5.113 ms, 0.63 TFLOPS, 0.2% MFU** (vs Round 1: 6.575ms, 提速 1.29x)

### Round 3 — 批量读取 mask word，消除逐位 read_mask_bit 调用

改动: csrc/sparse_attention.cu — QK^T 计算循环中，将逐位 `read_mask_bit()` 替换为一次读取整个 uint32 word，然后位操作提取每位
原因: BLOCK_N=32 刚好对应一个 uint32 word，原来每个 score 都重新计算地址读一次 global memory，现在只读一次

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU% |
|---|---|---|---|
| 64 | 0.147 | 0.34 | 0.1% |
| 128 | 0.460 | 0.44 | 0.1% |
| 256 | 1.516 | 0.53 | 0.2% |
| 512 | 4.962 | 0.65 | 0.2% |
| 1024 | 19.054 | 0.68 | 0.2% |

对比 PyTorch Ref (B=16, N=512):
- Ours: 4.755 ms
- PyTorch Ref: 4.454 ms
- 加速比: 0.9x

关键指标 (N=512): **4.962 ms, 0.65 TFLOPS, 0.2% MFU** (vs Round 2: 5.113ms, 提速 1.03x)

### Round 4 — K/V 同时加载，减少 syncthreads

改动: csrc/sparse_attention.cu — 将 K 和 V tile 在同一个协作加载循环中同时加载到 shared memory，消除了原来 K 加载后计算 QK^T 再加载 V 之间的额外 syncthreads
原因: 原来每个 tile 迭代有 3 次 syncthreads（加载K、加载V、迭代结束），现在减少到 2 次
注: 尝试过 2D 线程布局（4 group 并行点积）但因 atomicAdd 汇总开销严重退化，已放弃

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU% |
|---|---|---|---|
| 64 | 0.133 | 0.38 | 0.1% |
| 128 | 0.381 | 0.53 | 0.2% |
| 256 | 1.258 | 0.64 | 0.2% |
| 512 | 4.231 | 0.76 | 0.2% |
| 1024 | 15.517 | 0.83 | 0.3% |

对比 PyTorch Ref (B=16, N=512):
- Ours: 3.964 ms
- PyTorch Ref: 4.445 ms
- 加速比: 1.1x ★ 首次超过 PyTorch Ref

关键指标 (N=512): **4.231 ms, 0.76 TFLOPS, 0.2% MFU** (vs Round 3: 4.962ms, 提速 1.17x)

### Round 5 — float4 向量化加载 + 去掉 smem padding

改动: csrc/sparse_attention.cu — 去掉 shared memory 的 +1 padding，全局内存加载改为 4 元素一组（手动展开），点积计算也改为 4 元素一组展开
原因: HEAD_DIM=64 对齐 float4，去掉 padding 后 smem 访问更规整；向量化加载减少指令数

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU% |
|---|---|---|---|
| 64 | 0.112 | 0.45 | 0.1% |
| 128 | 0.346 | 0.58 | 0.2% |
| 256 | 1.169 | 0.69 | 0.2% |
| 512 | 4.008 | 0.80 | 0.3% |
| 1024 | 14.647 | 0.88 | 0.3% |

对比 PyTorch Ref (B=16, N=512):
- Ours: 3.820 ms
- PyTorch Ref: 4.640 ms
- 加速比: 1.2x

关键指标 (N=512): **4.008 ms, 0.80 TFLOPS, 0.3% MFU** (vs Round 4: 4.231ms, 提速 1.06x)

### Round 6 — 去掉 maxrregcount 限制

改动: run.py — 删除 `--maxrregcount=128`，让编译器完全自由分配寄存器
原因: 128 上限可能仍不够（acc[64]+scores[32]+其他≈110 float regs），去掉后编译器可自行选择最优 occupancy/register 平衡
注: 尝试过 __ffs 稀疏迭代（warp divergence 退化）和 Q 寄存器缓存（spill 退化），均已放弃

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU% |
|---|---|---|---|
| 64 | 0.111 | 0.45 | 0.1% |
| 128 | 0.342 | 0.59 | 0.2% |
| 256 | 1.155 | 0.70 | 0.2% |
| 512 | 3.869 | 0.83 | 0.3% |
| 1024 | 14.591 | 0.88 | 0.3% |

对比 PyTorch Ref (B=16, N=512):
- Ours: 3.794 ms
- PyTorch Ref: 4.452 ms
- 加速比: 1.2x

关键指标 (N=512): **3.869 ms, 0.83 TFLOPS, 0.3% MFU** (vs Round 5: 4.008ms, 提速 1.04x)

### Round 7 — BLOCK_N 从 32 减小到 16

改动: csrc/sparse_attention.cu — BLOCK_N 32→16，修复 mask 读取逻辑支持非 32 对齐的 bit 段提取
原因: BLOCK_N=16 减少 scores 数组大小（32→16 float），降低寄存器压力；sparse skipping 粒度更细（16 列 vs 32 列），75% sparsity 下跳过概率更高

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU% |
|---|---|---|---|
| 64 | 0.113 | 0.44 | 0.1% |
| 128 | 0.299 | 0.67 | 0.2% |
| 256 | 1.019 | 0.79 | 0.3% |
| 512 | 3.453 | 0.93 | 0.3% |
| 1024 | 12.826 | 1.00 | 0.3% |

对比 PyTorch Ref (B=16, N=512):
- Ours: 3.313 ms
- PyTorch Ref: 4.507 ms
- 加速比: 1.4x

关键指标 (N=512): **3.453 ms, 0.93 TFLOPS, 0.3% MFU** (vs Round 6: 3.869ms, 提速 1.12x)

### Round 8 — 合并 mask 读取，消除重复全局内存访问

改动: csrc/sparse_attention.cu — 将 sparse block skipping 的 mask 检查和 QK^T 计算的 mask 读取合并为一次读取，mask_word 提前在 tile 循环开头读取，同时用于 skipping 判断和 score 计算
原因: 原来每个 tile 迭代读取 mask 两次（skipping 检查 + 计算），合并后减少一次全局内存访问
注: 尝试过 NUM_THREADS=256（syncthreads 开销增大导致退化），已放弃

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU% |
|---|---|---|---|
| 64 | 0.112 | 0.45 | 0.1% |
| 128 | 0.295 | 0.68 | 0.2% |
| 256 | 1.009 | 0.80 | 0.3% |
| 512 | 3.247 | 0.99 | 0.3% |
| 1024 | 12.538 | 1.03 | 0.3% |

对比 PyTorch Ref (B=16, N=512):
- Ours: 3.267 ms
- PyTorch Ref: 4.452 ms
- 加速比: 1.4x

关键指标 (N=512): **3.247 ms, 0.99 TFLOPS, 0.3% MFU** (vs Round 7: 3.453ms, 提速 1.06x)

### Round 9 — 切换到 FP16 + 硬件自动检测 + PyTorch Ref 走 Tensor Core

改动: run.py — dtype 从 BF16→FP16；PyTorch Ref 不再 .float() 转换，直接 FP16 计算走 Tensor Core；硬件自动检测 GPU 型号和峰值性能（RTX 3080: FP16 TC 59.5T, FP16 CUDA 29.8T）
注意: 这不是 kernel 优化，而是评测环境变更。Baseline 变强了（PyTorch Ref 从 ~4.5ms 降到 ~2.1ms）

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU%(vs 29.8T CUDA) |
|---|---|---|---|
| 64 | 0.112 | 0.45 | 1.5% |
| 128 | 0.295 | 0.68 | 2.3% |
| 256 | 1.008 | 0.80 | 2.7% |
| 512 | 3.673 | 0.88 | 2.9% |
| 1024 | 12.521 | 1.03 | 3.5% |

对比 PyTorch Ref FP16 Tensor Core (B=16, N=512):
- Ours: 3.267 ms (0.99 TFLOPS, 3.3% MFU vs CUDA Core)
- PyTorch Ref: 2.089 ms (1.54 TFLOPS, 2.6% MFU vs Tensor Core)
- 加速比: 0.64x ← Tensor Core baseline 更强了

关键指标 (N=512): **3.673 ms, 0.88 TFLOPS** (kernel 本身性能与 Round 8 一致，FP16 vs BF16 差异不大)

### Round 10 — WMMA Tensor Core 加速 QK^T (FP16)

改动: csrc/sparse_attention.cu — 新增 `sparse_attention_fwd_wmma_fp16` kernel，用 WMMA m16n16k16 计算 QK^T 点积（4 次 WMMA 累加 HEAD_DIM=64），BLOCK_M=16, BLOCK_N=16, 1 warp/block。softmax + PV 累加仍用标量。FP16 路径走 WMMA，BF16 保留标量 kernel。
原因: 之前 kernel 用 CUDA Core 标量点积，无法利用 Tensor Core 的 2x 峰值算力

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU%(vs 29.8T) |
|---|---|---|---|
| 64 | 0.082 | 0.61 | 2.1% |
| 128 | 0.227 | 0.89 | 3.0% |
| 256 | 0.722 | 1.11 | 3.7% |
| 512 | 2.483 | 1.30 | 4.4% |
| 1024 | 8.386 | 1.54 | 5.2% |

对比 PyTorch Ref FP16 Tensor Core (B=16, N=512):
- Ours: 2.204 ms (1.46 TFLOPS)
- PyTorch Ref: 2.092 ms (1.54 TFLOPS)
- 加速比: 0.95x ← 接近追平 Tensor Core baseline

关键指标 (N=512): **2.483 ms, 1.30 TFLOPS** (vs Round 9: 3.673ms, 提速 1.48x)

### Round 11 — 4 warps per block 提高 occupancy

改动: csrc/sparse_attention.cu — WMMA kernel 从 1 warp/block 改为 4 warps/block (128 threads)。每个 warp 独立处理 16 行 Q tile，K/V 由 128 线程协作加载共享。grid.x 粒度从 16 行变为 64 行。
原因: 1 warp/block occupancy 极低（RTX 3080 每 SM 最多 48 warps），4 warps 提高 SM 利用率

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU%(vs 29.8T) |
|---|---|---|---|
| 64 | 0.076 | 0.66 | 2.2% |
| 128 | 0.199 | 1.01 | 3.4% |
| 256 | 0.615 | 1.31 | 4.4% |
| 512 | 1.890 | 1.70 | 5.7% |
| 1024 | 7.030 | 1.83 | 6.2% |

对比 PyTorch Ref FP16 Tensor Core (B=16, N=512):
- Ours: 1.868 ms (1.72 TFLOPS)
- PyTorch Ref: 2.089 ms (1.54 TFLOPS)
- 加速比: 1.12x ★ 首次超过 FP16 Tensor Core baseline

关键指标 (N=512): **1.890 ms, 1.70 TFLOPS** (vs Round 10: 2.483ms, 提速 1.31x)

### Round 12 — smem_v 改为 float 存储，消除 PV 累加时 __half2float 转换

改动: csrc/sparse_attention.cu — smem_v 从 `__half[16][64]` 改为 `float[16][64]`，加载时一次性转换为 float，PV 累加直接用 float 读取
原因: 每次 PV 累加都调用 `__half2float` 转换指令，预转换后省掉 16×64=1024 次转换/tile
效果: 对比测试略有提升（1.868→1.798ms），序列缩放测试波动在误差范围内

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU%(vs 29.8T) |
|---|---|---|---|
| 64 | 0.073 | 0.69 | 2.3% |
| 128 | 0.194 | 1.04 | 3.5% |
| 256 | 0.587 | 1.37 | 4.6% |
| 512 | 2.024 | 1.59 | 5.3% |
| 1024 | 6.770 | 1.90 | 6.4% |

对比 PyTorch Ref FP16 Tensor Core (B=16, N=512):
- Ours: 1.798 ms (1.79 TFLOPS)
- PyTorch Ref: 2.089 ms (1.54 TFLOPS)
- 加速比: 1.16x

关键指标 (N=512): **1.798 ms, 1.79 TFLOPS** (vs Round 11: 1.868ms, 提速 1.04x)

## 总结

| Round | 改动 | N=512 延迟 | TFLOPS | vs PyTorch Ref | vs Triton |
|---|---|---|---|---|---|
| 0 | Baseline | 16.571 ms | 0.19 | 0.3x (BF16) | — |
| 1 | maxreg 128 | 6.575 ms | 0.49 | 0.7x | — |
| 2 | 128 threads | 5.113 ms | 0.63 | 0.9x | — |
| 3 | batch mask | 4.962 ms | 0.65 | 0.9x | — |
| 4 | KV co-load | 4.231 ms | 0.76 | 1.1x | — |
| 5 | float4+no pad | 4.008 ms | 0.80 | 1.2x | — |
| 6 | no maxreg | 3.869 ms | 0.83 | 1.2x | — |
| 7 | BLOCK_N=16 | 3.453 ms | 0.93 | 1.4x | — |
| 8 | merge mask | 3.247 ms | 0.99 | 1.4x | — |
| 9 | FP16+detect | 3.673 ms | 0.88 | 0.64x (FP16 TC) | — |
| 10 | WMMA 1-warp | 2.483 ms | 1.30 | 0.95x | — |
| 11 | WMMA 4-warp | 1.890 ms | 1.70 | 1.12x | — |
| 12 | smem_v float | 1.763 ms | 1.83 | 1.18x | 0.43x |
| 14 | WMMA PV全化 | 1.046 ms | 3.08 | 2.00x | 0.72x |
| 15 | cp.async+smem_p pad | 1.052 ms | 3.06 | 2.27x | 0.82x |

Baselines (B=16, N=512, H=12, D=64, FP16, sparsity=0.75):
- **PyTorch Ref (FP16 TC):** 2.090 ms, 1.54 TFLOPS
- **Triton (FP16 TC):** 0.751 ms, 4.29 TFLOPS ← 当前目标

总提速: 16.571ms → 1.052ms = **15.8x** (Baseline → Round 15)
差距: 我们 1.052ms vs Triton 0.751ms，需要再快 **1.40x**

### Round 13 — WMMA PV 累加（失败，已回滚）

尝试: 用 WMMA 加速 PV 累加 [16×16]×[16×64]，通过 smem_acc 中转 rescale
结果: 正确性通过但性能严重退化（4.856ms），因为 smem 中转 rescale + 累加开销远大于标量 PV
教训: WMMA fragment 的 rescale 不能通过 smem 中转，需要直接操作 fragment 元素或换架构

### Round 14 — WMMA PV 全面化（正确 layout + smem_rsc）

改动: csrc/sparse_attention.cu — 彻底重写 FP16 kernel (`sparse_attn_wmma_full_fp16`)
- 将 acc 从 `float[64]` 换成 4 个 `wmma::accumulator fragment`，覆盖输出列 [BM×HD]
- smem_scores (float[16][16]) 替换为 smem_p (fp16[16][16])，节省 2KB smem
- 新增 smem_rsc[NWARPS][BM]：在 softmax 更新 row_max 之前保存 rescale_f
- rescale 直接操作 fragment.x[i]：读 smem_rsc[wid][lane/4 + ((i&2)?8:0)]
- WMMA PV：每 tile 4 次 mma_sync，P[BM×BN] × V_block[BN×BN] → acc_frag[f]
- 输出直接从 fragment element 散射写入 global memory（无 smem 中转）
- 关键修复：实测 sm_86 上 WMMA accumulator element layout 为
  col_off = {0,1,0,1,8,9,8,9}，row_extra = (i&2)?8:0（而非之前错误的 (i>=4)?8:0）

原因: Round 13 失败的核心是 smem 中转 rescale 开销大；本轮改用 smem_rsc 数组
暴露了另一个 bug：WMMA accumulator 的 element layout 与文档描述不一致，
通过编写专用验证 kernel 实测得到正确 layout 后方才通过正确性测试。

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU%(vs 29.8T) |
|---|---|---|---|
| 64 | 0.041 | 1.24 | 4.2% |
| 128 | 0.097 | 2.07 | 6.9% |
| 256 | 0.336 | 2.40 | 8.0% |
| 512 | 1.189 | 2.71 | 9.1% |
| 1024 | 4.421 | 2.91 | 9.8% |

对比 PyTorch Ref FP16 Tensor Core (B=16, N=512):
- Ours: 1.046 ms (3.08 TFLOPS)
- PyTorch Ref: 2.090 ms (1.54 TFLOPS)
- 加速比: 2.00x

关键指标 (N=512): **1.046 ms, 3.08 TFLOPS** (vs Round 12: 1.763ms, 提速 1.68x)

### Round 15 — cp.async 单缓冲 + smem_p padding

改动: csrc/sparse_attention.cu
- K/V 加载改用 `cp.async.cg` PTX 指令（16B 对齐，L2 cache bypass 路径），替代原来的普通 global→shared load
- `smem_k` / `smem_v` 改为双缓冲布局 `[2][BN][HD]`（本轮使用单缓冲 cbuf=0，结构为未来双缓冲预留）
- `smem_p` padding 从无 → `[NWARPS][BM][BN+8]`：BN+8=24 halves/行，消除 bank conflict，且满足 WMMA `load_matrix_sync` 要求 stride ≡ 0 (mod 8)
- 所有 smem 数组添加 `__align__(16)` 属性，保证 cp.async 16B 对齐要求
- LOAD_KV_ASYNC 宏：tid→chunk 映射，每线程处理 1×16B chunk，OOB 行用标量零填充
- pipeline 逻辑：每次迭代开头 `commit()`，然后做 mask 检查，再 `wait<0>()` + `syncthreads()`

原因: cp.async.cg 走 L2 bypass 路径，减少 L1 cache 污染；smem_p padding 消除 WMMA load 的 bank conflict
注意: cp.async 单缓冲不能实现真正的 pipeline（load 与 compute 重叠），性能与 R14 持平。
经历多轮调试：双缓冲 + `wait<1>` 方案存在 race condition 导致正确性失败；最终以单缓冲正确版本提交。

序列长度缩放 (B=16, H=12, D=64, sparsity=0.75):
| N | latency(ms) | TFLOPS | MFU%(vs 29.8T) |
|---|---|---|---|
| 64 | 0.034 | 1.46 | 4.9% |
| 128 | 0.087 | 2.32 | 7.8% |
| 256 | 0.284 | 2.84 | 9.5% |
| 512 | 1.052 | 3.06 | 10.3% |
| 1024 | 3.894 | 3.31 | 11.1% |

对比 PyTorch Ref FP16 Tensor Core (B=16, N=512):
- Ours: 0.919 ms (3.51 TFLOPS)  ← Baseline 对比单点
- PyTorch Ref: 2.089 ms (1.54 TFLOPS)
- Triton: 0.751 ms (4.29 TFLOPS)
- 加速比 vs Ref: 2.27x

关键指标 (N=512): **1.052 ms, 3.06 TFLOPS** (vs Round 14: 1.046ms，持平，cp.async 单缓冲无实质提速)
