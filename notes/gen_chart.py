"""生成性能优化图表 perf_chart.png"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Round 数据 (N=512 延迟 ms, TFLOPS 已修正为 dense FLOPs)
rounds = [
    (0,  "R0 Baseline",     16.571, 0.76),
    (1,  "R1 maxreg128",     6.575, 1.96),
    (2,  "R2 128 threads",   5.113, 2.52),
    (3,  "R3 batch mask",    4.962, 2.60),
    (4,  "R4 KV co-load",    4.231, 3.04),
    (5,  "R5 float4",        4.008, 3.20),
    (6,  "R6 no maxreg",     3.869, 3.32),
    (7,  "R7 BN=16",         3.453, 3.72),
    (8,  "R8 merge mask",    3.247, 3.96),
    (9,  "R9 FP16",          3.673, 3.52),
    (10, "R10 WMMA 1w",      2.483, 5.20),
    (11, "R11 WMMA 4w",      1.890, 6.80),
    (12, "R12 smem_v f32",   1.763, 7.32),
    (14, "R14 WMMA PV",      1.046, 12.32),
    (15, "R15 cp.async",     1.052, 12.24),
    (18, "R18 BN=64",        0.938, 13.72),
    (19, "R19 smem pad",     0.622, 20.73),
    (20, "R20 mask→smem",    0.595, 21.64),
    (21, "R21 2-lane sfmx",  0.583, 22.12),
    (22, "R22 merge sc/pv",  0.571, 22.58),
    (23, "R23 reg softmax",  0.512, 25.17),
]

r_ids   = [r[0]  for r in rounds]
labels  = [r[1]  for r in rounds]
latency = [r[2]  for r in rounds]
tflops  = [r[3]  for r in rounds]

# Baselines (B=16, N=512, H=12, D=64, FP16, sparsity=0.75)
TRITON_LAT   = 0.751
TRITON_TFLOP = 17.15
PT_REF_LAT   = 2.091
PT_REF_TFLOP = 6.16
CUDNN_LAT    = 0.892
CUDNN_TFLOP  = 14.45
FI_LAT       = 1.082
FI_TFLOP     = 11.91
FA_LAT       = 0.403
FA_TFLOP     = 31.96

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Sparse Mask Attention — Optimization Progress (B=16, N=512, H=12, D=64, FP16, sparsity=0.75)',
             fontsize=11, fontweight='bold')

# ── 左图：延迟曲线 ──────────────────────────────────────────
ax = axes[0]
x = np.arange(len(rounds))

# 按阶段着色
colors = []
for rnd in r_ids:
    if rnd <= 8:
        colors.append('#4C72B0')   # 蓝：标量优化阶段
    elif rnd <= 12:
        colors.append('#DD8452')   # 橙：WMMA QK 阶段
    else:
        colors.append('#55A868')   # 绿：WMMA PV 阶段

bars = ax.bar(x, latency, color=colors, edgecolor='white', linewidth=0.8, zorder=3)

# Baseline 参考线
ax.axhline(FA_LAT,      color='#E377C2', linestyle='-.',  linewidth=1.5, zorder=4, label=f'flash-attn (dense) {FA_LAT:.3f} ms')
ax.axhline(TRITON_LAT,  color='red',     linestyle='--',  linewidth=1.5, zorder=4, label=f'Triton {TRITON_LAT:.3f} ms')
ax.axhline(CUDNN_LAT,   color='#2CA02C', linestyle='--',  linewidth=1.2, zorder=4, label=f'cuDNN SDPA {CUDNN_LAT:.3f} ms')
ax.axhline(FI_LAT,      color='#FF7F0E', linestyle=':',   linewidth=1.2, zorder=4, label=f'FlashInfer {FI_LAT:.3f} ms')
ax.axhline(PT_REF_LAT,  color='purple',  linestyle=':',   linewidth=1.5, zorder=4, label=f'PyTorch Ref {PT_REF_LAT:.3f} ms')

# 标注每根柱
for bar, val in zip(bars, latency):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
            f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Latency (ms)', fontsize=10)
ax.set_title('Kernel Latency (N=512, lower is better)', fontsize=10)
ax.set_ylim(0, max(latency) * 1.25)
ax.grid(axis='y', alpha=0.4, zorder=1)

# 图例 + 阶段说明
leg_patches = [
    mpatches.Patch(color='#4C72B0', label='Scalar optimization (R0-R8)'),
    mpatches.Patch(color='#DD8452', label='WMMA QK^T (R9-R12)'),
    mpatches.Patch(color='#55A868', label='WMMA QK+PV (R14-R23)'),
]
ax.legend(handles=leg_patches + ax.get_legend_handles_labels()[0][::-1],
          fontsize=7, loc='upper right')

# ── 右图：TFLOPS 曲线 ─────────────────────────────────────
ax2 = axes[1]
ax2.plot(x, tflops, 'o-', color='#4C72B0', linewidth=2, markersize=6, zorder=3, label='Our kernel')

# Baseline 参考线
ax2.axhline(FA_TFLOP,      color='#E377C2', linestyle='-.',  linewidth=1.5, zorder=4, label=f'flash-attn (dense) {FA_TFLOP:.1f} TFLOPS')
ax2.axhline(TRITON_TFLOP,  color='red',     linestyle='--',  linewidth=1.5, zorder=4, label=f'Triton {TRITON_TFLOP:.2f} TFLOPS')
ax2.axhline(CUDNN_TFLOP,   color='#2CA02C', linestyle='--',  linewidth=1.2, zorder=4, label=f'cuDNN SDPA {CUDNN_TFLOP:.2f} TFLOPS')
ax2.axhline(FI_TFLOP,      color='#FF7F0E', linestyle=':',   linewidth=1.2, zorder=4, label=f'FlashInfer {FI_TFLOP:.2f} TFLOPS')
ax2.axhline(PT_REF_TFLOP,  color='purple',  linestyle=':',   linewidth=1.5, zorder=4, label=f'PyTorch Ref {PT_REF_TFLOP:.2f} TFLOPS')

# 标注数据点
for xi, val in zip(x, tflops):
    ax2.annotate(f'{val:.2f}', (xi, val), textcoords='offset points',
                 xytext=(0, 6), ha='center', fontsize=7)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('TFLOPS', fontsize=10)
ax2.set_title('Effective TFLOPS (N=512, higher is better)', fontsize=10)
ax2.set_ylim(0, FA_TFLOP * 1.2)
ax2.grid(axis='y', alpha=0.4)
ax2.legend(fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig('/home/wangjiawei/sparse-mask-attention/notes/perf_chart.png', dpi=150, bbox_inches='tight')
print("图表已保存到 notes/perf_chart.png")
