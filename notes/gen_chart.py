"""生成性能优化图表 perf_chart.png"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Round 数据 (N=512 延迟 ms)
rounds = [
    (0,  "R0 Baseline",     16.571, 0.19),
    (1,  "R1 maxreg128",     6.575, 0.49),
    (2,  "R2 128 threads",   5.113, 0.63),
    (3,  "R3 batch mask",    4.962, 0.65),
    (4,  "R4 KV co-load",    4.231, 0.76),
    (5,  "R5 float4",        4.008, 0.80),
    (6,  "R6 no maxreg",     3.869, 0.83),
    (7,  "R7 BN=16",         3.453, 0.93),
    (8,  "R8 merge mask",    3.247, 0.99),
    (9,  "R9 FP16",          3.673, 0.88),
    (10, "R10 WMMA 1w",      2.483, 1.30),
    (11, "R11 WMMA 4w",      1.890, 1.70),
    (12, "R12 smem_v f32",   1.763, 1.83),
    (14, "R14 WMMA PV",      1.046, 3.08),
    (15, "R15 cp.async",     1.052, 3.06),
]

r_ids   = [r[0]  for r in rounds]
labels  = [r[1]  for r in rounds]
latency = [r[2]  for r in rounds]
tflops  = [r[3]  for r in rounds]

TRITON_LAT   = 0.751
TRITON_TFLOP = 4.29
PT_REF_LAT   = 2.090
PT_REF_TFLOP = 1.54

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
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
ax.axhline(TRITON_LAT,  color='red',    linestyle='--', linewidth=1.5, zorder=4, label=f'Triton {TRITON_LAT:.3f} ms')
ax.axhline(PT_REF_LAT,  color='purple', linestyle=':',  linewidth=1.5, zorder=4, label=f'PyTorch Ref {PT_REF_LAT:.3f} ms')

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
    mpatches.Patch(color='#4C72B0', label='Scalar optimization (R0–R8)'),
    mpatches.Patch(color='#DD8452', label='WMMA QK^T (R9–R12)'),
    mpatches.Patch(color='#55A868', label='WMMA QK+PV (R14–R15)'),
]
ax.legend(handles=leg_patches + ax.get_legend_handles_labels()[0][::-1],
          fontsize=8, loc='upper right')

# ── 右图：TFLOPS 曲线 ─────────────────────────────────────
ax2 = axes[1]
ax2.plot(x, tflops, 'o-', color='#4C72B0', linewidth=2, markersize=6, zorder=3, label='Our kernel')
ax2.axhline(TRITON_TFLOP,  color='red',    linestyle='--', linewidth=1.5, zorder=4, label=f'Triton {TRITON_TFLOP:.2f} TFLOPS')
ax2.axhline(PT_REF_TFLOP,  color='purple', linestyle=':',  linewidth=1.5, zorder=4, label=f'PyTorch Ref {PT_REF_TFLOP:.2f} TFLOPS')

# 标注数据点
for xi, val in zip(x, tflops):
    ax2.annotate(f'{val:.2f}', (xi, val), textcoords='offset points',
                 xytext=(0, 6), ha='center', fontsize=7)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('TFLOPS', fontsize=10)
ax2.set_title('Effective TFLOPS (N=512, higher is better)', fontsize=10)
ax2.set_ylim(0, TRITON_TFLOP * 1.25)
ax2.grid(axis='y', alpha=0.4)
ax2.legend(fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('/home/wangjiawei/sparse-mask-attention/notes/perf_chart.png', dpi=150, bbox_inches='tight')
print("图表已保存到 notes/perf_chart.png")
