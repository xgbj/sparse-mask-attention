"""
Sparse Mask Attention 工作流程图
参考 Flash Attention V2 论文风格 - 清晰的垂直数据流
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.patches import ConnectionPatch

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# 配色方案
C_HBM   = '#FFF4E6'  # 浅橙
C_SRAM  = '#E3F2FD'  # 浅蓝
C_COMP  = '#E8F5E9'  # 浅绿
C_COND  = '#FFF9C4'  # 浅黄
C_SKIP  = '#FFEBEE'  # 浅红
C_BLOCK = '#42A5F5'  # 蓝色块
C_MASK  = '#EF5350'  # 红色块
C_TEXT  = '#263238'  # 深灰文字

fig, ax = plt.subplots(figsize=(12, 16), facecolor='white')
ax.set_xlim(0, 12)
ax.set_ylim(0, 18)
ax.axis('off')

# 标题
ax.text(6, 17.5, 'Sparse Mask Attention — Workflow', 
        ha='center', fontsize=18, fontweight='bold', color=C_TEXT)
ax.text(6, 17.0, 'Block-level skipping + element-level score masking',
        ha='center', fontsize=11, style='italic', color='gray')

y = 16.2  # 起始 y 坐标

# ═══════════════════════════════════════════════════════
# ① HBM (主存)
# ═══════════════════════════════════════════════════════
box1 = FancyBboxPatch((1, y-1.2), 10, 1.2, boxstyle='round,pad=0.1',
                      facecolor=C_HBM, edgecolor='#E65100', linewidth=2.5)
ax.add_patch(box1)
ax.text(6, y-0.6, 'HBM  (High Bandwidth Memory)', 
        ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)

# HBM 内容
for i, (xc, label, color) in enumerate([(2.5, 'Q', C_BLOCK), (5, 'K', C_BLOCK), 
                                          (7.5, 'V', C_BLOCK), (9.8, 'Mask\n(uint32)', C_MASK)]):
    w = 1.2 if i < 3 else 1.4
    rect = Rectangle((xc-w/2, y-1.05), w, 0.7, facecolor=color, 
                      edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(xc, y-0.7, label, ha='center', va='center', 
            fontsize=10, color='white', fontweight='bold')

y -= 1.5

# 箭头 HBM → SRAM
for xc in [2.5, 5, 7.5, 9.8]:
    color = '#E65100' if xc < 9 else C_MASK
    ax.annotate('', xy=(xc, y-0.3), xytext=(xc, y+0.2),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=color))

ax.text(6, y-0.05, 'Load all Q, K, V blocks + bit-packed Mask',
        ha='center', fontsize=9, style='italic', color='gray')

y -= 0.8

# ═══════════════════════════════════════════════════════
# ② SRAM (片上缓存)
# ═══════════════════════════════════════════════════════
box2 = FancyBboxPatch((1, y-1.4), 10, 1.4, boxstyle='round,pad=0.1',
                      facecolor=C_SRAM, edgecolor='#1565C0', linewidth=2.5)
ax.add_patch(box2)
ax.text(6, y-0.25, 'SRAM  (On-chip Cache)', 
        ha='center', va='top', fontsize=12, fontweight='bold', color=C_TEXT)

# SRAM 内容
for xc, label, color in [(2.2, 'Qi', C_BLOCK), (4.5, 'Kj', C_BLOCK), 
                          (6.8, 'Vj', C_BLOCK), (9.3, 'smem_mask', '#F48FB1')]:
    w = 1.1 if xc < 8 else 1.8
    rect = Rectangle((xc-w/2, y-1.15), w, 0.65, facecolor=color, 
                      edgecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(xc, y-0.82, label, ha='center', va='center',
            fontsize=9, color='white', fontweight='bold')

y -= 1.7

# 箭头
ax.annotate('', xy=(6, y-0.2), xytext=(6, y+0.15),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#1565C0'))

y -= 0.5

# ═══════════════════════════════════════════════════════
# ③ Block-level Check (__ballot_sync)
# ═══════════════════════════════════════════════════════
box3 = FancyBboxPatch((1.5, y-1.3), 9, 1.3, boxstyle='round,pad=0.08',
                      facecolor=C_COND, edgecolor='#F57F17', linewidth=2.5)
ax.add_patch(box3)

ax.text(6, y-0.35, '__ballot_sync()', ha='center', fontsize=11, 
        fontweight='bold', color='#E65100')
ax.text(6, y-0.65, 'Any unmasked bit in BN-block?',
        ha='center', fontsize=10, style='italic', color=C_TEXT)
ax.text(6, y-0.95, '(warp-level vote on smem_mask)',
        ha='center', fontsize=8, color='gray')

y -= 1.5

# 分支箭头 - YES (左) / NO (右)
# YES 分支
ax.annotate('', xy=(4, y-0.2), xytext=(4.5, y+0.2),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#2E7D32'))
ax.text(3.2, y+0.05, 'YES', fontsize=10, fontweight='bold', 
        color='#2E7D32', ha='right')
ax.text(3.2, y-0.2, '(has active)', fontsize=8, color='#2E7D32', ha='right')

# NO 分支
ax.annotate('', xy=(8.5, y-0.2), xytext=(7.5, y+0.2),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#C62828'))
ax.text(9, y+0.05, 'NO', fontsize=10, fontweight='bold',
        color='#C62828', ha='left')
ax.text(9, y-0.2, '(all zero)', fontsize=8, color='#C62828', ha='left')

y -= 0.6

# NO → Skip box (右侧)
skip_box = FancyBboxPatch((7, y-0.8), 3.5, 0.8, boxstyle='round,pad=0.05',
                          facecolor=C_SKIP, edgecolor='#C62828', linewidth=2)
ax.add_patch(skip_box)
ax.text(8.75, y-0.4, 'Skip entire block', ha='center', va='center',
        fontsize=10, fontweight='bold', color='#C62828')
ax.text(8.75, y-0.65, '(continue loop)', ha='center', va='center',
        fontsize=8, color='gray')

y -= 1.1

# ═══════════════════════════════════════════════════════
# ④ Compute QK^T (WMMA)
# ═══════════════════════════════════════════════════════
box4 = FancyBboxPatch((1.5, y-1.3), 5, 1.3, boxstyle='round,pad=0.08',
                      facecolor=C_COMP, edgecolor='#388E3C', linewidth=2.5)
ax.add_patch(box4)
ax.text(4, y-0.35, r'Compute  $S = QK^T$', ha='center', fontsize=11,
        fontweight='bold', color='#1B5E20', style='italic')
ax.text(4, y-0.65, '(WMMA tensor core)', ha='center', fontsize=9, color=C_TEXT)
ax.text(4, y-0.95, 'Full BN×BM block computed', ha='center', fontsize=8, color='gray')

y -= 1.5

# 箭头
ax.annotate('', xy=(4, y-0.2), xytext=(4, y+0.15),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#388E3C'))

y -= 0.5

# ═══════════════════════════════════════════════════════
# ⑤ Score Masking (element-wise)
# ═══════════════════════════════════════════════════════
box5 = FancyBboxPatch((1.5, y-1.5), 5, 1.5, boxstyle='round,pad=0.08',
                      facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2.5)
ax.add_patch(box5)
ax.text(4, y-0.3, 'Apply Mask to Scores', ha='center', fontsize=11,
        fontweight='bold', color='#4A148C')
ax.text(4, y-0.6, r'For each element $(i,j)$:', ha='center', fontsize=9, 
        color=C_TEXT, style='italic')

# 两种情况
ax.text(4, y-0.9, r'smem_mask bit = 0  $\Rightarrow$  $S_{ij} = -65504$  (−∞)', 
        ha='center', fontsize=9, color='#C62828')
ax.text(4, y-1.15, r'smem_mask bit = 1  $\Rightarrow$  $S_{ij}$ keeps', 
        ha='center', fontsize=9, color='#2E7D32')
ax.text(4, y-1.4, '(read bit-packed mask from smem)', ha='center', 
        fontsize=8, color='gray')

y -= 1.7

# 箭头
ax.annotate('', xy=(4, y-0.2), xytext=(4, y+0.15),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#7B1FA2'))

y -= 0.5

# ═══════════════════════════════════════════════════════
# ⑥ Online Softmax
# ═══════════════════════════════════════════════════════
box6 = FancyBboxPatch((1.5, y-1.2), 5, 1.2, boxstyle='round,pad=0.08',
                      facecolor=C_COMP, edgecolor='#00897B', linewidth=2.5)
ax.add_patch(box6)
ax.text(4, y-0.35, r'Online Softmax', ha='center', fontsize=11,
        fontweight='bold', color='#004D40', style='italic')
ax.text(4, y-0.65, r'$P_{ij} = \frac{\exp(S_{ij} - m)}{\sum \exp(...)}$', 
        ha='center', fontsize=10, color=C_TEXT)
ax.text(4, y-0.95, r'Masked scores ($S=-\infty$) → weight ≈ 0', 
        ha='center', fontsize=8, color='gray')

y -= 1.4

# 箭头
ax.annotate('', xy=(4, y-0.2), xytext=(4, y+0.15),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#00897B'))

y -= 0.5

# ═══════════════════════════════════════════════════════
# ⑦ Compute PV (WMMA)
# ═══════════════════════════════════════════════════════
box7 = FancyBboxPatch((1.5, y-1.2), 5, 1.2, boxstyle='round,pad=0.08',
                      facecolor=C_COMP, edgecolor='#388E3C', linewidth=2.5)
ax.add_patch(box7)
ax.text(4, y-0.35, r'Accumulate  $O = PV$', ha='center', fontsize=11,
        fontweight='bold', color='#1B5E20', style='italic')
ax.text(4, y-0.65, '(WMMA tensor core)', ha='center', fontsize=9, color=C_TEXT)
ax.text(4, y-0.95, 'Only unmasked tokens contribute', ha='center', 
        fontsize=8, color='gray')

y -= 1.4

# 箭头
ax.annotate('', xy=(4, y-0.2), xytext=(4, y+0.15),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#388E3C'))

y -= 0.5

# ═══════════════════════════════════════════════════════
# ⑧ Loop & Normalize
# ═══════════════════════════════════════════════════════
box8 = FancyBboxPatch((1.5, y-1.0), 5, 1.0, boxstyle='round,pad=0.08',
                      facecolor='#E0F7FA', edgecolor='#0097A7', linewidth=2.5)
ax.add_patch(box8)
ax.text(4, y-0.3, 'Loop over all BN blocks', ha='center', fontsize=10,
        fontweight='bold', color='#006064')
ax.text(4, y-0.6, 'Update max/sum for each row', ha='center', fontsize=8, color=C_TEXT)
ax.text(4, y-0.85, '(online softmax rescaling)', ha='center', fontsize=8, color='gray')

# 反馈箭头（回到 ③）
ax.annotate('', xy=(7, y+10.5), xytext=(6.8, y-0.5),
            arrowprops=dict(arrowstyle='->', lw=1.8, color='#0097A7',
                            linestyle='--', connectionstyle='arc3,rad=0.3'))
ax.text(7.5, y+5, 'Next\nblock', ha='center', fontsize=8, 
        color='#0097A7', style='italic')

y -= 1.3

# 箭头
ax.annotate('', xy=(4, y-0.2), xytext=(4, y+0.15),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#0097A7'))

y -= 0.5

# ═══════════════════════════════════════════════════════
# ⑨ Final Normalize & Write
# ═══════════════════════════════════════════════════════
box9 = FancyBboxPatch((1.5, y-1.1), 5, 1.1, boxstyle='round,pad=0.08',
                      facecolor=C_HBM, edgecolor='#E65100', linewidth=2.5)
ax.add_patch(box9)
ax.text(4, y-0.3, r'Normalize  $O_{final} = O / \sum$', ha='center', fontsize=11,
        fontweight='bold', color='#BF360C', style='italic')
ax.text(4, y-0.6, 'Write output to HBM', ha='center', fontsize=9, color=C_TEXT)
ax.text(4, y-0.9, r'Save LSE for backward: $m + \log(\sum)$', ha='center',
        fontsize=8, color='gray')

# ═══════════════════════════════════════════════════════
# 右侧注释：Attention Matrix 示意
# ═══════════════════════════════════════════════════════
anno_x = 8.2
anno_y = 10

ax.text(9.3, anno_y+0.5, 'Attention Matrix', ha='center', fontsize=10,
        fontweight='bold', color=C_TEXT)
ax.text(9.3, anno_y+0.15, '(example pattern)', ha='center', fontsize=8, color='gray')

# 绘制小型 attention 矩阵示例
mask_pat = [
    [1, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
]

bw, bh = 0.22, 0.22
ox, oy = 7.8, anno_y - 1.5
for i in range(6):
    for j in range(6):
        fc = C_BLOCK if mask_pat[i][j] else '#FFCDD2'
        rect = Rectangle((ox + j*bw, oy + (5-i)*bh), bw, bh,
                          facecolor=fc, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        if not mask_pat[i][j]:
            ax.text(ox + j*bw + bw/2, oy + (5-i)*bh + bh/2, '×',
                    ha='center', va='center', fontsize=7, color='#C62828')

# 图例
legend_y = anno_y - 2.5
ax.add_patch(Rectangle((7.8, legend_y), 0.3, 0.2, facecolor=C_BLOCK, edgecolor='gray'))
ax.text(8.3, legend_y+0.1, 'Active (compute)', va='center', fontsize=7)

ax.add_patch(Rectangle((7.8, legend_y-0.35), 0.3, 0.2, facecolor='#FFCDD2', edgecolor='gray'))
ax.text(8.3, legend_y-0.25, 'Masked (skip)', va='center', fontsize=7)

# 底部说明框
note_box = FancyBboxPatch((0.5, 0.3), 11, 1.0, boxstyle='round,pad=0.08',
                          facecolor='#FAFAFA', edgecolor='#BDBDBD', linewidth=1.5)
ax.add_patch(note_box)
ax.text(6, 0.95, 'Key Features', ha='center', fontsize=10, fontweight='bold', color=C_TEXT)
ax.text(6, 0.65, '• Block-level skip: __ballot_sync() checks entire BN-block (~75% blocks skipped)',
        ha='center', fontsize=8, color=C_TEXT)
ax.text(6, 0.45, '• Element-level masking: masked positions set to -∞ before softmax',
        ha='center', fontsize=8, color=C_TEXT)

plt.tight_layout()
plt.savefig('/home/wangjiawei/sparse-mask-attention/notes/flash_attention_comparison.png',
            dpi=250, bbox_inches='tight', facecolor='white')
print('✓ Workflow diagram saved: notes/flash_attention_comparison.png')
