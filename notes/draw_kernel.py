"""
Cartoon-style diagram of the Sparse Mask Attention CUDA kernel.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

fig = plt.figure(figsize=(22, 14), facecolor='#F0F4FF')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')

# ── palette ────────────────────────────────────────────────────
C_GLOBAL  = '#2979FF'
C_SMEM    = '#FF8F00'
C_WMMA    = '#2E7D32'
C_SCALAR  = '#6A1B9A'
C_SKIP    = '#C62828'
C_OUT     = '#00695C'
C_BAND_G  = '#E3EEFF'
C_BAND_S  = '#FFF3E0'
C_BAND_C  = '#F3E5F5'
C_BAND_O  = '#E8F5E9'

# ── helpers ────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, color, title, body=None,
         tfont=11, bfont=8, radius=0.22, lw=2.5, alpha=1.0):
    shadow = FancyBboxPatch((x+0.07, y-0.07), w, h,
                            boxstyle=f"round,pad=0.05,rounding_size={radius}",
                            facecolor='#AAAAAA', edgecolor='none',
                            linewidth=0, alpha=0.25, zorder=2)
    ax.add_patch(shadow)
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.05,rounding_size={radius}",
                         facecolor=color, edgecolor='white',
                         linewidth=lw, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ty = y + h/2 + (0.15 if body else 0)
    ax.text(x+w/2, ty, title, ha='center', va='center',
            fontsize=tfont, fontweight='bold', color='white', zorder=4,
            path_effects=[pe.withStroke(linewidth=2.5, foreground='#00000066')])
    if body:
        ax.text(x+w/2, y+h/2-0.22, body, ha='center', va='center',
                fontsize=bfont, color='#FFFFFFDD', zorder=4,
                linespacing=1.4)

def arr(ax, x1, y1, x2, y2, color='#444', lw=2.0, rad=0.0, label=None, lside='right'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dx = 0.12 if lside == 'right' else -0.12
        ax.text(mx+dx, my, label, fontsize=7.5, color=color,
                ha='left' if lside=='right' else 'right', va='center', zorder=6,
                fontweight='bold')

def band(ax, yb, yt, color, label):
    r = FancyBboxPatch((0.35, yb), 21.3, yt-yb,
                       boxstyle="round,pad=0.08,rounding_size=0.15",
                       facecolor=color, edgecolor='#CCCCCC',
                       linewidth=1, alpha=0.55, zorder=1)
    ax.add_patch(r)
    ax.text(0.58, (yb+yt)/2, label, fontsize=7.5, color='#888',
            va='center', rotation=90, zorder=2, fontstyle='italic')

def infobox(ax, x, y, w, h, fc, ec, title, body, tfont=9, bfont=7.5):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.1,rounding_size=0.2",
                       facecolor=fc, edgecolor=ec,
                       linewidth=2, alpha=0.95, zorder=3)
    ax.add_patch(b)
    ax.text(x+w/2, y+h-0.28, title, ha='center', va='center',
            fontsize=tfont, fontweight='bold', color=ec, zorder=4)
    ax.text(x+w/2, y+h/2-0.1, body, ha='center', va='center',
            fontsize=bfont, color='#333', zorder=4, linespacing=1.5)

# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
ax.text(11, 13.55, 'Sparse Mask Attention — CUDA Kernel Architecture',
        ha='center', va='center', fontsize=17, fontweight='bold',
        color='#1A1A2E', zorder=6)
ax.text(11, 13.1,
        'RTX 3080 (sm_86)  |  FP16  |  WMMA m16n16k16  |  BM=16, BN=64, HD=64  |  4 warps / block  |  128 threads / block',
        ha='center', va='center', fontsize=9, color='#445566', zorder=6)

# ══════════════════════════════════════════════════════════════
# BACKGROUND BANDS
# ══════════════════════════════════════════════════════════════
band(ax, 11.3, 12.8, C_BAND_G,  'GLOBAL MEMORY (HBM)')
band(ax,  8.0, 11.0, C_BAND_S,  'SHARED MEMORY (SRAM)')
band(ax,  1.8,  7.7, C_BAND_C,  'COMPUTE (per warp)')
band(ax,  0.2,  1.5, C_BAND_O,  'OUTPUT')

# ══════════════════════════════════════════════════════════════
# GLOBAL MEMORY ROW
# ══════════════════════════════════════════════════════════════
for lbl, sub, xp in [
    ('Q',  '[B,H,N,64] FP16', 1.2),
    ('K',  '[B,H,N,64] FP16', 4.8),
    ('V',  '[B,H,N,64] FP16', 8.4),
]:
    rbox(ax, xp, 11.55, 2.6, 0.9, C_GLOBAL, lbl, sub, tfont=14, bfont=8.5, radius=0.2)

# bool mask → pack → uint32
rbox(ax, 12.0, 11.55, 2.8, 0.9, '#777', 'bool mask', '[B,H,N,N]', tfont=10, bfont=8.5, radius=0.2)
ax.annotate('', xy=(15.5, 12.0), xytext=(14.8, 12.0),
            arrowprops=dict(arrowstyle='->', color=C_SKIP, lw=2), zorder=5)
ax.text(15.15, 12.35, 'pack_mask\nkernel', ha='center', fontsize=7.5,
        color=C_SKIP, zorder=6, fontweight='bold')
rbox(ax, 15.5, 11.55, 5.1, 0.9, C_SKIP, 'uint32 mask (bit-packed)',
     '[B,H,N,N/32]  —  8x memory saving', tfont=10, bfont=8.5, radius=0.2)

# ══════════════════════════════════════════════════════════════
# SHARED MEMORY ROW
# ══════════════════════════════════════════════════════════════
smem_items = [
    ('smem_q',       '[4][16][72] fp16\n9 KB  pad+8',  C_SMEM,   1.0),
    ('smem_k',       '[64][72] fp16\n9 KB  pad+8',     C_SMEM,   4.3),
    ('smem_v',       '[64][72] fp16\n9 KB  pad+8',     C_SMEM,   7.6),
    ('smem_p',       '[4][16][72] fp16\n9 KB  scores', C_SMEM,  10.9),
    ('smem_mask',    '[64][2] uint32\n0.5 KB',         C_SKIP,  14.2),
    ('smem_max/sum\n/rsc', '[4][16] float\nonline state', C_SCALAR, 17.5),
]
for lbl, sub, col, xp in smem_items:
    rbox(ax, xp, 8.3, 3.0, 1.0, col, lbl, sub, tfont=9, bfont=8, radius=0.2)

# global → smem arrows
arr(ax, 2.5,  11.55, 2.5,  9.3,  color=C_GLOBAL, lw=2)          # Q
arr(ax, 6.1,  11.55, 5.8,  9.3,  color=C_GLOBAL, lw=2)          # K
arr(ax, 9.7,  11.55, 9.1,  9.3,  color=C_GLOBAL, lw=2)          # V
ax.text(5.8, 9.75, 'cp.async.cg', ha='center', fontsize=7.5, color=C_GLOBAL, zorder=6, fontweight='bold')
ax.text(9.1, 9.75, 'cp.async.cg', ha='center', fontsize=7.5, color=C_GLOBAL, zorder=6, fontweight='bold')
arr(ax, 18.05, 11.55, 15.7, 9.3, color=C_SKIP, lw=2)            # mask

# ══════════════════════════════════════════════════════════════
# SPARSE SKIP BOX
# ══════════════════════════════════════════════════════════════
infobox(ax, 14.0, 6.0, 7.5, 1.5,
        '#FFEBEE', C_SKIP,
        'SPARSE BLOCK SKIP',
        '__ballot_sync() across 4 warps\n'
        'if ALL 64 rows fully masked  ->  skip tile (continue)\n'
        '75% sparsity: ~6 of 8 tiles skipped  ->  6x less compute',
        tfont=10, bfont=8)
arr(ax, 15.7, 8.3, 15.7, 7.5, color=C_SKIP, lw=2)   # smem_mask → skip
# skip → loop back
ax.annotate('', xy=(21.7, 7.0), xytext=(21.5, 7.0),
            arrowprops=dict(arrowstyle='->', color=C_SKIP, lw=2), zorder=5)
ax.text(21.75, 7.0, 'SKIP\n(continue)', ha='left', va='center',
        fontsize=8, color=C_SKIP, fontweight='bold', zorder=6)

# ══════════════════════════════════════════════════════════════
# COMPUTE STEPS  (2 columns × 3 rows)
# ══════════════════════════════════════════════════════════════

# ── col A: left ──────────────────────────────────────────────
rbox(ax, 1.0, 5.5, 5.8, 1.9, C_WMMA,
     '(1)  WMMA  QK^T',
     'Q[16x64] x K^T[64x64]  ->  scores[16x64]\n'
     '4 warps, each: 4 mma_sync (m16n16k16)\n'
     'smem_q[wid] x smem_k  ->  qk_sub[4 frags]',
     tfont=11, bfont=8.5)

rbox(ax, 1.0, 3.3, 5.8, 1.9, C_SKIP,
     '(2)  Apply Mask  ->  smem_p',
     'per fragment element: check smem_mask bit\n'
     'masked  ->  score = -65504  (exp underflows to 0)\n'
     'write fp16 scores to smem_p[wid][16][64]',
     tfont=11, bfont=8.5)

# ── col B: middle ────────────────────────────────────────────
rbox(ax, 7.5, 5.5, 5.8, 1.9, C_SCALAR,
     '(3)  Online Softmax',
     '2-lane parallel: 32 lanes, 2 per row\n'
     'each lane: 32 scores -> fmax -> expf -> sum\n'
     '__shfl_xor_sync(mask, val, 16) merge halves\n'
     'update smem_max / smem_sum / smem_rsc',
     tfont=11, bfont=8.5)

rbox(ax, 7.5, 3.3, 5.8, 1.9, C_SCALAR,
     '(4)  Rescale  acc_frag',
     'acc_frag[f].x[i]  *=  smem_rsc[wid][row]\n'
     'online softmax correction for new max\n'
     '4 warps x 4 frags x 8 elements each',
     tfont=11, bfont=8.5)

# ── col C: right ─────────────────────────────────────────────
rbox(ax, 14.0, 5.5, 7.5, 1.9, C_WMMA,
     '(5)  WMMA  PV',
     'P[16x64] x V[64x64]  ->  acc_frag[16x64]\n'
     '4 warps, each: 4x4=16 mma_sync (m16n16k16)\n'
     'smem_p[wid] x smem_v  ->  acc_frag[4 frags]',
     tfont=11, bfont=8.5)

rbox(ax, 14.0, 3.3, 7.5, 1.9, C_OUT,
     '(6)  Normalize  +  Write Out',
     'acc_frag[f].x[i]  /=  smem_sum[wid][row]\n'
     '__float2half  ->  scatter write to Out[gr][gc]\n'
     'each thread writes 8 output elements',
     tfont=11, bfont=8.5)

# ── compute arrows ───────────────────────────────────────────
# smem_q/k → (1)
arr(ax, 2.5,  8.3, 2.5,  7.4, color=C_SMEM, lw=2)
arr(ax, 5.8,  8.3, 4.5,  7.4, color=C_SMEM, lw=2)
# (1) → (2)
arr(ax, 3.9,  5.5, 3.9,  5.2, color=C_WMMA, lw=2)
# smem_mask → (2)
arr(ax, 15.7, 8.3, 5.5,  5.0, color=C_SKIP, lw=1.8, rad=-0.15)
# (2) → smem_p label
ax.text(3.9, 3.1, 'smem_p', ha='center', fontsize=8.5,
        color=C_SMEM, fontweight='bold', zorder=6)
arr(ax, 3.9, 3.3, 3.9, 3.1, color=C_SMEM, lw=1.5)

# smem_p → (3) and (5)
arr(ax, 6.8, 4.2, 7.5, 6.4, color=C_SMEM, lw=2)
arr(ax, 6.8, 4.2, 14.0, 6.4, color=C_SMEM, lw=2, rad=-0.1)

# smem_v → (5)
arr(ax, 9.1, 8.3, 16.5, 7.4, color=C_SMEM, lw=2)

# (3) → smem_max/sum/rsc
arr(ax, 10.4, 7.4, 19.0, 8.3, color=C_SCALAR, lw=1.8, rad=-0.2)

# smem_rsc → (4)
arr(ax, 19.0, 8.3, 10.4, 5.2, color=C_SCALAR, lw=1.8, rad=0.2)

# (3) → (4)  (softmax state flows down)
arr(ax, 10.4, 5.5, 10.4, 5.2, color=C_SCALAR, lw=2)

# (4) → (5)  acc_frag
arr(ax, 13.3, 4.25, 14.0, 4.25, color=C_WMMA, lw=2.5, label='acc_frag')

# (5) → (6)
arr(ax, 17.75, 5.5, 17.75, 5.2, color=C_WMMA, lw=2)

# ── loop arrow ───────────────────────────────────────────────
ax.annotate('', xy=(0.75, 8.3), xytext=(0.75, 3.3),
            arrowprops=dict(arrowstyle='<-', color='#999', lw=2.5,
                            connectionstyle='arc3,rad=0.0'), zorder=5)
ax.text(0.62, 5.8, 'for tn in\nrange(8)\n[N=512]',
        ha='right', va='center', fontsize=8, color='#666',
        rotation=90, zorder=6)

# ══════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════
rbox(ax, 7.0, 0.3, 8.0, 0.9, C_OUT,
     'Out  [B, H, N, 64]  FP16', None, tfont=13, radius=0.2)
arr(ax, 17.75, 3.3, 11.0, 1.2, color=C_OUT, lw=2.5)

# ══════════════════════════════════════════════════════════════
# INFO BOXES (bottom row)
# ══════════════════════════════════════════════════════════════
# smem budget
infobox(ax, 1.0, 1.8, 5.5, 1.3,
        '#FFF8E1', C_SMEM,
        'Shared Memory Budget',
        'smem_q 9KB + smem_k 9KB + smem_v 9KB\n'
        'smem_p 9KB + state 1KB + mask 0.5KB\n'
        '= 37 KB / block  ->  2 blocks / SM  (100 KB total)',
        tfont=9, bfont=8)

# bank conflict fix
infobox(ax, 7.2, 1.8, 5.5, 1.3,
        '#E8F5E9', C_WMMA,
        'Bank Conflict Fix  (R19  +1.51x)',
        'stride 64 -> 72 halves  (pad +8 cols)\n'
        '144 bytes / 4 = 36 slots  ->  36 mod 32 = 4\n'
        '16-way -> 2-way conflict  |  50M -> 858K  (-98%)',
        tfont=9, bfont=8)

# performance
infobox(ax, 13.4, 1.8, 7.5, 1.3,
        '#E8EAF6', '#283593',
        'Final Performance  (R22)',
        '0.571 ms  |  22.58 TFLOPS  |  75.8% MFU\n'
        'vs Triton 1.44x  |  vs cuDNN 1.56x  |  vs FlashInfer 1.82x\n'
        'Baseline -> R22 :  16.571 ms -> 0.571 ms  =  29x speedup',
        tfont=10, bfont=8)

# ══════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════
legend_items = [
    (C_GLOBAL, 'Global Memory (HBM)'),
    (C_SMEM,   'Shared Memory (SRAM)'),
    (C_WMMA,   'WMMA Tensor Core'),
    (C_SCALAR, 'Scalar Compute'),
    (C_SKIP,   'Sparse Skip / Mask'),
    (C_OUT,    'Output'),
]
lx, ly = 0.5, 0.15
for i, (col, lbl) in enumerate(legend_items):
    bx = lx + i * 3.5
    rect = FancyBboxPatch((bx, ly-0.08), 0.35, 0.22,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          facecolor=col, edgecolor='white', linewidth=1, zorder=4)
    ax.add_patch(rect)
    ax.text(bx+0.45, ly+0.03, lbl, fontsize=8, va='center', color='#333', zorder=5)

plt.savefig('/home/wangjiawei/sparse-mask-attention/notes/kernel_diagram.png',
            dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: notes/kernel_diagram.png")
