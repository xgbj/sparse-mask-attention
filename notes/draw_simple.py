"""
Simple cartoon-style principle diagram for Sparse Mask Attention.
Clean top-down flow, no messy arrows.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

fig = plt.figure(figsize=(14, 16), facecolor='#FAFBFF')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14)
ax.set_ylim(0, 16)
ax.axis('off')

# ── helpers ────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, fc, ec, title, sub=None, tfont=13, sfont=9.5, r=0.3):
    # shadow
    s = FancyBboxPatch((x+0.08, y-0.08), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={r}",
                       facecolor='#BBBBBB', edgecolor='none', alpha=0.35, zorder=2)
    ax.add_patch(s)
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=2.5, zorder=3)
    ax.add_patch(b)
    ty = y + h/2 + (0.18 if sub else 0)
    ax.text(x+w/2, ty, title, ha='center', va='center',
            fontsize=tfont, fontweight='bold', color='white', zorder=4,
            path_effects=[pe.withStroke(linewidth=3, foreground='#00000055')])
    if sub:
        ax.text(x+w/2, y+h/2-0.28, sub, ha='center', va='center',
                fontsize=sfont, color='#FFFFFFCC', zorder=4)

def down_arrow(ax, x, y1, y2, color='#555', lw=2.5, label=None):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=20), zorder=5)
    if label:
        ax.text(x+0.15, (y1+y2)/2, label, fontsize=9, color=color,
                va='center', ha='left', zorder=6, fontstyle='italic')

def skip_arrow(ax, x1, y, x2, label, color='#C62828'):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                mutation_scale=18,
                                connectionstyle='arc3,rad=0'), zorder=5)
    ax.text((x1+x2)/2, y+0.18, label, ha='center', fontsize=9,
            color=color, fontweight='bold', zorder=6)

# ── color scheme ───────────────────────────────────────────────
BLUE   = '#1565C0'
ORANGE = '#E65100'
GREEN  = '#2E7D32'
PURPLE = '#6A1B9A'
RED    = '#C62828'
TEAL   = '#00695C'
GRAY   = '#455A64'

# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
ax.text(7, 15.55, 'Sparse Mask Attention', ha='center', va='center',
        fontsize=22, fontweight='bold', color='#1A1A2E', zorder=6)
ax.text(7, 15.1, 'CUDA Kernel  —  How it works',
        ha='center', va='center', fontsize=12, color='#556677', zorder=6)

# ══════════════════════════════════════════════════════════════
# STEP 0 — Inputs
# ══════════════════════════════════════════════════════════════
# Three input boxes side by side
for i, (lbl, col) in enumerate([('Q', BLUE), ('K', BLUE), ('V', BLUE)]):
    rbox(ax, 1.2 + i*2.5, 13.5, 2.0, 0.9, col, 'white', lbl,
         'Query / Key / Value', tfont=16, sfont=8.5, r=0.25)

# Mask box
rbox(ax, 9.0, 13.5, 3.8, 0.9, GRAY, 'white', 'Sparse Mask',
     'bool [B, H, N, N]', tfont=13, sfont=8.5, r=0.25)

# ── bit-pack arrow ─────────────────────────────────────────────
ax.annotate('', xy=(10.9, 12.85), xytext=(10.9, 13.5),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2.5,
                            mutation_scale=18), zorder=5)
ax.text(11.15, 13.15, 'bit-pack\n8x smaller', ha='left', fontsize=8.5,
        color=RED, zorder=6, fontweight='bold')

# ══════════════════════════════════════════════════════════════
# STEP 1 — Bit-packed mask
# ══════════════════════════════════════════════════════════════
rbox(ax, 8.2, 11.95, 5.4, 0.85, RED, 'white',
     'uint32 Packed Mask',
     'bool[N,N]  ->  uint32[N, N/32]', tfont=12, sfont=8.5, r=0.25)

# ── main down arrow ────────────────────────────────────────────
down_arrow(ax, 7, 13.5, 12.85, color='#888', lw=3)

# ══════════════════════════════════════════════════════════════
# STEP 2 — For each K/V tile
# ══════════════════════════════════════════════════════════════
# tile loop box (dashed border)
tile_box = FancyBboxPatch((0.5, 5.2), 13.0, 7.4,
                          boxstyle="round,pad=0.1,rounding_size=0.3",
                          facecolor='#F0F4FF', edgecolor='#8899BB',
                          linewidth=2, linestyle='--', alpha=0.7, zorder=1)
ax.add_patch(tile_box)
ax.text(7.0, 12.45, 'for each K/V tile  (N=512: 8 tiles of 64)',
        ha='center', va='center', fontsize=10, color='#445577',
        fontstyle='italic', zorder=4)

# ── SKIP check ─────────────────────────────────────────────────
rbox(ax, 1.0, 10.8, 12.0, 1.1, RED, 'white',
     'Skip Check  —  is this tile fully masked?',
     'read packed mask bits  ->  warp vote  ->  if ALL masked: skip entire tile',
     tfont=12, sfont=9, r=0.25)

# skip arrow to right
ax.annotate('', xy=(13.5, 11.35), xytext=(13.0, 11.35),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2.5,
                            mutation_scale=18), zorder=5)
ax.text(13.55, 11.35, 'SKIP', ha='left', va='center',
        fontsize=11, color=RED, fontweight='bold', zorder=6)

down_arrow(ax, 7, 10.8, 10.2, color='#888', lw=3)

# ══════════════════════════════════════════════════════════════
# STEP 3 — QK^T
# ══════════════════════════════════════════════════════════════
rbox(ax, 1.0, 9.1, 12.0, 1.05, GREEN, 'white',
     'QK^T  via Tensor Core  (WMMA)',
     'Q tile [16x64]  x  K tile [64x64]  ->  scores [16x64]',
     tfont=12, sfont=9, r=0.25)

down_arrow(ax, 7, 9.1, 8.5, color='#888', lw=3)

# ══════════════════════════════════════════════════════════════
# STEP 4 — Apply mask
# ══════════════════════════════════════════════════════════════
rbox(ax, 1.0, 7.4, 12.0, 1.05, ORANGE, 'white',
     'Apply Mask  to Scores',
     'masked positions  ->  score = -inf     unmasked  ->  keep score',
     tfont=12, sfont=9, r=0.25)

down_arrow(ax, 7, 7.4, 6.8, color='#888', lw=3)

# ══════════════════════════════════════════════════════════════
# STEP 5 — Softmax
# ══════════════════════════════════════════════════════════════
rbox(ax, 1.0, 5.7, 12.0, 1.05, PURPLE, 'white',
     'Online Softmax',
     'exp(score - max)  /  sum     running max & sum updated each tile',
     tfont=12, sfont=9, r=0.25)

down_arrow(ax, 7, 5.7, 5.1, color='#888', lw=3)

# ══════════════════════════════════════════════════════════════
# STEP 6 — PV
# ══════════════════════════════════════════════════════════════
rbox(ax, 1.0, 4.0, 12.0, 1.05, GREEN, 'white',
     'P x V  via Tensor Core  (WMMA)',
     'attention weights [16x64]  x  V tile [64x64]  ->  accumulate output',
     tfont=12, sfont=9, r=0.25)

# end of loop
ax.text(7.0, 3.65, 'end of tile loop', ha='center', va='center',
        fontsize=9.5, color='#445577', fontstyle='italic', zorder=4)

down_arrow(ax, 7, 3.9, 3.2, color='#888', lw=3)

# ══════════════════════════════════════════════════════════════
# STEP 7 — Output
# ══════════════════════════════════════════════════════════════
rbox(ax, 1.5, 2.1, 11.0, 1.05, TEAL, 'white',
     'Normalize  +  Write Output',
     'output[row]  =  accumulator / sum     ->  Out [B, H, N, 64]  FP16',
     tfont=12, sfont=9, r=0.25)

# ══════════════════════════════════════════════════════════════
# RESULT BADGE
# ══════════════════════════════════════════════════════════════
badge = FancyBboxPatch((2.5, 0.3), 9.0, 1.55,
                       boxstyle="round,pad=0.1,rounding_size=0.3",
                       facecolor='#1A237E', edgecolor='#5C6BC0',
                       linewidth=2.5, zorder=3)
ax.add_patch(badge)
ax.text(7.0, 1.5, 'Result  (RTX 3080, N=512, sparsity=75%)',
        ha='center', va='center', fontsize=10, color='#B3C5FF',
        fontweight='bold', zorder=4)
ax.text(7.0, 1.05, '0.571 ms   |   22.58 TFLOPS   |   29x faster than baseline',
        ha='center', va='center', fontsize=12, color='#FFD700',
        fontweight='bold', zorder=4)
ax.text(7.0, 0.62, 'vs Triton  1.44x     vs cuDNN  1.56x     vs FlashInfer  1.82x',
        ha='center', va='center', fontsize=10, color='#90CAF9', zorder=4)

# ══════════════════════════════════════════════════════════════
# STEP NUMBERS on left margin
# ══════════════════════════════════════════════════════════════
steps = [
    (11.35, '1', RED),
    (10.35, '2', RED),
    (9.63,  '3', GREEN),
    (7.93,  '4', ORANGE),
    (6.23,  '5', PURPLE),
    (4.53,  '6', GREEN),
    (2.63,  '7', TEAL),
]
for (yc, num, col) in steps:
    circ = plt.Circle((0.55, yc), 0.28, color=col, zorder=5)
    ax.add_patch(circ)
    ax.text(0.55, yc, num, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', zorder=6)

plt.savefig('/home/wangjiawei/sparse-mask-attention/notes/kernel_simple.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved: notes/kernel_simple.png")
